"""
RAG Service FastAPI Application
Provides endpoints for code retrieval, web indexing, and tool execution
"""

import os
import json
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, List, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
import validators
import httpx

from web_indexer import WebIndexer, CrawlJob
from agent import AgentTask, AgentServices, create_agent
from agent.runner import run_task
from agent.redis_session import RedisSession
from agent.session_store import SessionStore
from product_indexer import ProductIndexer, ProductSyncJob
from product_classifier import ProductClassifier, ClassificationJob
from translator import TranslationPipeline, TranslationJob, GlossaryStore
from glossary_data import GLOSSARY, SUPPORTED_LANGUAGES, get_glossary_for_pair
from shopify_client import ShopifyClient
from shopify_graphql import ShopifyGraphQL
from catalog_indexer import CatalogIndexer
from sync_state import SyncStateStore, ContentHashStore
from webhook_handler import ShopifyWebhookHandler
from compatibility_analyzer import CompatibilityAnalyzer
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from reranker import get_reranker, init_reranker
from vault_builder import VaultBuilder
from vault_indexer import VaultIndexer
from content_learner import ContentLearner
from knowledge_synthesizer import KnowledgeSynthesizer
from activity_logger import init_activity_logger, log_activity, read_timeline
from metrics import (
    PrometheusMiddleware, metrics_response,
    VAULT_NOTES_TOTAL, VAULT_RECOMMENDATION_NOTES, QDRANT_COLLECTION_POINTS,
    SYNTHESIS_RUNS, SYNTHESIS_NOTES_GENERATED, SYNTHESIS_LLM_CALLS, SYNTHESIS_DURATION,
    REBUILD_RUNS, REBUILD_DURATION, REBUILD_COMPETITORS_CRAWLED, REBUILD_PRODUCTS_EXTRACTED,
    SEARCH_COUNT, SEARCH_LATENCY,
)
from deep_analyzer import DeepAnalyzer
from firecrawl_client import FirecrawlClient
from research_agent import ResearchAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://litellm:4000/v1")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "none")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
SHOPIFY_SHOP_DOMAIN = os.getenv("SHOPIFY_SHOP_DOMAIN", "")
SHOPIFY_ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN", "")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SHOPIFY_WEBHOOK_SECRET = os.getenv("SHOPIFY_WEBHOOK_SECRET", "")

# Global instances
web_indexer: WebIndexer = None
competitor_indexer: WebIndexer = None
product_indexer: ProductIndexer = None
catalog_indexer: CatalogIndexer = None
product_classifier: ProductClassifier = None
translator: TranslationPipeline = None
glossary_store: GlossaryStore = None
shopify_client: ShopifyClient = None
shopify_graphql_client: ShopifyGraphQL = None
webhook_handler: ShopifyWebhookHandler = None
sync_state_store: SyncStateStore = None
content_hash_store: ContentHashStore = None
llm_client: OpenAI = None
LLM_MODEL = os.getenv("LLM_MODEL", "smart")
rag_agent = None  # Agent SDK instance, created at startup
redis_client = None  # Async Redis client for session persistence
session_store: SessionStore = None
vault_builder: VaultBuilder = None
vault_indexer_instance: VaultIndexer = None
content_learner: ContentLearner = None
knowledge_synthesizer: KnowledgeSynthesizer = None
compatibility_analyzer: CompatibilityAnalyzer = None

# Active crawl jobs + queue
crawl_jobs: Dict[str, CrawlJob] = {}
crawl_queue: List[str] = []  # ordered list of job_ids waiting to run
_crawl_running: bool = False

# Product sync jobs
product_sync_jobs: Dict[str, ProductSyncJob] = {}

# Classification jobs
classification_jobs: Dict[str, ClassificationJob] = {}

# Translation jobs
translation_jobs: Dict[str, TranslationJob] = {}

# Agent tasks
agent_tasks: Dict[str, AgentTask] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global web_indexer, competitor_indexer, llm_client
    global product_indexer, catalog_indexer, product_classifier, translator
    global shopify_client, shopify_graphql_client, webhook_handler
    global sync_state_store, content_hash_store, rag_agent
    global redis_client, session_store, glossary_store

    logger.info("Initializing RAG service components...")

    # Load shared embedding model once (~440MB)
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    web_indexer = WebIndexer(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        model=embedding_model,
    )
    competitor_indexer = WebIndexer(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        model=embedding_model,
        collection_name="competitor_products",
    )
    product_indexer = ProductIndexer(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        model=embedding_model,
    )
    catalog_indexer = CatalogIndexer(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        model=embedding_model,
    )
    llm_client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key=LITELLM_API_KEY,
        default_headers={"X-Source-Service": "pocharlies-rag"},
    )


    # Shopify client (optional)
    if SHOPIFY_SHOP_DOMAIN and SHOPIFY_ACCESS_TOKEN:
        shopify_client = ShopifyClient(
            shop_domain=SHOPIFY_SHOP_DOMAIN,
            access_token=SHOPIFY_ACCESS_TOKEN,
        )
        shopify_graphql_client = shopify_client.graphql
        logger.info(f"Shopify client configured for {SHOPIFY_SHOP_DOMAIN}")

    # Product classifier + translator (use LLM)
    product_classifier = ProductClassifier(
        qdrant_client=web_indexer.client,
        model=embedding_model,
        llm_client=llm_client,
    )
    # translator is initialized after Redis (needs glossary_store)

    # Create SDK agent (after llm_client is ready)
    try:
        rag_agent, _model_id = create_agent(VLLM_BASE_URL, LITELLM_API_KEY)
        logger.info(f"Agent SDK initialized with model: {_model_id}")
    except Exception as e:
        logger.warning(f"Agent SDK init failed (agent endpoints will fail): {e}")

    # Redis for agent session persistence + glossary
    try:
        from redis.asyncio import Redis as AsyncRedis
        redis_client = AsyncRedis.from_url(REDIS_URL, decode_responses=False)
        await redis_client.ping()
        session_store = SessionStore(redis_client)
        logger.info(f"Redis connected: {REDIS_URL}")
    except Exception as e:
        logger.warning(f"Redis connection failed ({e}) — agent sessions will not persist")
        redis_client = None
        session_store = None

    # Sync state + content hash dedup (Redis-backed)
    if redis_client:
        sync_state_store = SyncStateStore(redis_client)
        content_hash_store = ContentHashStore(redis_client)
        logger.info("Sync state + content hash stores initialized")

    # Webhook handler
    if shopify_client and SHOPIFY_WEBHOOK_SECRET:
        webhook_handler = ShopifyWebhookHandler(
            webhook_secret=SHOPIFY_WEBHOOK_SECRET,
            shopify_graphql=shopify_graphql_client,
            product_indexer=product_indexer,
            catalog_indexer=catalog_indexer,
            shopify_client=shopify_client,
            content_hash_store=content_hash_store,
        )
        logger.info("Shopify webhook handler configured")

    # Translation glossary (Redis-backed for custom terms, multi-language)
    glossary_store = GlossaryStore(redis=redis_client)
    if redis_client:
        custom_count = await glossary_store.load_all_pairs()
        logger.info(f"Glossary loaded: {len(GLOSSARY)} built-in terms × {len(SUPPORTED_LANGUAGES)} languages + {custom_count} custom entries")
    translator = TranslationPipeline(llm_client=llm_client, glossary_store=glossary_store)

    # Pre-load BM25 sparse encoder so first search is fast
    try:
        from sparse_encoder import get_bm25_model
        get_bm25_model()
        logger.info("BM25 sparse encoder loaded")
    except Exception as e:
        logger.warning(f"BM25 preload failed (will load on first search): {e}")

    # Reranker (optional, ~568MB model)
    if RERANKER_ENABLED:
        try:
            init_reranker(RERANKER_MODEL)
            logger.info(f"Reranker loaded: {RERANKER_MODEL}")
        except Exception as e:
            logger.warning(f"Reranker load failed (search will work without reranking): {e}")

    # Knowledge Vault (Obsidian brain)
    global vault_builder, vault_indexer_instance, content_learner, knowledge_synthesizer, compatibility_analyzer
    vault_path = os.getenv("VAULT_PATH", "/app/knowledge-vault")
    config_path = os.getenv("VAULT_CONFIG", "/app/vault_config.yaml")
    # Fallback to local paths for dev
    if not os.path.exists(vault_path):
        vault_path = os.path.join(os.path.dirname(__file__), "..", "knowledge-vault")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "..", "vault_config.yaml")

    init_activity_logger(vault_path)

    try:
        vault_builder = VaultBuilder(
            vault_path=vault_path,
            config_path=config_path,
            rag_base_url="http://localhost:5000",
            llm_client=llm_client,
            redis_client=redis_client,
            glossary=GLOSSARY,
            competitor_indexer=competitor_indexer,
            product_classifier=product_classifier,
            crawl_jobs=crawl_jobs,
            crawl_queue=crawl_queue,
        )
        vault_indexer_instance = VaultIndexer(
            vault_path=vault_path,
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY,
            model=embedding_model,
            redis_client=redis_client,
        )
        content_learner = ContentLearner(
            vault_path=vault_path,
            config_path=config_path,
            llm_client=llm_client,
            redis_client=redis_client,
            competitor_indexer=competitor_indexer,
        )
        logger.info(f"Knowledge vault initialized: {vault_path} ({len(content_learner.sources)} content sources)")

        # Knowledge Synthesizer (generates recommendation notes)
        knowledge_synthesizer = KnowledgeSynthesizer(
            vault_path=vault_path,
            llm_client=llm_client,
            product_indexer=product_indexer,
            catalog_indexer=catalog_indexer,
            vault_indexer=vault_indexer_instance,
            redis_client=redis_client,
        )
        logger.info("Knowledge synthesizer initialized")

        # Compatibility Analyzer (needs catalog_indexer + vault_indexer)
        compatibility_analyzer = CompatibilityAnalyzer(
            llm_client=llm_client,
            llm_model=LLM_MODEL,
            catalog_searcher=lambda query, top_k: catalog_indexer.search_pages(query=query, top_k=top_k),
            vault_searcher=lambda query, top_k: vault_indexer_instance.search_recommendations(query=query, top_k=top_k) if vault_indexer_instance else [],
        )
        logger.info("Compatibility analyzer initialized")

        # Wire compatibility analyzer into webhook handler (initialized earlier)
        if webhook_handler:
            webhook_handler.compatibility_analyzer = compatibility_analyzer

    except Exception as e:
        logger.warning(f"Knowledge vault init failed: {e}")

    # Daily rebuild scheduler
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    _rebuild_in_progress = False

    async def _daily_rebuild():
        """Sequential rebuild chain. Guarded against concurrent runs."""
        nonlocal _rebuild_in_progress
        if _rebuild_in_progress:
            logger.warning("Daily rebuild skipped — already in progress")
            return
        _rebuild_in_progress = True
        log_activity("rebuild_start", "Daily rebuild started")
        logger.info("Daily rebuild started")
        try:
            if vault_builder:
                await vault_builder.build_full()
            if content_learner:
                await content_learner.learn()
            if knowledge_synthesizer:
                await knowledge_synthesizer.synthesize()
            if vault_indexer_instance:
                await vault_indexer_instance.index_vault()
            log_activity("rebuild_complete", "Daily rebuild complete")
            logger.info("Daily rebuild complete")
        except Exception as e:
            logger.error(f"Daily rebuild failed: {e}")
        finally:
            _rebuild_in_progress = False

    rebuild_scheduler = AsyncIOScheduler()
    rebuild_scheduler.add_job(_daily_rebuild, CronTrigger(hour=3, minute=0))
    rebuild_scheduler.start()
    logger.info("Daily rebuild scheduler started (3:00 AM)")

    logger.info("RAG service initialized successfully")

    yield


app = FastAPI(
    title="Pocharlies Qdrant RAG Service",
    description="Web indexing, product catalog, and RAG pipelines",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(PrometheusMiddleware)

# Serve embedded dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")


# Request/Response Models
class ChatRequest(BaseModel):
    query: str
    model: Optional[str] = None  # optional override; defaults to LLM_MODEL
    use_rag: bool = True
    use_tools: bool = True
    max_tokens: int = 4096


class WebIndexRequest(BaseModel):
    url: str
    max_depth: int = 2
    max_pages: int = 50
    smart_mode: bool = True


class WebSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    domain: Optional[str] = None
    rerank: bool = True


class UnifiedSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    collection: str = "all"  # "all" | "web" | "products" | "competitors"
    domain: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    rerank: bool = True


class AgentTaskRequest(BaseModel):
    prompt: str


class AgentContinueRequest(BaseModel):
    message: str


class ProductSyncRequest(BaseModel):
    sync_type: str = "full"  # "full" | "incremental"
    since: Optional[str] = None  # ISO timestamp for incremental


class ProductSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    brand: Optional[str] = None
    category: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    rerank: bool = True
    # Compatibility filters
    compatible_with: Optional[str] = None
    upgrade_type: Optional[str] = None
    exclude_base_platforms: bool = False


class CompetitorCrawlRequest(BaseModel):
    url: str
    max_depth: int = 2
    max_pages: int = 50
    smart_mode: bool = True


class ClassifyRequest(BaseModel):
    domain: str
    collection: str = "competitor_products"  # which collection to extract from
    batch_size: int = 5


class EntityResolveRequest(BaseModel):
    domain: str


class TranslateRequest(BaseModel):
    texts: List[str]
    source_lang: str = "en"
    target_lang: str = "es"
    use_rag: bool = True
    resource_type: str = "product"


class GlossaryEntryRequest(BaseModel):
    source: str
    target: str
    source_lang: str = "en"
    target_lang: str = "es"


class GlossaryBulkRequest(BaseModel):
    entries: Dict[str, str]  # source_term -> target_term
    source_lang: str = "en"
    target_lang: str = "es"


# ── Dashboard ─────────────────────────────────────────────────

@app.get("/")
async def dashboard():
    """Serve the embedded RAG dashboard."""
    return FileResponse("static/index.html")


@app.get("/api")
async def api_info():
    return {
        "service": "Pocharlies Qdrant RAG Service",
        "version": "2.0.0",
        "endpoints": [
            "/health - Health check",
            "/retrieve - Retrieve code chunks",
            "/index - Index a repository",
            "/delete - Delete a repository from index",
            "/tool - Execute a tool",
            "/tools - Get available tools",
            "/chat - Chat with RAG context",
            "/web/index-url - Start web crawl job",
            "/web/status/{job_id} - SSE crawl progress",
            "/web/sources - List indexed web domains",
            "/web/source/{domain} - Delete web domain",
            "/web/search - Search web content",
            "/search - Unified hybrid search (cross-collection)",
            "/web/collections - Collection stats",
            "/web/jobs - List crawl jobs",
            "/web/jobs/{job_id}/stop - Stop a running/queued crawl job",
            "/web/jobs/{job_id}/resume - Resume a stopped crawl job",
            "/web/logs/{job_id} - Get crawl logs",
        ]
    }


# ── Health ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    qdrant_ok = False
    qdrant_version = None
    qdrant_collections = 0
    try:
        import httpx as _httpx
        headers = {}
        if QDRANT_API_KEY:
            headers["api-key"] = QDRANT_API_KEY
        resp = _httpx.get(f"{QDRANT_URL}/collections", headers=headers, timeout=3.0)
        if resp.status_code == 200:
            qdrant_ok = True
            qdrant_collections = len(resp.json().get("result", {}).get("collections", []))
        # Get Qdrant version
        resp_info = _httpx.get(f"{QDRANT_URL}/", headers=headers, timeout=3.0)
        if resp_info.status_code == 200:
            qdrant_version = resp_info.json().get("version")
    except Exception:
        pass

    # Get per-collection stats from all known collections
    collection_stats = {}
    if qdrant_ok:
        try:
            collection_stats = {c["name"]: c for c in web_indexer.get_collection_stats()}
        except Exception:
            pass
        # Add collections from other indexers not covered by web_indexer
        import httpx as _httpx2
        headers2 = {}
        if QDRANT_API_KEY:
            headers2["api-key"] = QDRANT_API_KEY
        all_colls_resp = resp.json().get("result", {}).get("collections", [])
        for coll_info in all_colls_resp:
            cname = coll_info.get("name", "")
            if cname and cname not in collection_stats:
                try:
                    cr = _httpx2.get(f"{QDRANT_URL}/collections/{cname}", headers=headers2, timeout=3.0)
                    if cr.status_code == 200:
                        cdata = cr.json().get("result", {})
                        vec_count = cdata.get("vectors_count")
                        if vec_count is None:
                            vec_count = cdata.get("indexed_vectors_count", 0)
                        collection_stats[cname] = {
                            "name": cname,
                            "points_count": cdata.get("points_count", 0),
                            "vectors_count": vec_count or 0,
                            "status": cdata.get("status", "unknown"),
                        }
                except Exception:
                    pass

    return {
        "status": "healthy" if qdrant_ok else "degraded",
        "service": "pocharlies-qdrant-rag",
        "qdrant": {
            "status": "healthy" if qdrant_ok else "unreachable",
            "url": QDRANT_URL,
            "version": qdrant_version,
            "collections_count": qdrant_collections,
            "collections": collection_stats,
        },
        "redis": {
            "status": "healthy" if redis_client else "not configured",
        },
        "services": {
            "reranker": RERANKER_ENABLED and get_reranker() is not None,
            "shopify": shopify_client is not None,
            "product_indexer": product_indexer is not None,
        },
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    # Update gauge metrics on each scrape
    try:
        if vault_indexer_instance:
            info = vault_indexer_instance.get_collection_info()
            if info:
                VAULT_NOTES_TOTAL.set(info.get("points_count", 0))
        if product_indexer:
            for col in product_indexer.client.get_collections().collections:
                try:
                    col_info = product_indexer.client.get_collection(col.name)
                    QDRANT_COLLECTION_POINTS.labels(collection=col.name).set(col_info.points_count)
                except Exception:
                    pass
    except Exception:
        pass
    return metrics_response()


@app.post("/chat")
async def chat_with_rag(request: ChatRequest):
    """Chat endpoint with RAG context injection and tool support."""
    model = request.model or LLM_MODEL

    messages = []

    system_prompt = """You are a knowledgeable assistant with access to indexed content from multiple sources:
- **Product Catalog**: Products from the Shopify store (titles, descriptions, prices, brands)
- **Web Pages**: Crawled website content from various indexed sites
- **Competitor Products**: Product data from competitor stores

When answering:
1. Use the provided RAG context to give factual, data-driven answers
2. Cite sources (domain, product name, collection) when referencing specific data
3. If comparing prices or products, present the data in a clear table or list
4. If context is insufficient, say what data might be missing and suggest indexing more sources
5. Be specific — use exact names, prices, and URLs from the context"""

    messages.append({"role": "system", "content": system_prompt})

    if request.use_rag:
        context_sections = []

        # Search across ALL collections for relevant context
        try:
            # Product catalog
            prod_results = product_indexer.search(query=request.query, top_k=5)
            if prod_results:
                prod_block = "\n".join([
                    f"- [{r.get('title', 'Unknown')}]({r.get('url', '')}) | Price: {r.get('price', 'N/A')} | Brand: {r.get('brand', 'N/A')} | SKU: {r.get('sku', '')} | {r.get('text', '')[:200]}"
                    for r in prod_results
                ])
                context_sections.append(f"## Product Catalog Results\n{prod_block}")
        except Exception as e:
            logger.warning(f"Product search failed in chat: {e}")

        try:
            # Web pages (crawled content)
            web_results = web_indexer.search(query=request.query, top_k=5)
            if web_results:
                web_block = "\n".join([
                    f"- [{r.get('title', 'Untitled')}]({r.get('url', '')}) ({r.get('domain', 'unknown')})\n  {r.get('text', '')[:300]}"
                    for r in web_results
                ])
                context_sections.append(f"## Web Page Results\n{web_block}")
        except Exception as e:
            logger.warning(f"Web search failed in chat: {e}")

        try:
            # Competitor products
            comp_results = competitor_indexer.search(query=request.query, top_k=5)
            if comp_results:
                comp_block = "\n".join([
                    f"- [{r.get('title', 'Untitled')}]({r.get('url', '')}) ({r.get('domain', 'unknown')})\n  {r.get('text', '')[:300]}"
                    for r in comp_results
                ])
                context_sections.append(f"## Competitor Store Results\n{comp_block}")
        except Exception as e:
            logger.warning(f"Competitor search failed in chat: {e}")

        if context_sections:
            full_context = "\n\n".join(context_sections)
            messages.append({
                "role": "user",
                "content": f"Here is relevant context from indexed sources:\n\n{full_context}\n\n---\nQUESTION: {request.query}"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"No relevant indexed content found for this query.\n\nQUESTION: {request.query}"
            })
    else:
        messages.append({"role": "user", "content": request.query})

    # Log context stats for debugging
    user_msg = messages[-1]["content"] if messages else ""
    context_len = len(user_msg)
    has_context = "RAG context" in user_msg or "indexed sources" in user_msg
    logger.info(f"Chat: query='{request.query[:60]}', model={model}, context={context_len} chars, has_rag={has_context}")

    try:
        call_kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": 0.2,
            "user": "rag:chat",
        }

        response = llm_client.chat.completions.create(**call_kwargs)

        message = response.choices[0].message

        return {
            "response": message.content,
            "tools_used": [],
                "model": model
            }

        return {
            "response": message.content,
            "tools_used": [],
            "model": model
        }
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Web Indexing Endpoints ──────────────────────────────────────


def _cleanup_old_jobs():
    """Remove completed/failed jobs older than 1 hour, stopped jobs older than 4 hours."""
    now = datetime.now(timezone.utc)
    stale = []
    for jid, job in crawl_jobs.items():
        if job.status in ("completed", "failed", "stopped") and job.ended_at:
            try:
                ended = datetime.fromisoformat(job.ended_at)
                max_age = 14400 if job.status == "stopped" else 3600
                if (now - ended).total_seconds() > max_age:
                    stale.append(jid)
            except (ValueError, TypeError):
                pass
    for jid in stale:
        del crawl_jobs[jid]
    if stale:
        logger.info(f"Cleaned up {len(stale)} old crawl jobs")


async def _process_crawl_queue():
    """Process crawl jobs one at a time from the queue."""
    global _crawl_running
    if _crawl_running:
        return
    _crawl_running = True

    try:
        while crawl_queue:
            _cleanup_old_jobs()

            job_id = crawl_queue[0]
            job = crawl_jobs.get(job_id)
            if not job:
                crawl_queue.pop(0)
                continue

            def progress_cb(j: CrawlJob):
                crawl_jobs[job_id] = j
                j.job_id = job_id

            # Build resume state if this is a resumed job
            initial_visited = None
            initial_bfs_queue = None
            if job.resumed_from:
                old_job = crawl_jobs.get(job.resumed_from)
                if old_job and old_job.visited_urls:
                    from collections import deque
                    initial_visited = set(old_job.visited_urls)
                    initial_bfs_queue = deque(
                        [(url, depth) for url, depth in old_job.pending_bfs_queue]
                    )
                    job.log(f"Resume state loaded: {len(initial_visited)} visited URLs, {len(initial_bfs_queue)} pending in BFS queue")

            # Use competitor_indexer for jobs tagged as competitor crawls
            active_indexer = competitor_indexer if getattr(job, '_competitor', False) else web_indexer

            result = await active_indexer.crawl_and_index(
                start_url=job.url,
                max_depth=job.max_depth,
                max_pages=job.max_pages,
                progress_callback=progress_cb,
                smart_mode=getattr(job, 'smart_mode', True),
                llm_client=llm_client,
                initial_visited=initial_visited,
                initial_bfs_queue=initial_bfs_queue,
            )
            result.job_id = job_id
            result.resumed_from = job.resumed_from
            crawl_jobs[job_id] = result

            crawl_queue.pop(0)
    finally:
        _crawl_running = False


@app.post("/web/index-url")
async def web_index_url(request: WebIndexRequest):
    """Start or queue a web crawl job. Returns job_id for progress tracking."""
    if not validators.url(request.url):
        raise HTTPException(status_code=400, detail=f"Invalid URL: {request.url}")

    if request.max_depth < 0 or request.max_depth > 5:
        raise HTTPException(status_code=400, detail="max_depth must be 0-5")

    if request.max_pages < 1 or request.max_pages > 25000:
        raise HTTPException(status_code=400, detail="max_pages must be 1-25000")

    import uuid
    job_id = uuid.uuid4().hex[:12]

    placeholder_job = CrawlJob(
        job_id=job_id,
        url=request.url,
        max_depth=request.max_depth,
        max_pages=request.max_pages,
        status="queued",
        smart_mode=request.smart_mode,
    )
    crawl_jobs[job_id] = placeholder_job
    crawl_queue.append(job_id)

    queue_pos = len(crawl_queue)
    logger.info(f"Queued crawl job {job_id} for {request.url} (position {queue_pos})")

    asyncio.create_task(_process_crawl_queue())

    return {
        "job_id": job_id,
        "status": "queued",
        "url": request.url,
        "queue_position": queue_pos,
    }


@app.get("/web/status/{job_id}")
async def web_crawl_status(job_id: str):
    """SSE stream of crawl job progress."""
    if job_id not in crawl_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    async def event_stream():
        prev_data = None
        while True:
            job = crawl_jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'Job disappeared'})}\n\n"
                break

            data = json.dumps(job.to_dict())
            if data != prev_data:
                yield f"data: {data}\n\n"
                prev_data = data

            if job.status in ("completed", "failed", "stopped"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/web/sources")
async def web_sources():
    """List indexed web sources grouped by domain."""
    return {"sources": web_indexer.get_sources()}


@app.delete("/web/source/{domain:path}")
async def web_delete_source(domain: str):
    """Delete all indexed content from a domain."""
    success = web_indexer.delete_source(domain)
    if success:
        return {"status": "deleted", "domain": domain}
    raise HTTPException(status_code=500, detail=f"Failed to delete domain: {domain}")


@app.post("/web/search")
async def web_search(request: WebSearchRequest):
    """Search web_pages collection."""
    reranker = get_reranker()
    effective_top_k = request.top_k * 4 if (reranker and request.rerank) else request.top_k

    results = web_indexer.search(
        query=request.query,
        top_k=effective_top_k,
        domain_filter=request.domain,
    )

    if reranker and request.rerank and results:
        results = reranker.rerank(request.query, results, top_k=request.top_k)

    return {"results": results, "query": request.query}


def _cross_collection_rrf(results: List[dict], top_k: int, k: int = 60) -> List[dict]:
    """Client-side RRF to merge results from multiple collections."""
    for i, r in enumerate(results):
        r["_rrf_score"] = 1.0 / (k + i + 1)

    # Group by collection, re-rank within each group by original order
    by_collection: Dict[str, List[dict]] = {}
    for r in results:
        col = r.get("collection", "unknown")
        by_collection.setdefault(col, []).append(r)

    # Assign per-collection RRF ranks
    merged = []
    for col, items in by_collection.items():
        for rank, item in enumerate(items):
            item["_rrf_score"] = 1.0 / (k + rank + 1)
            merged.append(item)

    # Sort by RRF score descending, take top_k
    merged.sort(key=lambda x: x["_rrf_score"], reverse=True)
    merged = merged[:top_k]

    # Normalize scores: top result = 1.0
    if merged:
        max_score = merged[0]["_rrf_score"]
        for r in merged:
            r["score"] = round(r["_rrf_score"] / max_score, 4) if max_score > 0 else 0
            del r["_rrf_score"]

    return merged


@app.post("/search")
async def unified_search(request: UnifiedSearchRequest):
    """Unified hybrid search across all collections."""
    results = []

    if request.collection in ("all", "web"):
        try:
            web_results = web_indexer.search(
                query=request.query,
                top_k=request.top_k,
                domain_filter=request.domain,
            )
            for r in web_results:
                r["collection"] = "web_pages"
                r["source_type"] = "web"
            results.extend(web_results)
        except Exception as e:
            logger.warning(f"Web search failed: {e}")

    if request.collection in ("all", "products"):
        try:
            prod_results = product_indexer.search(
                query=request.query,
                top_k=request.top_k,
                brand_filter=request.brand,
                category_filter=request.category,
            )
            for r in prod_results:
                r["collection"] = "product_catalog"
            results.extend(prod_results)
        except Exception as e:
            logger.warning(f"Product search failed: {e}")

    if request.collection in ("all", "competitors"):
        try:
            comp_results = competitor_indexer.search(
                query=request.query,
                top_k=request.top_k,
                domain_filter=request.domain,
            )
            for r in comp_results:
                r["collection"] = "competitor_products"
                r["source_type"] = "competitor"
            results.extend(comp_results)
        except Exception as e:
            logger.warning(f"Competitor search failed: {e}")

    if request.collection == "all" and results:
        results = _cross_collection_rrf(results, request.top_k)

    # Rerank merged results
    reranker = get_reranker()
    if reranker and request.rerank and results:
        results = reranker.rerank(request.query, results, top_k=request.top_k)

    return {"results": results, "query": request.query, "collection": request.collection}


@app.get("/web/collections")
async def web_collections():
    """Get stats for all RAG collections."""
    return {"collections": web_indexer.get_collection_stats()}


@app.post("/web/jobs/{job_id}/stop")
async def web_stop_job(job_id: str):
    """Stop a running or queued crawl job."""
    job = crawl_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.status == "queued":
        if job_id in crawl_queue:
            crawl_queue.remove(job_id)
        job.status = "stopped"
        job.ended_at = datetime.now(timezone.utc).isoformat()
        job.log("Job cancelled while still queued")
        return {"status": "stopped", "job_id": job_id, "message": "Queued job cancelled"}

    if job.status == "running":
        job.cancel_requested = True
        job.log("Stop requested — will halt after current page completes")
        return {"status": "stopping", "job_id": job_id, "message": "Stop signal sent, job will halt gracefully"}

    raise HTTPException(
        status_code=400,
        detail=f"Job is already {job.status}, cannot stop"
    )


@app.post("/web/jobs/{job_id}/resume")
async def web_resume_job(job_id: str):
    """Resume a previously stopped crawl job."""
    old_job = crawl_jobs.get(job_id)
    if not old_job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if old_job.status != "stopped":
        raise HTTPException(
            status_code=400,
            detail=f"Only stopped jobs can be resumed. Job is: {old_job.status}"
        )

    import uuid
    new_job_id = uuid.uuid4().hex[:12]

    new_job = CrawlJob(
        job_id=new_job_id,
        url=old_job.url,
        max_depth=old_job.max_depth,
        max_pages=old_job.max_pages,
        status="queued",
        smart_mode=old_job.smart_mode,
        resumed_from=job_id,
        pages_scraped=old_job.pages_scraped,
        pages_visited=old_job.pages_visited,
        pages_indexed=old_job.pages_indexed,
        chunks_indexed=old_job.chunks_indexed,
        chunks_queued=old_job.chunks_queued,
    )
    new_job.log(f"Resumed from job {job_id} ({old_job.pages_visited} pages already visited)")

    crawl_jobs[new_job_id] = new_job
    crawl_queue.append(new_job_id)

    queue_pos = len(crawl_queue)
    logger.info(f"Resume job {new_job_id} from {job_id} for {old_job.url} (position {queue_pos})")

    asyncio.create_task(_process_crawl_queue())

    return {
        "job_id": new_job_id,
        "resumed_from": job_id,
        "status": "queued",
        "url": old_job.url,
        "queue_position": queue_pos,
        "pages_already_visited": old_job.pages_visited,
    }


@app.get("/web/jobs")
async def web_jobs():
    """List all crawl jobs with queue position."""
    jobs = []
    for job in crawl_jobs.values():
        d = job.to_dict()
        if job.job_id in crawl_queue:
            d["queue_position"] = crawl_queue.index(job.job_id) + 1
        else:
            d["queue_position"] = None
        jobs.append(d)
    return {"jobs": jobs}


@app.get("/web/logs/{job_id}")
async def web_logs(job_id: str, offset: int = 0):
    """Get logs for a crawl job. Use offset to paginate."""
    job = crawl_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    logs = job.logs[offset:]
    return {
        "job_id": job_id,
        "logs": logs,
        "total": len(job.logs),
        "offset": offset,
        "status": job.status,
    }


# ── Product Catalog Endpoints ─────────────────────────────────


@app.post("/products/sync")
async def product_sync(request: ProductSyncRequest):
    """Start a product catalog sync from Shopify (runs in background)."""
    if not shopify_client:
        raise HTTPException(
            status_code=400,
            detail="Shopify not configured. Set SHOPIFY_SHOP_DOMAIN and SHOPIFY_ACCESS_TOKEN.",
        )

    import uuid as _uuid
    job_id = _uuid.uuid4().hex[:12]

    # Create a placeholder job so the frontend can poll immediately
    placeholder = ProductSyncJob(
        job_id=job_id,
        sync_type=request.sync_type or "full",
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    product_sync_jobs[job_id] = placeholder

    def progress_cb(j: ProductSyncJob):
        product_sync_jobs[j.job_id] = j

    async def _run_sync():
        try:
            if request.sync_type == "incremental" and request.since:
                job = await product_indexer.incremental_sync(
                    shopify_client, request.since,
                    progress_callback=progress_cb,
                    content_hash_store=content_hash_store,
                )
            else:
                job = await product_indexer.index_all_products(
                    shopify_client,
                    progress_callback=progress_cb,
                    content_hash_store=content_hash_store,
                )
            product_sync_jobs[job.job_id] = job
        except Exception as e:
            placeholder.status = "failed"
            placeholder.errors.append(str(e)[:200])
            placeholder.log(f"FAILED: {str(e)[:200]}")

    asyncio.create_task(_run_sync())

    return {"job_id": job_id, "status": "running", "sync_type": placeholder.sync_type}


@app.get("/products/sync/history")
async def sync_history():
    """Get sync state history."""
    if not sync_state_store:
        return {"history": [], "message": "Redis not available"}
    history = await sync_state_store.get_sync_history(limit=10)
    return {"history": history}


@app.get("/products/sync/{job_id}")
async def product_sync_status(job_id: str):
    """Get product sync job status."""
    job = product_sync_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_dict()


@app.post("/products/search")
async def product_search(request: ProductSearchRequest):
    """Search product catalog with structured filters."""
    reranker = get_reranker()
    effective_top_k = request.top_k * 4 if (reranker and request.rerank) else request.top_k

    results = product_indexer.search(
        query=request.query,
        top_k=effective_top_k,
        brand_filter=request.brand,
        category_filter=request.category,
        min_price=request.min_price,
        max_price=request.max_price,
        compatible_with=request.compatible_with,
        upgrade_type=request.upgrade_type,
        exclude_base_platforms=request.exclude_base_platforms,
    )

    if reranker and request.rerank and results:
        results = reranker.rerank(request.query, results, top_k=request.top_k)

    return {"results": results, "query": request.query}


@app.get("/products/stats")
async def product_stats():
    """Get product catalog stats."""
    stats = product_indexer.get_stats()
    if catalog_indexer:
        stats["catalog"] = catalog_indexer.get_stats()
    return stats


# ── New Catalog Endpoints ─────────────────────────────────────


class CollectionSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class CatalogSearchRequest(BaseModel):
    query: str
    types: Optional[List[str]] = None  # ["product", "collection", "page"]
    top_k: int = 10
    brand: Optional[str] = None
    category: Optional[str] = None
    # Compatibility filters
    compatible_with: Optional[str] = None
    upgrade_type: Optional[str] = None
    exclude_base_platforms: bool = False


class FullSyncRequest(BaseModel):
    sync_type: str = "full"  # "full" | "incremental"


@app.post("/collections/search")
async def search_collections(request: CollectionSearchRequest):
    """Semantic search on product collections."""
    if not catalog_indexer:
        raise HTTPException(status_code=500, detail="Catalog indexer not initialized")
    results = catalog_indexer.search_collections(request.query, top_k=request.top_k)
    return {"results": results, "query": request.query}


@app.get("/collections/{id_or_handle}/products")
async def get_collection_products(id_or_handle: str, limit: int = 20):
    """List products belonging to a collection."""
    if not catalog_indexer:
        raise HTTPException(status_code=500, detail="Catalog indexer not initialized")
    products = catalog_indexer.get_collection_products(product_indexer, id_or_handle, limit=limit)
    return {"collection": id_or_handle, "total_products": len(products), "products": products}


@app.get("/products/{id_or_handle}")
async def get_product_detail(id_or_handle: str):
    """Get full product detail by GID or handle."""
    if not catalog_indexer:
        raise HTTPException(status_code=500, detail="Catalog indexer not initialized")
    product = catalog_indexer.get_product_by_id_or_handle(product_indexer, id_or_handle)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@app.get("/products/{id_or_handle}/inventory")
async def get_product_inventory(id_or_handle: str):
    """Get inventory/stock levels for a product."""
    if not catalog_indexer:
        raise HTTPException(status_code=500, detail="Catalog indexer not initialized")
    inventory = catalog_indexer.get_inventory(product_indexer, id_or_handle)
    if not inventory:
        raise HTTPException(status_code=404, detail="Product not found")
    return inventory


# Spanish→English keyword map for cross-language page/guide search.
# Guides are written in English; customer queries are often in Spanish.
_ES_EN_KEYWORDS = {
    "mejorar": "upgrade improve",
    "mejora": "upgrade improvement",
    "mejoras": "upgrades improvements",
    "actualizar": "upgrade update",
    "guia": "guide",
    "guía": "guide",
    "rendimiento": "performance",
    "precisión": "accuracy precision",
    "precision": "accuracy precision",
    "alcance": "range",
    "potencia": "power fps joules",
    "silencioso": "silent quiet suppressor",
    "silenciador": "suppressor",
    "muelle": "spring",
    "cañón": "barrel inner barrel",
    "cañon": "barrel inner barrel",
    "pistón": "piston",
    "piston": "piston",
    "cilindro": "cylinder",
    "hop": "hop-up hop up",
    "goma": "bucking rubber",
    "gatillo": "trigger",
    "cargador": "magazine",
    "mira": "scope sight optic",
    "visor": "scope sight optic",
    "culata": "stock",
    "pistola": "pistol handgun",
    "fusil": "rifle",
    "escopeta": "shotgun",
    "francotirador": "sniper",
    "mantenimiento": "maintenance",
    "limpiar": "clean maintenance",
    "desarmar": "disassemble",
    "montar": "assemble install",
    "instalar": "install installation",
    "mejor": "best recommended",
    "recomendar": "recommend",
    "recomendación": "recommendation",
    "comprar": "buy purchase",
    "comparar": "compare comparison",
    "diferencia": "difference versus",
}


def _augment_query_for_pages(query: str) -> str:
    """Augment a (possibly Spanish) query with English keywords for page search."""
    words = query.lower().split()
    additions = set()
    for w in words:
        en = _ES_EN_KEYWORDS.get(w)
        if en:
            additions.update(en.split())
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


# ── Query intent extraction for auto-filtering ──────────────────────
# Maps query keywords → Qdrant category values (same categories as shopify_client.py)
_CATEGORY_KEYWORDS: list[tuple[list[str], str]] = [
    (["gbb", "gas blowback", "gas blow back"], "gbb"),
    (["aeg", "electric", "electrica", "automatica"], "aeg"),
    (["sniper", "bolt action", "bolt-action", "cerrojo", "francotirador"], "sniper"),
    (["pistol", "pistola", "handgun", "sidearm"], "pistol"),
    (["shotgun", "escopeta"], "shotgun"),
    (["smg", "submachine", "subfusil"], "smg"),
    (["grenade", "launcher", "lanzagranadas", "granada"], "launcher"),
    (["magazine", "cargador de bolas", "mag ", "hi-cap", "mid-cap", "low-cap"], "magazine"),
    (["battery", "bateria", "batería", "charger", "cargador bater"], "battery"),
    (["scope", "sight", "optic", "mira", "visor", "red dot", "holographic"], "optic"),
    (["bb", "bbs", "bola", "bolas", "municion", "munición", "ammo", "ammunition"], "ammunition"),
    (["vest", "plate carrier", "chaleco", "tactical gear"], "gear"),
    (["mask", "goggle", "gafas", "proteccion", "protección", "protection", "gafa"], "protection"),
]

# Brand aliases: query terms → canonical brand name in Qdrant
_BRAND_KEYWORDS: dict[str, str] = {
    "nuprol": "Nuprol",
    "specna": "Specna Arms", "specna arms": "Specna Arms",
    "silverback": "Silverback", "srs": "Silverback",
    "novritsch": "Novritsch",
    "tokyo marui": "Tokyo Marui", "marui": "Tokyo Marui",
    "g&g": "G&G", "g and g": "G&G",
    "cyma": "CYMA",
    "modify": "Modify",
    "action army": "Action Army",
    "maple leaf": "Maple Leaf",
    "ares": "Ares", "amoeba": "Ares",
    "vfc": "VFC",
    "krytac": "Krytac",
    "lct": "LCT",
    "ics": "ICS",
    "asg": "ASG",
    "we": "WE",
    "kjw": "KJW",
    "classic army": "Classic Army",
    "jag arms": "JAG Arms",
    "valken": "Valken",
    "warrior assault": "Warrior Assault Systems",
    "emerson": "Emerson",
    "invader gear": "Invader Gear",
}


def _extract_search_intent(query: str) -> dict:
    """Extract category and brand intent from a natural-language query.

    Returns dict with optional 'category' and 'brand' keys.
    Only sets a value when there's a clear keyword match.
    """
    lower = query.lower()
    result = {}

    # Extract category
    for keywords, category in _CATEGORY_KEYWORDS:
        for kw in keywords:
            # Use word boundary-ish matching to avoid false positives
            # e.g. "bb" in "rubber" would be wrong, but "bb" or "bbs" standalone is fine
            if f" {kw}" in f" {lower}" or lower.startswith(kw):
                result["category"] = category
                break
        if "category" in result:
            break

    # Extract brand (check longer phrases first to avoid partial matches)
    for alias in sorted(_BRAND_KEYWORDS.keys(), key=len, reverse=True):
        if alias in lower:
            result["brand"] = _BRAND_KEYWORDS[alias]
            break

    return result


@app.post("/catalog/search")
async def catalog_search(request: CatalogSearchRequest):
    """Unified search across products, collections, and pages.

    Uses a balanced merge strategy: each type gets a guaranteed minimum
    number of slots so that high-scoring products don't push out
    relevant guide/page chunks.

    Auto-extracts category/brand from query when not explicitly provided.
    """
    import time as _time
    _search_start = _time.time()
    types = request.types or ["product", "collection", "page"]
    results_by_type: dict[str, list] = {}

    # Auto-extract filters from query when not explicitly set
    intent = _extract_search_intent(request.query)
    effective_brand = request.brand or intent.get("brand")
    effective_category = request.category or intent.get("category")

    if intent:
        logger.info(f"Query intent extracted: {intent} (effective brand={effective_brand}, category={effective_category})")

    if "product" in types:
        reranker = get_reranker()
        effective_top_k = request.top_k * 4 if reranker else request.top_k
        products = product_indexer.search(
            query=request.query,
            top_k=effective_top_k,
            brand_filter=effective_brand,
            category_filter=effective_category,
            compatible_with=request.compatible_with,
            upgrade_type=request.upgrade_type,
            exclude_base_platforms=request.exclude_base_platforms,
        )
        if reranker and products:
            products = reranker.rerank(request.query, products, top_k=request.top_k)
        results_by_type["product"] = products

    if "collection" in types and catalog_indexer:
        results_by_type["collection"] = catalog_indexer.search_collections(request.query, top_k=request.top_k)

    if "page" in types and catalog_indexer:
        # Augment query with English keywords for cross-language guide search
        page_query = _augment_query_for_pages(request.query)
        results_by_type["page"] = catalog_indexer.search_pages(page_query, top_k=request.top_k)

    if "recommendation" in types and vault_indexer_instance:
        results_by_type["recommendation"] = vault_indexer_instance.search_recommendations(
            query=request.query, top_k=request.top_k
        )

    # Balanced merge: guarantee min slots per type, fill remainder by score
    active_types = [t for t in types if results_by_type.get(t)]
    if not active_types:
        return {"results": [], "query": request.query, "types": types}

    min_per_type = max(1, request.top_k // (len(active_types) + 1))
    guaranteed = []
    remaining_pool = []

    for t in active_types:
        items = results_by_type[t]
        guaranteed.extend(items[:min_per_type])
        remaining_pool.extend(items[min_per_type:])

    # Fill remaining slots from the pool sorted by score
    remaining_pool.sort(key=lambda r: r.get("score", 0), reverse=True)
    slots_left = request.top_k - len(guaranteed)
    all_results = guaranteed + remaining_pool[:max(0, slots_left)]

    # Final sort by score for presentation
    all_results.sort(key=lambda r: r.get("score", 0), reverse=True)

    SEARCH_COUNT.labels(search_type="catalog").inc()
    SEARCH_LATENCY.labels(search_type="catalog").observe(_time.time() - _search_start)
    return {"results": all_results, "query": request.query, "types": types}


@app.post("/catalog/full-sync")
async def catalog_full_sync(request: FullSyncRequest):
    """Full catalog sync using GraphQL bulk operations: products -> collections -> pages."""
    if not shopify_client:
        raise HTTPException(status_code=400, detail="Shopify not configured")
    if not shopify_graphql_client:
        raise HTTPException(status_code=400, detail="GraphQL client not initialized")

    sync_id = None
    if sync_state_store:
        sync_id = await sync_state_store.create_sync(request.sync_type)

    async def _run_full_sync():
        try:
            gql = shopify_graphql_client
            items_total = 0

            if request.sync_type == "full":
                # 1. Bulk products
                logger.info("Starting bulk products export...")
                op_id = await gql.start_products_bulk_query()
                result = await gql.poll_until_complete(op_id)
                if result.get("url"):
                    raw_items = await gql.download_results(result["url"])
                    products = [ShopifyGraphQL.flatten_graphql_product(item)
                                for item in raw_items if item.get("id", "").startswith("gid://shopify/Product/")]
                    if products:
                        job = await product_indexer.index_all_products(
                            shopify_client, content_hash_store=content_hash_store
                        )
                        items_total += job.products_indexed
                    logger.info(f"Bulk products: {len(products)} items")

                # 2. Bulk collections (Shopify allows one bulk op at a time)
                logger.info("Starting bulk collections export...")
                op_id = await gql.start_collections_bulk_query()
                result = await gql.poll_until_complete(op_id)
                if result.get("url"):
                    raw_items = await gql.download_results(result["url"])
                    collections = [ShopifyGraphQL.flatten_graphql_collection(item) for item in raw_items]
                    if collections and catalog_indexer:
                        stats = await catalog_indexer.index_collections(
                            collections, shopify_client, content_hash_store
                        )
                        items_total += stats.get("indexed", 0)
                    logger.info(f"Bulk collections: {len(collections)} items")

                # 3. Bulk pages
                logger.info("Starting bulk pages export...")
                op_id = await gql.start_pages_bulk_query()
                result = await gql.poll_until_complete(op_id)
                if result.get("url"):
                    raw_items = await gql.download_results(result["url"])
                    pages = [ShopifyGraphQL.flatten_graphql_page(item) for item in raw_items]
                    if pages and catalog_indexer:
                        stats = await catalog_indexer.index_pages(
                            pages, shopify_client, content_hash_store
                        )
                        items_total += stats.get("indexed", 0)
                    logger.info(f"Bulk pages: {len(pages)} items")

            else:
                # Incremental: use cursor from last sync
                since = None
                if sync_state_store:
                    since = await sync_state_store.get_last_cursor()
                if not since:
                    from datetime import timedelta
                    since = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

                products = await gql.fetch_products_updated_since(since)
                if products:
                    flat_products = [ShopifyGraphQL.flatten_graphql_product(p) for p in products]
                    job = await product_indexer.incremental_sync(
                        shopify_client, since, content_hash_store=content_hash_store,
                    )
                    items_total += job.products_indexed

                collections = await gql.fetch_collections_updated_since(since)
                if collections:
                    flat_colls = [ShopifyGraphQL.flatten_graphql_collection(c) for c in collections]
                    if catalog_indexer:
                        stats = await catalog_indexer.index_collections(
                            flat_colls, shopify_client, content_hash_store
                        )
                        items_total += stats.get("indexed", 0)

            cursor = datetime.now(timezone.utc).isoformat()
            if sync_state_store and sync_id:
                await sync_state_store.complete_sync(sync_id, cursor=cursor)
                await sync_state_store.update_sync(sync_id, items_processed=items_total)

            logger.info(f"Catalog sync completed: {items_total} items")

        except Exception as e:
            logger.error(f"Catalog sync failed: {e}")
            if sync_state_store and sync_id:
                await sync_state_store.complete_sync(sync_id, error=str(e)[:500])

    asyncio.create_task(_run_full_sync())
    return {"sync_id": sync_id, "status": "running", "sync_type": request.sync_type}


from fastapi import Request as FastAPIRequest


@app.post("/webhooks/shopify")
async def shopify_webhook_handler(request: FastAPIRequest):
    """Receive and process Shopify webhooks."""
    if not webhook_handler:
        return {"error": "Webhook handler not configured"}, 503

    raw_body = await request.body()
    hmac_header = request.headers.get("x-shopify-hmac-sha256", "")
    topic = request.headers.get("x-shopify-topic", "")
    shop_domain = request.headers.get("x-shopify-shop-domain", "")

    if not hmac_header or not topic:
        raise HTTPException(status_code=400, detail="Missing Shopify headers")

    if not webhook_handler.verify_hmac(raw_body, hmac_header):
        logger.warning(f"Invalid webhook HMAC from {shop_domain}")
        raise HTTPException(status_code=401, detail="Invalid HMAC")

    # Respond immediately (Shopify expects 200 within 5s)
    import json as _json
    body = _json.loads(raw_body)

    async def _process():
        try:
            await webhook_handler.handle(topic, shop_domain, body)
        except Exception as e:
            logger.error(f"Error processing webhook {topic}: {e}")

    asyncio.create_task(_process())
    return {"received": True}


# ── Competitor Indexing Endpoints ─────────────────────────────


@app.post("/competitor/index-url")
async def competitor_index_url(request: CompetitorCrawlRequest):
    """Start crawling a competitor site into competitor_products collection."""
    if not validators.url(request.url):
        raise HTTPException(status_code=400, detail=f"Invalid URL: {request.url}")

    import uuid
    job_id = uuid.uuid4().hex[:12]

    placeholder_job = CrawlJob(
        job_id=job_id,
        url=request.url,
        max_depth=request.max_depth,
        max_pages=request.max_pages,
        status="queued",
        smart_mode=request.smart_mode,
    )
    crawl_jobs[job_id] = placeholder_job
    crawl_queue.append(job_id)

    # Tag this job for competitor indexer
    placeholder_job._competitor = True

    asyncio.create_task(_process_crawl_queue())

    return {"job_id": job_id, "status": "queued", "url": request.url, "collection": "competitor_products"}


@app.get("/competitor/sources")
async def competitor_sources():
    """List indexed competitor domains."""
    return {"sources": competitor_indexer.get_sources()}


@app.delete("/competitor/source/{domain:path}")
async def competitor_delete_source(domain: str):
    """Delete competitor content for a domain."""
    success = competitor_indexer.delete_source(domain)
    if success:
        return {"status": "deleted", "domain": domain}
    raise HTTPException(status_code=500, detail=f"Failed to delete domain: {domain}")


# ── Classification Endpoints ──────────────────────────────────


@app.post("/classify/extract")
async def classify_extract(request: ClassifyRequest):
    """Start product extraction job for a domain."""
    def progress_cb(j: ClassificationJob):
        classification_jobs[j.job_id] = j

    job = await product_classifier.extract_products_from_collection(
        domain=request.domain,
        collection_name=request.collection,
        batch_size=request.batch_size,
        progress_callback=progress_cb,
    )
    classification_jobs[job.job_id] = job
    return job.to_dict()


@app.get("/classify/status/{job_id}")
async def classify_status(job_id: str):
    """Get classification job status."""
    job = classification_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_dict()


@app.get("/classify/products/{job_id}")
async def classify_products(job_id: str):
    """Get extracted products from a classification job."""
    job = classification_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return {"products": job.results, "count": len(job.results)}


@app.post("/classify/resolve")
async def classify_resolve(request: EntityResolveRequest):
    """Run entity resolution for a domain's extracted products against the catalog."""
    # Find the latest classification job for this domain
    domain_job = None
    for job in classification_jobs.values():
        if job.domain == request.domain and job.status == "completed":
            domain_job = job

    if not domain_job or not domain_job.results:
        raise HTTPException(
            status_code=404,
            detail=f"No completed extraction found for domain: {request.domain}. Run /classify/extract first.",
        )

    matches = await product_classifier.resolve_entities(
        extracted_products=domain_job.results,
        product_indexer=product_indexer,
    )
    report = ProductClassifier.generate_price_report(matches)
    return report


# ── Translation Endpoints ─────────────────────────────────────


@app.post("/translate/batch")
async def translate_batch(request: TranslateRequest):
    """Translate a list of texts with glossary + optional RAG context."""
    rag_context = None

    if request.use_rag and product_indexer is not None and request.texts:
        try:
            # Search for similar products to provide domain context
            combined_query = " ".join(request.texts)[:500]
            similar = product_indexer.search(query=combined_query, top_k=5)
            if similar:
                lines = []
                for r in similar:
                    title = r.get("title", "Unknown")
                    brand = r.get("brand", "")
                    category = r.get("category", "")
                    # Include structured metadata for terminology reference
                    parts = [f"Product: {title}"]
                    if brand:
                        parts.append(f"Brand: {brand}")
                    if category:
                        parts.append(f"Category: {category}")
                    # Include a snippet of the description for context
                    text = r.get("text", "")[:150]
                    if text:
                        parts.append(f"Description: {text}")
                    lines.append(" | ".join(parts))
                rag_context = "\n".join(lines)
                logger.info(f"RAG context built for translation: {len(similar)} similar products")
        except Exception as e:
            logger.warning(f"RAG context lookup failed, continuing without: {e}")

    job = await translator.translate_batch(
        texts=request.texts,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        rag_context=rag_context,
    )
    translation_jobs[job.job_id] = job

    if job.status == "failed":
        raise HTTPException(
            status_code=502,
            detail=f"Translation failed: {job.logs[-1] if job.logs else 'unknown error'}",
        )

    return {
        "job_id": job.job_id,
        "status": job.status,
        "translations": job.results,
    }


@app.get("/translate/status/{job_id}")
async def translate_status(job_id: str):
    """Get translation job status."""
    job = translation_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_dict()


@app.post("/translate/normalize")
async def translate_normalize(product: dict):
    """Normalize specs in a product dict."""
    return TranslationPipeline.normalize_specs(product)


# ── Glossary Endpoints ────────────────────────────────────────


@app.get("/glossary")
async def glossary_list(source_lang: str = "en", target_lang: str = "es"):
    """List glossary entries (built-in + custom) for a language pair."""
    builtin = get_glossary_for_pair(source_lang, target_lang)
    custom = await glossary_store.get_all(source_lang, target_lang)
    return {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "builtin_count": len(builtin),
        "custom_count": len(custom),
        "builtin": builtin,
        "custom": custom,
        "languages": SUPPORTED_LANGUAGES,
    }


@app.post("/glossary")
async def glossary_add(request: GlossaryEntryRequest):
    """Add or update a custom glossary entry for a language pair."""
    if not request.source.strip() or not request.target.strip():
        raise HTTPException(status_code=400, detail="Both source and target terms are required")
    await glossary_store.add(request.source, request.target, request.source_lang, request.target_lang)
    return {
        "status": "added",
        "source": request.source.strip().lower(),
        "target": request.target.strip(),
        "source_lang": request.source_lang,
        "target_lang": request.target_lang,
    }


@app.post("/glossary/bulk")
async def glossary_bulk_add(request: GlossaryBulkRequest):
    """Add multiple glossary entries for a language pair."""
    count = await glossary_store.add_bulk(request.entries, request.source_lang, request.target_lang)
    return {"status": "added", "count": count, "source_lang": request.source_lang, "target_lang": request.target_lang}


@app.delete("/glossary/{term:path}")
async def glossary_delete(term: str, source_lang: str = "en", target_lang: str = "es"):
    """Remove a custom glossary entry for a language pair."""
    existed = await glossary_store.remove(term, source_lang, target_lang)
    if not existed:
        raise HTTPException(status_code=404, detail=f"Term not found: {term} ({source_lang}→{target_lang})")
    return {"status": "deleted", "term": term, "source_lang": source_lang, "target_lang": target_lang}


@app.get("/glossary/test")
async def glossary_test(text: str, source_lang: str = "en", target_lang: str = "es"):
    """Test which glossary terms would be matched for a given text."""
    relevant = glossary_store.get_relevant(text, source_lang, target_lang)
    return {
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "matched_terms": len(relevant),
        "glossary": relevant,
    }


@app.get("/glossary/languages")
async def glossary_languages():
    """List all supported languages."""
    return {"languages": SUPPORTED_LANGUAGES}


# ── Agent Endpoints ───────────────────────────────────────────


def _cleanup_old_agent_tasks():
    """Remove in-memory tasks older than 5 minutes (Redis has the persistent copy)."""
    now = datetime.now(timezone.utc)
    stale = []
    for tid, task in agent_tasks.items():
        if task.status in ("completed", "failed", "cancelled") and task.ended_at:
            try:
                ended = datetime.fromisoformat(task.ended_at)
                if (now - ended).total_seconds() > 300:
                    stale.append(tid)
            except (ValueError, TypeError):
                pass
    for tid in stale:
        del agent_tasks[tid]


def _make_services(task: AgentTask = None) -> AgentServices:
    return AgentServices(
        web_indexer=web_indexer,
        product_indexer=product_indexer,
        llm_client=llm_client,
        task=task,
    )


@app.get("/agent/status")
async def agent_service_status():
    """Check if LLM is available for agent tasks."""
    return {"available": True, "model_id": LLM_MODEL, "redis": redis_client is not None}


@app.post("/agent/task")
async def create_agent_task(request: AgentTaskRequest):
    """Create and start a new agent task."""
    _cleanup_old_agent_tasks()

    if not rag_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized — LLM may be unavailable")

    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    running = sum(1 for t in agent_tasks.values() if t.status == "running")
    if running >= 3:
        raise HTTPException(status_code=429, detail="Maximum 3 concurrent agent tasks")

    import uuid
    task_id = uuid.uuid4().hex[:12]
    task = AgentTask(
        task_id=task_id,
        prompt=prompt,
        started_at=datetime.now(timezone.utc).isoformat(),
        source="web",
    )

    # Wire Redis persistence callbacks
    if session_store:
        task._on_log = session_store.add_log
        task._on_step = session_store.add_step
        await session_store.create_task(task, source="web")

    agent_tasks[task_id] = task

    session = RedisSession(task_id, redis_client) if redis_client else None
    services = _make_services(task)

    async def _run():
        await run_task(rag_agent, services, prompt, task=task,
                       session=session, session_store=session_store)

    asyncio.create_task(_run())
    logger.info(f"Started agent task {task_id}: {prompt[:80]}")

    return {"task_id": task_id, "status": "running"}


@app.post("/agent/task/{task_id}/message")
async def send_agent_message(task_id: str, request: AgentContinueRequest):
    """Send a message to an agent task.

    - If the task is **running**: queues the message — it will be processed
      automatically after the current agent run finishes.
    - If the task is **completed/failed**: restarts the session with the
      new message (like the old /continue endpoint).
    """
    if not rag_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    if not session_store or not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available for session resume")

    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Check in-memory first (running tasks), then Redis
    mem_task = agent_tasks.get(task_id)
    if mem_task and mem_task.status == "running":
        # Task is running — queue the message for after current run
        queue_len = await session_store.push_message(task_id, message)
        mem_task.log(f"User message queued (position {queue_len}): {message[:200]}")
        mem_task.add_step("user_message", f"[queued] {message}")
        logger.info(f"Queued message for running task {task_id}: {message[:80]}")
        return {"task_id": task_id, "status": "queued", "queue_position": queue_len}

    # Task is not running in memory — check Redis
    existing = await session_store.get_task(task_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Session not found: {task_id}")

    if existing["status"] == "running":
        # Running in Redis but not in memory (e.g. after restart) — queue it
        queue_len = await session_store.push_message(task_id, message)
        logger.info(f"Queued message for task {task_id} (Redis-only): {message[:80]}")
        return {"task_id": task_id, "status": "queued", "queue_position": queue_len}

    # Task is completed/failed — restart the session
    task = AgentTask(
        task_id=task_id,
        prompt=existing.get("prompt", ""),
        started_at=existing.get("started_at", ""),
        source=existing.get("source", "web"),
    )
    task._on_log = session_store.add_log
    task._on_step = session_store.add_step
    task.status = "running"
    await session_store.update_task(task)

    agent_tasks[task_id] = task

    session = RedisSession(task_id, redis_client)
    services = _make_services(task)

    task.add_step("user_message", message)
    task.log(f"Resume with: {message[:200]}")

    async def _run():
        await run_task(rag_agent, services, message, task=task,
                       session=session, session_store=session_store)

    asyncio.create_task(_run())
    logger.info(f"Continued agent task {task_id}: {message[:80]}")

    return {"task_id": task_id, "status": "running", "continued": True}


@app.post("/agent/task/{task_id}/continue")
async def continue_agent_task(task_id: str, request: AgentContinueRequest):
    """Legacy endpoint — redirects to /message. Kept for backwards compatibility."""
    return await send_agent_message(task_id, request)


@app.get("/agent/tasks")
async def list_agent_tasks():
    """List all agent tasks (from Redis if available, else in-memory)."""
    if session_store:
        try:
            tasks = await session_store.list_tasks(limit=50)
            # Merge in-memory running tasks (freshest data)
            redis_ids = {t["task_id"] for t in tasks}
            for tid, mem_task in agent_tasks.items():
                if tid not in redis_ids:
                    tasks.append(mem_task.to_dict())
            tasks.sort(key=lambda t: t.get("started_at", ""), reverse=True)
            return {"tasks": tasks}
        except Exception as e:
            logger.warning(f"Redis list_tasks failed, falling back to memory: {e}")

    tasks = [t.to_dict() for t in agent_tasks.values()]
    tasks.sort(key=lambda t: t.get("started_at", ""), reverse=True)
    return {"tasks": tasks}


@app.get("/agent/task/{task_id}")
async def get_agent_task(task_id: str):
    # In-memory first (running tasks have freshest data)
    if task_id in agent_tasks:
        return agent_tasks[task_id].to_dict()
    # Then Redis
    if session_store:
        task_data = await session_store.get_task(task_id)
        if task_data:
            task_data["steps"] = await session_store.get_steps(task_id, last_n=50)
            return task_data
    raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")


@app.get("/agent/task/{task_id}/stream")
async def agent_task_stream(task_id: str):
    """SSE stream of agent task progress."""
    if task_id not in agent_tasks:
        raise HTTPException(status_code=404, detail=f"Task not found in active tasks: {task_id}")

    async def event_stream():
        prev_data = None
        while True:
            task = agent_tasks.get(task_id)
            if not task:
                yield f"data: {json.dumps({'error': 'Task disappeared'})}\n\n"
                break

            data = json.dumps(task.to_dict())
            if data != prev_data:
                yield f"data: {data}\n\n"
                prev_data = data

            if task.status in ("completed", "failed", "cancelled"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/agent/task/{task_id}/logs")
async def agent_task_logs(task_id: str, offset: int = 0):
    # In-memory first
    if task_id in agent_tasks:
        task = agent_tasks[task_id]
        return {
            "task_id": task_id,
            "logs": task.logs[offset:],
            "total": len(task.logs),
            "offset": offset,
            "status": task.status,
        }
    # Then Redis
    if session_store:
        logs, total = await session_store.get_logs(task_id, offset=offset)
        task_data = await session_store.get_task(task_id)
        if task_data:
            return {
                "task_id": task_id,
                "logs": logs,
                "total": total,
                "offset": offset,
                "status": task_data.get("status", "unknown"),
            }
    raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")


@app.get("/agent/task/{task_id}/history")
async def get_agent_conversation_history(task_id: str):
    """Get the full SDK conversation history for a session (for debugging/inspection)."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    session = RedisSession(task_id, redis_client)
    items = await session.get_items()
    return {"task_id": task_id, "items": items, "count": len(items)}


@app.get("/agent/task/{task_id}/steps")
async def get_agent_all_steps(task_id: str):
    """Get ALL steps for a task (not just last 50)."""
    if session_store:
        steps = await session_store.get_all_steps(task_id)
        if steps:
            return {"task_id": task_id, "steps": steps, "count": len(steps)}
    # Fallback to in-memory
    if task_id in agent_tasks:
        return {"task_id": task_id, "steps": agent_tasks[task_id].steps, "count": len(agent_tasks[task_id].steps)}
    raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)


# ── Order Endpoints ─────────────────────────────────────────────


@app.get("/orders")
async def list_orders(
    query: str = "",
    limit: int = 20,
    status: str = "any",
):
    """List/search orders. query: customer name, email, or order number."""
    if not shopify_graphql_client:
        raise HTTPException(
            status_code=503,
            detail="Shopify not configured. Set SHOPIFY_SHOP_DOMAIN and SHOPIFY_ACCESS_TOKEN.",
        )
    try:
        orders = await shopify_graphql_client.fetch_orders(
            query=query or None, first=min(limit, 50), status=status
        )
        return {"total": len(orders), "orders": orders}
    except Exception as e:
        logger.error(f"Order list failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:300])


@app.get("/orders/{id_or_name}")
async def get_order_detail(id_or_name: str):
    """Get single order by name (#1234) or GID."""
    if not shopify_graphql_client:
        raise HTTPException(
            status_code=503,
            detail="Shopify not configured. Set SHOPIFY_SHOP_DOMAIN and SHOPIFY_ACCESS_TOKEN.",
        )
    try:
        order = await shopify_graphql_client.fetch_order(id_or_name)
        if not order:
            raise HTTPException(status_code=404, detail=f"Order {id_or_name} not found")
        return order
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Order detail failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:300])


@app.get("/orders/{id_or_name}/fulfillments")
async def get_order_fulfillments(id_or_name: str):
    """Get fulfillment/shipment info for an order."""
    if not shopify_graphql_client:
        raise HTTPException(
            status_code=503,
            detail="Shopify not configured. Set SHOPIFY_SHOP_DOMAIN and SHOPIFY_ACCESS_TOKEN.",
        )
    try:
        order = await shopify_graphql_client.fetch_order(id_or_name)
        if not order:
            raise HTTPException(status_code=404, detail=f"Order {id_or_name} not found")
        return {
            "order_name": order.get("name", ""),
            "fulfillment_status": order.get("fulfillment_status", ""),
            "fulfillments": order.get("fulfillments", []),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fulfillment lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:300])


# ── Knowledge Vault Endpoints ────────────────────────────────


@app.post("/knowledge/rebuild")
async def knowledge_rebuild():
    """Trigger full vault rebuild: competitors + content sources. Redis-locked."""
    if not vault_builder:
        raise HTTPException(status_code=503, detail="Knowledge vault not configured")

    result = await vault_builder.build_full()
    if result.errors and result.errors[0] == "Rebuild already in progress":
        raise HTTPException(status_code=409, detail="Rebuild already in progress")

    # Phase 2: Content learning (runs after competitors)
    content_result = None
    if content_learner and content_learner.sources:
        try:
            content_result = await content_learner.learn()
            result.notes_written += content_result.notes_written
            logger.info(f"Content learning: {content_result.sources_processed} sources, {content_result.notes_written} notes")
        except Exception as e:
            logger.error(f"Content learning failed: {e}")
            result.errors.append(f"Content learning: {str(e)[:200]}")

    # Auto-reindex after build
    if vault_indexer_instance and result.notes_written > 0:
        asyncio.create_task(_reindex_vault())

    response = {
        "status": "completed",
        "competitors_processed": result.competitors_processed,
        "competitors_failed": result.competitors_failed,
        "notes_written": result.notes_written,
        "errors": result.errors,
        "competitor_results": result.competitor_results,
    }
    if content_result:
        response["content_learning"] = {
            "sources_processed": content_result.sources_processed,
            "sources_failed": content_result.sources_failed,
            "guides_extracted": content_result.guides_extracted,
            "trends_detected": content_result.trends_detected,
        }
    return response


@app.post("/knowledge/rebuild/{domain}")
async def knowledge_rebuild_domain(domain: str):
    """Rebuild vault for a single competitor domain."""
    if not vault_builder:
        raise HTTPException(status_code=503, detail="Knowledge vault not configured")

    result = await vault_builder.build_incremental(domain)
    if result.errors and result.errors[0] == "Rebuild already in progress":
        raise HTTPException(status_code=409, detail="Rebuild already in progress")

    # Auto-reindex after build
    if vault_indexer_instance and result.notes_written > 0:
        asyncio.create_task(_reindex_vault())

    return {
        "status": "completed",
        "competitors_processed": result.competitors_processed,
        "notes_written": result.notes_written,
        "errors": result.errors,
    }


@app.post("/knowledge/reindex")
async def knowledge_reindex(force: bool = False):
    """Re-index vault notes into Qdrant knowledge_brain collection."""
    if not vault_indexer_instance:
        raise HTTPException(status_code=503, detail="Knowledge vault not configured")

    result = await vault_indexer_instance.index_vault(force=force)
    return {"status": "completed", **result}


@app.post("/knowledge/synthesize")
async def knowledge_synthesize_endpoint(force: bool = False):
    """Trigger knowledge synthesis (generate/update recommendation notes)."""
    if not knowledge_synthesizer:
        raise HTTPException(status_code=503, detail="Knowledge synthesizer not configured")

    import time as _time
    _synth_start = _time.time()
    result = await knowledge_synthesizer.synthesize(force=force)
    SYNTHESIS_DURATION.observe(_time.time() - _synth_start)
    SYNTHESIS_RUNS.labels(status="success" if result.get("notes_failed", 0) == 0 else "partial").inc()
    SYNTHESIS_NOTES_GENERATED.inc(result.get("notes_generated", 0))
    SYNTHESIS_LLM_CALLS.inc(result.get("llm_calls", 0))

    # Auto-reindex after synthesis
    if vault_indexer_instance and result.get("notes_generated", 0) > 0:
        asyncio.create_task(vault_indexer_instance.index_vault())

    return {"status": "completed", **result}


@app.get("/dashboard/api/timeline")
async def dashboard_timeline(hours: int = 24, limit: int = 100):
    """Get recent activity events for the dashboard."""
    return read_timeline(hours=hours, limit=limit)


@app.get("/dashboard/api/collections")
async def dashboard_collections():
    """Get Qdrant collection stats for the dashboard."""
    try:
        collections_response = product_indexer.client.get_collections()
        result = []
        for col in collections_response.collections:
            try:
                info = product_indexer.client.get_collection(col.name)
                result.append({
                    "name": col.name,
                    "points_count": info.points_count,
                    "vectors_count": getattr(info, "vectors_count", info.points_count),
                    "status": info.status.value if info.status else "unknown",
                })
            except Exception:
                result.append({"name": col.name, "points_count": 0, "status": "error"})
        return {"collections": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Dashboard SPA — must be after /dashboard/api/* routes
app.mount("/dashboard", StaticFiles(directory="static/dashboard", html=True), name="dashboard")


@app.post("/knowledge/learn")
async def knowledge_learn():
    """Trigger content learning only (Reddit, forums, blogs — no competitor product extraction)."""
    if not content_learner:
        raise HTTPException(status_code=503, detail="Content learner not configured")

    result = await content_learner.learn()

    # Auto-reindex after learning
    if vault_indexer_instance and result.notes_written > 0:
        asyncio.create_task(_reindex_vault())

    return {
        "status": "completed",
        "sources_processed": result.sources_processed,
        "sources_failed": result.sources_failed,
        "notes_written": result.notes_written,
        "guides_extracted": result.guides_extracted,
        "trends_detected": result.trends_detected,
        "errors": result.errors,
        "source_results": result.source_results,
    }


class DeepAnalyzeRequest(BaseModel):
    urls: Optional[List[str]] = None


@app.post("/knowledge/deep-analyze")
async def knowledge_deep_analyze(request: DeepAnalyzeRequest = None):
    """Run deep competitive analysis on all competitors or specific URLs.

    Uses Firecrawl sitemap mapping + strategic page scraping + LLM deep analysis.
    Much faster than full rebuild — analyzes site structure, not every page.
    """
    vault_path = os.getenv("VAULT_PATH", "/app/knowledge-vault")
    fc = FirecrawlClient()

    if not await fc.is_available():
        raise HTTPException(status_code=503, detail="Firecrawl not available at " + fc.base_url)

    analyzer = DeepAnalyzer(firecrawl=fc, llm_client=llm_client, vault_path=vault_path)

    # Determine targets
    urls = request.urls if request else None
    if urls:
        targets = [{"url": u, "name": u.split("//")[1].split("/")[0], "slug": u.split("//")[1].split("/")[0].replace(".", "-").replace("www-", "")} for u in urls]
    elif vault_builder:
        targets = [{"url": c.url, "name": c.name, "slug": c.slug} for c in vault_builder.competitors]
    else:
        raise HTTPException(status_code=400, detail="No targets specified and no competitors configured")

    results = {}
    notes_written = 0
    for target in targets:
        logger.info(f"Deep analyzing: {target['name']} ({target['url']})")
        result = await analyzer.analyze_competitor(target["url"], target["name"], target["slug"])
        note_path = analyzer.write_deep_note(result)
        notes_written += 1
        results[target["slug"]] = {
            "name": target["name"],
            "pages_scraped": result.get("pages_scraped", 0),
            "sitemap_urls": result.get("sitemap", {}).get("total_urls", 0),
            "has_analysis": bool(result.get("analysis", {}).get("MARKET_POSITIONING")),
            "errors": result.get("errors", []),
        }

    # Reindex vault after analysis
    if vault_indexer_instance and notes_written > 0:
        asyncio.create_task(_reindex_vault())

    return {
        "status": "completed",
        "competitors_analyzed": len(results),
        "notes_written": notes_written,
        "results": results,
    }


@app.post("/knowledge/research")
async def knowledge_research():
    """Run research agent to fill intelligence gaps in competitor analyses.

    Reads existing deep analysis notes, identifies missing data (brands, categories,
    prices, services, reputation), and researches via web search + LLM extraction.
    Designed to run hourly to continuously improve intelligence.
    """
    vault_path = os.getenv("VAULT_PATH", "/app/knowledge-vault")
    fc = FirecrawlClient()

    if not await fc.is_available():
        raise HTTPException(status_code=503, detail="Firecrawl not available")

    agent = ResearchAgent(firecrawl=fc, llm_client=llm_client, vault_path=vault_path)
    result = await agent.research_all()

    # Reindex after research updates
    if vault_indexer_instance and result.get("notes_updated", 0) > 0:
        asyncio.create_task(_reindex_vault())

    return result


@app.get("/knowledge/status")
async def knowledge_status():
    """Get vault build status, note count, and health."""
    if not vault_builder:
        raise HTTPException(status_code=503, detail="Knowledge vault not configured")

    status = vault_builder.get_status()

    # Add Qdrant collection info
    if vault_indexer_instance:
        collection_info = vault_indexer_instance.get_collection_info()
        if collection_info:
            status["qdrant"] = collection_info

    # Add content learner status
    if content_learner:
        status["content_learning"] = content_learner.get_status()

    return status


@app.get("/knowledge/notes")
async def knowledge_notes():
    """List all vault notes with their frontmatter."""
    if not vault_builder:
        raise HTTPException(status_code=503, detail="Knowledge vault not configured")

    import frontmatter as fm
    from pathlib import Path

    vault_path = Path(vault_builder.vault_path)
    notes = []
    for md_file in vault_path.rglob("*.md"):
        rel = md_file.relative_to(vault_path)
        if any(part in ("_templates", "_meta", ".obsidian") for part in rel.parts):
            continue
        try:
            post = fm.load(str(md_file))
            notes.append({
                "path": str(rel),
                "type": post.metadata.get("type", ""),
                "frontmatter": dict(post.metadata),
            })
        except Exception:
            notes.append({"path": str(rel), "type": "unknown", "frontmatter": {}})

    return {"notes": notes, "count": len(notes)}


@app.get("/knowledge/note/{path:path}")
async def knowledge_note(path: str):
    """Read a single vault note by relative path."""
    if not vault_builder:
        raise HTTPException(status_code=503, detail="Knowledge vault not configured")

    from pathlib import Path
    note_path = Path(vault_builder.vault_path) / path
    if not note_path.exists() or not note_path.suffix == ".md":
        raise HTTPException(status_code=404, detail=f"Note not found: {path}")

    # Security: ensure path doesn't escape vault
    try:
        note_path.resolve().relative_to(Path(vault_builder.vault_path).resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Path traversal not allowed")

    return {
        "path": path,
        "content": note_path.read_text(encoding="utf-8"),
    }


async def _reindex_vault():
    """Background task to reindex vault after a build."""
    try:
        result = await vault_indexer_instance.index_vault()
        logger.info(f"Auto-reindex after build: {result}")
    except Exception as e:
        logger.error(f"Auto-reindex failed: {e}")


# ── Compatibility Admin Endpoints ──────────────────────────────


@app.get("/admin/compatibility/stats")
async def compatibility_stats():
    """Show compatibility data coverage statistics."""
    from qdrant_client.http.models import IsEmptyCondition
    from metrics import (
        COMPAT_PRODUCTS_TOTAL, COMPAT_WITH_PLATFORMS, COMPAT_WITH_TYPE,
        COMPAT_BASE_PLATFORMS, COMPAT_COVERAGE_PCT,
    )

    total = product_indexer.client.count(
        collection_name=product_indexer.COLLECTION_NAME,
    ).count

    with_compat = product_indexer.client.count(
        collection_name=product_indexer.COLLECTION_NAME,
        count_filter=Filter(must_not=[
            IsEmptyCondition(is_empty={"key": "compatible_platforms"})
        ]),
    ).count

    with_type = product_indexer.client.count(
        collection_name=product_indexer.COLLECTION_NAME,
        count_filter=Filter(must_not=[
            IsEmptyCondition(is_empty={"key": "upgrade_type"})
        ]),
    ).count

    base_platforms = product_indexer.client.count(
        collection_name=product_indexer.COLLECTION_NAME,
        count_filter=Filter(must=[
            FieldCondition(key="is_base_platform", match=MatchValue(value=True))
        ]),
    ).count

    coverage = round(with_compat / total * 100, 1) if total else 0

    # Update Prometheus gauges
    COMPAT_PRODUCTS_TOTAL.set(total)
    COMPAT_WITH_PLATFORMS.set(with_compat)
    COMPAT_WITH_TYPE.set(with_type)
    COMPAT_BASE_PLATFORMS.set(base_platforms)
    COMPAT_COVERAGE_PCT.set(coverage)

    return {
        "total_products": total,
        "with_compatible_platforms": with_compat,
        "with_upgrade_type": with_type,
        "base_platforms": base_platforms,
        "coverage_pct": coverage,
    }


@app.post("/admin/compatibility/analyze-product/{shopify_id}")
async def analyze_product_compatibility(shopify_id: str):
    """Analyze a single product's compatibility."""
    if not compatibility_analyzer:
        raise HTTPException(status_code=500, detail="Compatibility analyzer not initialized")

    results = product_indexer.client.scroll(
        collection_name=product_indexer.COLLECTION_NAME,
        scroll_filter=Filter(must=[
            FieldCondition(key="shopify_id", match=MatchValue(value=shopify_id))
        ]),
        limit=1,
        with_payload=True,
    )
    points = results[0]
    if not points:
        raise HTTPException(status_code=404, detail=f"Product {shopify_id} not found")

    product = points[0].payload
    result = await compatibility_analyzer.analyze_product(product)

    product_indexer.update_compatibility(shopify_id, result.to_payload())

    return {
        "shopify_id": shopify_id,
        "title": product.get("title"),
        **result.to_payload(),
        "confidence": result.confidence,
        "reasoning": result.reasoning,
    }


@app.post("/admin/compatibility/analyze-all")
async def analyze_all_compatibility():
    """Run batch compatibility analysis on products missing data. Runs in background."""
    if not compatibility_analyzer:
        raise HTTPException(status_code=500, detail="Compatibility analyzer not initialized")

    from qdrant_client.http.models import IsEmptyCondition

    async def _update_compat_metrics():
        """Refresh compatibility Prometheus gauges."""
        try:
            from qdrant_client.http.models import IsEmptyCondition as _IE
            from metrics import (
                COMPAT_PRODUCTS_TOTAL, COMPAT_WITH_PLATFORMS, COMPAT_WITH_TYPE,
                COMPAT_BASE_PLATFORMS, COMPAT_COVERAGE_PCT,
            )
            total = product_indexer.client.count(collection_name=product_indexer.COLLECTION_NAME).count
            with_compat = product_indexer.client.count(
                collection_name=product_indexer.COLLECTION_NAME,
                count_filter=Filter(must_not=[_IE(is_empty={"key": "compatible_platforms"})]),
            ).count
            with_type = product_indexer.client.count(
                collection_name=product_indexer.COLLECTION_NAME,
                count_filter=Filter(must_not=[_IE(is_empty={"key": "upgrade_type"})]),
            ).count
            base = product_indexer.client.count(
                collection_name=product_indexer.COLLECTION_NAME,
                count_filter=Filter(must=[FieldCondition(key="is_base_platform", match=MatchValue(value=True))]),
            ).count
            COMPAT_PRODUCTS_TOTAL.set(total)
            COMPAT_WITH_PLATFORMS.set(with_compat)
            COMPAT_WITH_TYPE.set(with_type)
            COMPAT_BASE_PLATFORMS.set(base)
            COMPAT_COVERAGE_PCT.set(round(with_compat / total * 100, 1) if total else 0)
        except Exception as e:
            logger.warning(f"Failed to update compat metrics: {e}")

    async def _run_batch():
        logger.info("Starting batch compatibility analysis...")
        offset = None
        analyzed = 0
        skipped = 0
        errors = 0

        while True:
            # Only process products MISSING compatibility data
            results, offset = product_indexer.client.scroll(
                collection_name=product_indexer.COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key="source_type", match=MatchValue(value="product")),
                    IsEmptyCondition(is_empty={"key": "compatible_platforms"}),
                ]),
                limit=50,
                offset=offset,
                with_payload=True,
            )

            if not results:
                break

            for point in results:
                product = point.payload
                shopify_id = product.get("shopify_id")
                if not shopify_id:
                    skipped += 1
                    continue

                try:
                    result = await compatibility_analyzer.analyze_product(product)
                    product_indexer.update_compatibility(str(shopify_id), result.to_payload())
                    analyzed += 1
                    if analyzed % 100 == 0:
                        logger.info(f"Compatibility batch: {analyzed} done, {skipped} skipped, {errors} errors")
                        await _update_compat_metrics()
                except Exception as e:
                    logger.error(f"Failed to analyze {shopify_id}: {e}")
                    errors += 1

            if offset is None:
                break

        logger.info(f"Batch compatibility complete: {analyzed} analyzed, {skipped} skipped, {errors} errors")
        await _update_compat_metrics()

    asyncio.create_task(_run_batch())

    return {"status": "started", "message": "Batch analysis running in background. Check /admin/compatibility/stats for progress."}
