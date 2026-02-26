"""
Qdrant RAG MCP Server
Provides tools for Claude to analyze webpages, create collections with hybrid search,
scrape/index content, and search across Qdrant collections.
"""

import hashlib
import json
import logging
import os
import re
import sys
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura
from bs4 import BeautifulSoup, NavigableString, Tag
from fastembed import SparseTextEmbedding
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

# Logging to stderr only (stdout is reserved for MCP JSON-RPC)
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
POCHARLIES_RAG_URL = os.getenv("POCHARLIES_RAG_URL", "http://localhost:8080")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MIN_CONTENT_LENGTH = 200

# URL path patterns to exclude from crawling
_URL_BLACKLIST_PATTERNS = [
    "/go/product/", "/go/category/",
    "/tags/", "/brands/",
    "/admin", "/login", "/cart", "/checkout", "/account",
    "/privacy", "/terms", "/sitemap", "/wishlist", "/compare",
    "?page=", "?sort=", "?limit=",
]

# CSS selectors for e-commerce boilerplate elements (always removed)
_ECOMMERCE_EXCLUDE_SELECTORS = [
    "nav", ".nav", "#nav", ".menu", "#menu", ".breadcrumb", ".breadcrumbs",
    "header", "footer", ".footer", "#footer", "aside", ".sidebar",
    "[class*='cookie']", "[id*='cookie']",
    "[class*='consent']", "[class*='gdpr']",
    ".cart-widget", ".mini-cart", ".wishlist", ".compare",
    ".related-products", ".also-bought", ".upsell", ".cross-sell",
    "[class*='newsletter']", "[class*='popup']", "[class*='modal']",
]

# Regex patterns for e-commerce boilerplate text
_BOILERPLATE_PATTERNS = [
    re.compile(r"A[n\u00f1]adir al carrito.*?(?:\n|$)", re.IGNORECASE),
    re.compile(r"Add to cart.*?(?:\n|$)", re.IGNORECASE),
    re.compile(r"Productos?\s+relacionados?.*", re.DOTALL | re.IGNORECASE),
    re.compile(r"Related products?.*", re.DOTALL | re.IGNORECASE),
    re.compile(r"Lorem ipsum.*?(?:\n\n|\Z)", re.DOTALL | re.IGNORECASE),
    re.compile(r"(?:Utilizamos|Usamos|We use)\s+cookies.*?(?:Aceptar|Accept|Rechazar|Reject|cerrar|close)", re.DOTALL | re.IGNORECASE),
]


def _is_blacklisted_url(url: str) -> bool:
    """Check if URL path matches any blacklist pattern."""
    parsed = urlparse(url)
    path_query = parsed.path + ("?" + parsed.query if parsed.query else "")
    return any(p in path_query for p in _URL_BLACKLIST_PATTERNS)


def _remove_ecommerce_boilerplate(soup: BeautifulSoup) -> None:
    """Remove hardcoded e-commerce boilerplate elements from soup."""
    for selector in _ECOMMERCE_EXCLUDE_SELECTORS:
        try:
            for el in soup.select(selector):
                el.decompose()
        except Exception:
            continue


def _strip_boilerplate(text: str) -> str:
    """Remove common boilerplate text patterns."""
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_jsonld(html: str) -> Optional[Dict]:
    """Extract Product structured data from JSON-LD script tags."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        all_items: list[dict] = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string, strict=False)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(data, dict):
                if "@graph" in data:
                    all_items.extend(data["@graph"])
                else:
                    all_items.append(data)
            elif isinstance(data, list):
                all_items.extend(data)

        result: Dict[str, str] = {}

        # Extract BreadcrumbList → category_path
        for item in all_items:
            if not isinstance(item, dict):
                continue
            if item.get("@type") == "BreadcrumbList":
                breadcrumbs = item.get("itemListElement", [])
                if breadcrumbs:
                    path = " > ".join(
                        str(bc.get("item", {}).get("name", "") if isinstance(bc.get("item"), dict) else bc.get("name", ""))
                        for bc in sorted(breadcrumbs, key=lambda x: x.get("position", 0))
                        if (bc.get("item", {}).get("name") if isinstance(bc.get("item"), dict) else bc.get("name"))
                    )
                    if path:
                        result["category_path"] = path
                break

        # Extract Product schema
        for item in all_items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("@type", "")
            type_list = item_type if isinstance(item_type, list) else [item_type]
            if "Product" not in type_list:
                continue

            offers = item.get("offers", {})
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            if not isinstance(offers, dict):
                offers = {}

            if item.get("name"):
                result["product_name"] = str(item["name"])[:200]
            if item.get("sku"):
                result["product_sku"] = str(item["sku"])
            ean = item.get("gtin13") or item.get("gtin14") or item.get("gtin")
            if ean:
                result["product_ean"] = str(ean)
            brand = item.get("brand")
            if isinstance(brand, dict):
                result["product_brand"] = str(brand.get("name", ""))[:100]
            elif brand:
                result["product_brand"] = str(brand)[:100]
            if item.get("description"):
                result["product_description"] = str(item["description"])[:500]
            img = item.get("image")
            if img:
                result["product_image"] = str(img[0] if isinstance(img, list) else img)[:500]
            if offers.get("price"):
                result["product_price"] = str(offers["price"])
            if offers.get("priceCurrency"):
                result["product_currency"] = str(offers["priceCurrency"])
            if offers.get("availability"):
                avail = str(offers["availability"])
                if "InStock" in avail:
                    result["product_availability"] = "InStock"
                elif "OutOfStock" in avail:
                    result["product_availability"] = "OutOfStock"
                else:
                    result["product_availability"] = avail.split("/")[-1]
            break

        return result if result else None
    except Exception:
        pass
    return None


# ── Shared state via lifespan ─────────────────────────────────────


@dataclass
class AppContext:
    qdrant: QdrantClient
    embed_model: SentenceTransformer
    bm25_model: SparseTextEmbedding
    embed_dim: int


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    logger.info("Loading embedding models (this may take a minute on first run)...")

    parsed = urlparse(QDRANT_URL)
    if parsed.scheme == "https":
        qdrant = QdrantClient(
            host=parsed.hostname,
            port=parsed.port or 443,
            https=True,
            api_key=QDRANT_API_KEY,
        )
    else:
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    embed_dim = embed_model.get_sentence_embedding_dimension()

    logger.info("Loading BM25 sparse encoder (Qdrant/bm25)...")
    bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    # Clean up removed collections (code_index, devops_docs, jira_tickets)
    _REMOVED_COLLECTIONS = ["code_index", "devops_docs", "jira_tickets"]
    for col_name in _REMOVED_COLLECTIONS:
        try:
            qdrant.delete_collection(col_name)
            logger.info(f"Deleted legacy collection: {col_name}")
        except Exception:
            pass  # collection doesn't exist, nothing to do

    logger.info(f"MCP server ready — Qdrant: {QDRANT_URL}, embed_dim: {embed_dim}")

    try:
        yield AppContext(
            qdrant=qdrant,
            embed_model=embed_model,
            bm25_model=bm25_model,
            embed_dim=embed_dim,
        )
    finally:
        qdrant.close()


mcp = FastMCP(
    "Qdrant RAG",
    lifespan=app_lifespan,
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8000")),
)


# ── Helper functions ──────────────────────────────────────────────


def _get_ctx(ctx: Context) -> AppContext:
    return ctx.request_context.lifespan_context


def _generate_id(collection: str, url: str, chunk_idx: int) -> int:
    key = f"{collection}:{url}:{chunk_idx}"
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:16], 16)


def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[Dict]:
    """Paragraph-aware text chunking."""
    if not text:
        return []

    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para)

        if current_len + para_len + 1 > max_chars and current:
            chunk_text = "\n\n".join(current)
            chunks.append({"text": chunk_text, "chunk_idx": len(chunks)})

            # Overlap: keep last paragraph(s) that fit
            overlap_parts = []
            overlap_len = 0
            for p in reversed(current):
                if overlap_len + len(p) > overlap:
                    break
                overlap_parts.insert(0, p)
                overlap_len += len(p) + 2

            current = overlap_parts
            current_len = overlap_len

        current.append(para)
        current_len += para_len + 2

    if current:
        chunk_text = "\n\n".join(current)
        if chunk_text.strip():
            chunks.append({"text": chunk_text, "chunk_idx": len(chunks)})

    return chunks


def _extract_content(html: str, url: str) -> Dict[str, str]:
    """Extract clean text from HTML using trafilatura, fallback to BS4."""
    title = ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
    except Exception:
        pass

    text = trafilatura.extract(html, include_links=False, include_tables=True)
    if text:
        text = _strip_boilerplate(text)

    if not text or len(text) < MIN_CONTENT_LENGTH:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            for sel in soup.select(
                '[class*="cookie"], [class*="consent"], [id*="cookie"], '
                '[id*="consent"], [class*="gdpr"], [id*="gdpr"]'
            ):
                sel.decompose()
            _remove_ecommerce_boilerplate(soup)
            text = soup.get_text(separator="\n", strip=True)
            if text:
                text = _strip_boilerplate(text)
        except Exception:
            text = ""

    if text:
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return {"title": title, "text": text or "", "url": url}


def _extract_content_smart(
    html: str,
    url: str,
    content_selectors: List[str],
    exclude_selectors: List[str],
) -> Dict[str, str]:
    """Extract content using CSS selectors, fallback to standard."""
    title = ""
    try:
        soup = BeautifulSoup(html, "html.parser")

        # Remove hardcoded e-commerce boilerplate before custom selectors
        _remove_ecommerce_boilerplate(soup)

        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        for selector in exclude_selectors:
            try:
                for el in soup.select(selector):
                    el.decompose()
            except Exception:
                continue

        content_parts = []
        for selector in content_selectors:
            try:
                for el in soup.select(selector):
                    text = el.get_text(separator="\n", strip=True)
                    if text and len(text) > 20:
                        content_parts.append(text)
            except Exception:
                continue

        text = "\n\n".join(content_parts)
        if text:
            text = _strip_boilerplate(text)

        if not text or len(text) < MIN_CONTENT_LENGTH:
            return _extract_content(html, url)

        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return {"title": title, "text": text, "url": url}

    except Exception:
        return _extract_content(html, url)


def _extract_links(html: str, base_url: str) -> List[str]:
    """Extract same-domain links from HTML."""
    base_parsed = urlparse(base_url)
    base_domain = base_parsed.netloc
    links = []

    try:
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)

            if parsed.netloc != base_domain:
                continue

            clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                clean += f"?{parsed.query}"

            skip_exts = {
                ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg",
                ".zip", ".tar", ".gz", ".mp4", ".mp3", ".webp",
            }
            if any(clean.lower().endswith(ext) for ext in skip_exts):
                continue

            if _is_blacklisted_url(clean):
                continue

            links.append(clean)
    except Exception:
        pass

    return list(set(links))


def _trim_html_for_analysis(html: str, max_chars: int = 4000) -> str:
    """Reduce HTML to a structural skeleton for Claude to analyze."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "svg", "noscript", "iframe", "link", "meta"]):
        tag.decompose()

    for element in soup.descendants:
        if isinstance(element, NavigableString) and not isinstance(element, Tag):
            text = element.strip()
            if len(text) > 80:
                element.replace_with(text[:80] + "...")

    keep_attrs = {"class", "id", "role"}
    for tag in soup.find_all(True):
        attrs_to_remove = [
            a for a in tag.attrs
            if a not in keep_attrs and not a.startswith("data-")
        ]
        for attr in attrs_to_remove:
            del tag[attr]

    result = soup.prettify()
    if len(result) > max_chars:
        result = result[:max_chars] + "\n<!-- truncated -->"
    return result


def _detect_content_selectors(html: str) -> List[str]:
    """Detect likely content area CSS selectors from HTML structure."""
    soup = BeautifulSoup(html, "html.parser")
    selectors = []

    # Common content area patterns
    content_patterns = [
        ("main", None),
        ("article", None),
        (None, {"role": "main"}),
        ("div", {"class": re.compile(r"content|article|post|entry|page-body", re.I)}),
        ("div", {"id": re.compile(r"content|article|main|post", re.I)}),
        ("section", {"class": re.compile(r"content|article|post", re.I)}),
    ]

    for tag_name, attrs in content_patterns:
        if tag_name and attrs:
            elements = soup.find_all(tag_name, attrs)
        elif tag_name:
            elements = soup.find_all(tag_name)
        else:
            elements = soup.find_all(attrs=attrs)

        for el in elements:
            if tag_name and el.get("class"):
                selectors.append(f"{tag_name}.{el['class'][0]}")
            elif tag_name and el.get("id"):
                selectors.append(f"{tag_name}#{el['id']}")
            elif tag_name:
                selectors.append(tag_name)
            elif el.get("role"):
                selectors.append(f"[role='{el['role']}']")

    return list(dict.fromkeys(selectors))[:10]  # deduplicate, max 10


def _encode_sparse(bm25_model: SparseTextEmbedding, texts: List[str]) -> List[SparseVector]:
    """Batch encode documents into Qdrant SparseVector objects."""
    return [
        SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
        for emb in bm25_model.embed(texts)
    ]


def _encode_sparse_query(bm25_model: SparseTextEmbedding, text: str) -> SparseVector:
    """Encode a single query with BM25 query weighting."""
    emb = list(bm25_model.query_embed(text))[0]
    return SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())


def _ensure_collection(qdrant: QdrantClient, name: str, embed_dim: int):
    """Create a collection with hybrid search if it doesn't exist."""
    collections = [c.name for c in qdrant.get_collections().collections]
    if name in collections:
        info = qdrant.get_collection(name)
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict) and "dense" in vectors_config:
            return  # Already has hybrid config
        # Old format — recreate
        qdrant.delete_collection(name)

    qdrant.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(size=embed_dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )
    logger.info(f"Created hybrid collection: {name}")


# ── MCP Tools (direct Qdrant) ─────────────────────────────────────


@mcp.tool()
async def search_collection(
    collection: str,
    query: str,
    ctx: Context[ServerSession, AppContext],
    top_k: int = 10,
) -> str:
    """Hybrid search (dense + sparse BM25 with RRF fusion) on a Qdrant collection.
    Returns the most relevant text chunks matching the query."""

    app = _get_ctx(ctx)

    try:
        collections = [c.name for c in app.qdrant.get_collections().collections]
        if collection not in collections:
            return json.dumps({"error": f"Collection '{collection}' not found", "available": collections})
    except Exception as e:
        return json.dumps({"error": f"Qdrant error: {str(e)[:200]}"})

    # Dense embedding with BGE query prefix
    prefixed_query = f"{BGE_QUERY_PREFIX}{query}"
    dense_vector = app.embed_model.encode(prefixed_query, normalize_embeddings=True).tolist()

    # Sparse embedding
    sparse_vector = _encode_sparse_query(app.bm25_model, query)

    # Hybrid query with RRF fusion
    try:
        results = app.qdrant.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(query=dense_vector, using="dense", limit=top_k * 3),
                Prefetch(query=sparse_vector, using="sparse", limit=top_k * 3),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)[:200]}"})

    _PRODUCT_KEYS = (
        "product_name", "product_price", "product_currency", "product_sku",
        "product_ean", "product_brand", "product_image", "product_availability",
        "product_description", "category_path",
    )
    hits = []
    for r in results.points:
        item = {
            "url": r.payload.get("url", ""),
            "title": r.payload.get("title", ""),
            "domain": r.payload.get("domain", ""),
            "text": r.payload.get("text", ""),
            "chunk_idx": r.payload.get("chunk_idx", 0),
            "score": round(r.score, 4),
        }
        for key in _PRODUCT_KEYS:
            val = r.payload.get(key)
            if val:
                item[key] = val
        hits.append(item)

    return json.dumps({
        "collection": collection,
        "query": query,
        "results": hits,
        "total_results": len(hits),
    }, indent=2)


@mcp.tool()
async def delete_collection(
    name: str,
    ctx: Context[ServerSession, AppContext],
) -> str:
    """Delete a Qdrant collection and all its data. This action is irreversible."""

    app = _get_ctx(ctx)

    try:
        collections = [c.name for c in app.qdrant.get_collections().collections]
        if name not in collections:
            return json.dumps({"error": f"Collection '{name}' not found", "available": collections})

        app.qdrant.delete_collection(name)
        return json.dumps({"status": "deleted", "name": name})

    except Exception as e:
        return json.dumps({"error": f"Failed to delete collection: {str(e)[:200]}"})


# ── RAG Tools (pocharlies API + direct Qdrant) ───────────────────


async def _pocharlies_post(path: str, payload: dict, timeout: float = 30.0) -> dict:
    """POST to pocharlies RAG service and return JSON response."""
    url = f"{POCHARLIES_RAG_URL}{path}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


async def _pocharlies_get(path: str, timeout: float = 10.0, params: Optional[Dict] = None) -> dict:
    """GET from pocharlies RAG service and return JSON response."""
    url = f"{POCHARLIES_RAG_URL}{path}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


async def _pocharlies_delete(path: str, timeout: float = 10.0) -> dict:
    """DELETE on pocharlies RAG service and return JSON response."""
    url = f"{POCHARLIES_RAG_URL}{path}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.delete(url)
        resp.raise_for_status()
        return resp.json()


def _connect_error_msg() -> str:
    return f"Cannot connect to pocharlies RAG service at {POCHARLIES_RAG_URL}. Is it running?"


# ── 1. Unified RAG Search ─────────────────────────────────────────


@mcp.tool()
async def rag_query(
    query: str,
    ctx: Context[ServerSession, AppContext],
    collection: str = "all",
    top_k: int = 10,
    domain: Optional[str] = None,
    brand: Optional[str] = None,
    category: Optional[str] = None,
) -> str:
    """RAG search: query → embed → search top-K chunks across all indexed collections.
    Returns the most relevant content from products, web pages, and competitors.
    Use collection='all' to search everywhere, or filter by: 'products', 'web', 'competitors'.
    This is the primary tool for retrieving context to augment your responses."""

    await ctx.info(f"RAG search: '{query}' in {collection} (top_k={top_k})...")

    try:
        payload: Dict = {
            "query": query,
            "top_k": top_k,
            "collection": collection,
            "rerank": True,
        }
        if domain:
            payload["domain"] = domain
        if brand:
            payload["brand"] = brand
        if category:
            payload["category"] = category

        data = await _pocharlies_post("/search", payload)
        results = data.get("results", [])

        if not results:
            return json.dumps({
                "query": query,
                "collection": collection,
                "results": [],
                "message": "No results found. Try a different query or check that collections are indexed.",
            })

        formatted = []
        for r in results:
            item: Dict = {
                "text": r.get("text", ""),
                "score": r.get("score", 0),
                "collection": r.get("collection", "unknown"),
            }
            for key in ("title", "url", "domain", "brand", "price"):
                if r.get(key):
                    item[key] = r[key]
            if r.get("path"):
                item["file_path"] = r["path"]
            formatted.append(item)

        return json.dumps({
            "query": query,
            "collection": collection,
            "total_results": len(formatted),
            "results": formatted,
        }, indent=2)

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"RAG search failed: {str(e)[:300]}"})


# ── 2. Product Catalog ────────────────────────────────────────────


@mcp.tool()
async def product_search(
    query: str,
    ctx: Context[ServerSession, AppContext],
    top_k: int = 5,
    brand: Optional[str] = None,
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
) -> str:
    """Search the Shopify product catalog using hybrid search (dense + sparse + reranking).
    Supports filtering by brand, category, and price range.
    Categories: gbb, aeg, sniper, pistol, shotgun, smg, launcher, magazine, battery, optic, ammunition, gear, protection, accessory.
    Returns product titles, descriptions, prices, brands, and availability."""

    await ctx.info(f"Product search: '{query}' (top_k={top_k})...")

    try:
        payload: Dict = {"query": query, "top_k": top_k, "rerank": True}
        if brand:
            payload["brand"] = brand
        if category:
            payload["category"] = category
        if min_price is not None:
            payload["min_price"] = min_price
        if max_price is not None:
            payload["max_price"] = max_price

        data = await _pocharlies_post("/products/search", payload)
        return json.dumps({
            "query": query,
            "total_results": len(data.get("results", [])),
            "results": data.get("results", []),
        }, indent=2)

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"Product search failed: {str(e)[:300]}"})


@mcp.tool()
async def product_sync(
    ctx: Context[ServerSession, AppContext],
    sync_type: str = "full",
    since: Optional[str] = None,
) -> str:
    """Start a product catalog sync from Shopify into Qdrant.
    sync_type='full' re-indexes all products. sync_type='incremental' only updates products modified since the given ISO timestamp.
    Requires Shopify client to be configured on the pocharlies service."""

    await ctx.info(f"Starting product sync ({sync_type})...")

    try:
        payload: Dict = {"sync_type": sync_type}
        if since:
            payload["since"] = since

        data = await _pocharlies_post("/products/sync", payload, timeout=600.0)
        return json.dumps(data, indent=2)

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"Product sync failed: {str(e)[:300]}"})


@mcp.tool()
async def product_stats(
    ctx: Context[ServerSession, AppContext],
) -> str:
    """Get product catalog statistics: total products indexed, collection info, brand/category distribution."""

    try:
        data = await _pocharlies_get("/products/stats")
        return json.dumps(data, indent=2)

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"Product stats failed: {str(e)[:300]}"})


# ── 3. Web Content Search ─────────────────────────────────────────


@mcp.tool()
async def web_search_indexed(
    query: str,
    ctx: Context[ServerSession, AppContext],
    top_k: int = 5,
    domain: Optional[str] = None,
) -> str:
    """Search previously crawled and indexed web pages using hybrid search.
    Returns relevant text chunks from websites that have been crawled into the web_pages collection.
    Optionally filter by domain (e.g., 'skirmshop.es')."""

    await ctx.info(f"Web search: '{query}' (top_k={top_k}, domain={domain})...")

    try:
        payload: Dict = {"query": query, "top_k": top_k, "rerank": True}
        if domain:
            payload["domain"] = domain

        data = await _pocharlies_post("/web/search", payload)
        return json.dumps({
            "query": query,
            "total_results": len(data.get("results", [])),
            "results": data.get("results", []),
        }, indent=2)

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"Web search failed: {str(e)[:300]}"})


@mcp.tool()
async def web_sources(
    ctx: Context[ServerSession, AppContext],
) -> str:
    """List all indexed web domains with their page counts and chunk counts.
    Use this to see what websites have been crawled and are available for search."""

    try:
        data = await _pocharlies_get("/web/sources")
        return json.dumps(data, indent=2)

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"Failed to list web sources: {str(e)[:300]}"})


@mcp.tool()
async def web_delete_source(
    domain: str,
    ctx: Context[ServerSession, AppContext],
) -> str:
    """Delete all indexed content from a specific web domain. This removes all chunks from web_pages collection for that domain."""

    await ctx.info(f"Deleting web source: {domain}...")

    try:
        data = await _pocharlies_delete(f"/web/source/{domain}")
        return json.dumps(data, indent=2)

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"Delete failed: {str(e)[:300]}"})


@mcp.tool()
async def web_crawl_start(
    url: str,
    ctx: Context[ServerSession, AppContext],
    max_depth: int = 2,
    max_pages: int = 50,
    smart_mode: bool = True,
) -> str:
    """Start crawling a website and index all pages into the web_pages collection.
    smart_mode=True uses LLM to analyze site structure and generate optimal CSS selectors.
    Returns a job_id for tracking progress with web_crawl_status."""

    await ctx.info(f"Starting web crawl: {url} (depth={max_depth}, max_pages={max_pages})...")

    try:
        payload = {
            "url": url,
            "max_depth": max_depth,
            "max_pages": max_pages,
            "smart_mode": smart_mode,
        }
        data = await _pocharlies_post("/web/index-url", payload)
        return json.dumps(data, indent=2)

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"Crawl start failed: {str(e)[:300]}"})


@mcp.tool()
async def web_crawl_status(
    job_id: str,
    ctx: Context[ServerSession, AppContext],
) -> str:
    """Get the current status of a web crawl job. Returns pages visited, pages indexed, chunks indexed, errors, and ETA."""

    try:
        data = await _pocharlies_get(f"/web/jobs")
        jobs = data.get("jobs", [])
        for job in jobs:
            if job.get("job_id") == job_id:
                return json.dumps(job, indent=2)
        return json.dumps({"error": f"Job {job_id} not found", "available_jobs": [j.get("job_id") for j in jobs]})

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"Status check failed: {str(e)[:300]}"})


@mcp.tool()
async def web_crawl_stop(
    job_id: str,
    ctx: Context[ServerSession, AppContext],
) -> str:
    """Stop a running or queued web crawl job. The job will halt gracefully after the current page."""

    try:
        data = await _pocharlies_post(f"/web/jobs/{job_id}/stop", {})
        return json.dumps(data, indent=2)

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"Stop failed: {str(e)[:300]}"})


# ── 4. Competitor Analysis ────────────────────────────────────────


@mcp.tool()
async def competitor_crawl(
    url: str,
    ctx: Context[ServerSession, AppContext],
    max_depth: int = 2,
    max_pages: int = 50,
    smart_mode: bool = True,
) -> str:
    """Start crawling a competitor website and index products into the competitor_products collection.
    Returns a job_id for tracking progress."""

    await ctx.info(f"Starting competitor crawl: {url}...")

    try:
        payload = {
            "url": url,
            "max_depth": max_depth,
            "max_pages": max_pages,
            "smart_mode": smart_mode,
        }
        data = await _pocharlies_post("/competitor/index-url", payload)
        return json.dumps(data, indent=2)

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"Competitor crawl failed: {str(e)[:300]}"})


@mcp.tool()
async def competitor_sources(
    ctx: Context[ServerSession, AppContext],
) -> str:
    """List all indexed competitor domains with stats."""

    try:
        data = await _pocharlies_get("/competitor/sources")
        return json.dumps(data, indent=2)

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"Failed: {str(e)[:300]}"})


@mcp.tool()
async def competitor_delete_source(
    domain: str,
    ctx: Context[ServerSession, AppContext],
) -> str:
    """Delete all indexed content from a competitor domain."""

    try:
        data = await _pocharlies_delete(f"/competitor/source/{domain}")
        return json.dumps(data, indent=2)

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"Delete failed: {str(e)[:300]}"})


@mcp.tool()
async def classify_competitor_products(
    domain: str,
    ctx: Context[ServerSession, AppContext],
    collection: str = "competitor_products",
    batch_size: int = 5,
) -> str:
    """Extract structured product data from crawled competitor pages using LLM.
    Analyzes web content chunks for a domain and extracts: name, category, brand, model, FPS, price, etc.
    Run this after crawling a competitor site."""

    await ctx.info(f"Classifying products from {domain}...")

    try:
        payload = {
            "domain": domain,
            "collection": collection,
            "batch_size": batch_size,
        }
        data = await _pocharlies_post("/classify/extract", payload, timeout=300.0)
        return json.dumps(data, indent=2)

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"Classification failed: {str(e)[:300]}"})


@mcp.tool()
async def compare_prices(
    domain: str,
    ctx: Context[ServerSession, AppContext],
) -> str:
    """Compare competitor product prices against your Shopify catalog.
    Runs entity resolution (embedding similarity) to match competitor products to yours,
    then generates a price report: cheaper, pricier, and matched products.
    Requires classify_competitor_products to have been run first for the domain."""

    await ctx.info(f"Comparing prices for {domain}...")

    try:
        payload = {"domain": domain}
        data = await _pocharlies_post("/classify/resolve", payload, timeout=120.0)
        return json.dumps(data, indent=2)

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"Price comparison failed: {str(e)[:300]}"})


# ── 5. Translation ────────────────────────────────────────────────


@mcp.tool()
async def translate_text(
    texts: List[str],
    target_lang: str,
    ctx: Context[ServerSession, AppContext],
    source_lang: str = "es",
    use_rag: bool = True,
) -> str:
    """Translate texts using the DGX LLM with optional RAG context for terminology consistency.
    Supports: en, es, ca (Catalan), fr, de, it, pt, eu (Basque), gl (Galician).
    When use_rag=True, similar products from Qdrant are used as terminology reference.
    Preserves brand names, model numbers, and technical specs (FPS, caliber)."""

    await ctx.info(f"Translating {len(texts)} texts: {source_lang} → {target_lang} (RAG={use_rag})...")

    try:
        payload = {
            "texts": texts,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "use_rag": use_rag,
            "resource_type": "product",
        }

        data = await _pocharlies_post("/translate/batch", payload, timeout=120.0)

        return json.dumps({
            "source_lang": source_lang,
            "target_lang": target_lang,
            "use_rag": use_rag,
            "translations": data.get("translations", []),
            "job_id": data.get("job_id"),
        }, indent=2)

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"Translation failed: {str(e)[:300]}"})


@mcp.tool()
async def normalize_specs(
    product: dict,
    ctx: Context[ServerSession, AppContext],
) -> str:
    """Normalize product specifications: convert units (FPS↔m/s, oz↔g, lbs↔kg, inches↔mm)
    and canonicalize brand names (e.g., 'TM' → 'Tokyo Marui', 'WE' → 'WE-Tech').
    Pass a product dict with fields like fps, weight, brand, etc."""

    try:
        data = await _pocharlies_post("/translate/normalize", product)
        return json.dumps(data, indent=2)

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"Normalization failed: {str(e)[:300]}"})


# ── 6. Chat with RAG Context ──────────────────────────────────────


@mcp.tool()
async def chat_with_rag(
    query: str,
    ctx: Context[ServerSession, AppContext],
    use_rag: bool = True,
    max_tokens: int = 4096,
) -> str:
    """Chat with the DGX LLM using RAG-augmented context from all indexed collections.
    Automatically searches products, web pages, and competitors,
    then injects the most relevant context into the LLM prompt.
    Use this for complex questions that benefit from indexed knowledge."""

    await ctx.info(f"RAG chat: '{query[:60]}...' (rag={use_rag})...")

    try:
        payload: Dict = {
            "query": query,
            "use_rag": use_rag,
            "use_tools": False,
            "max_tokens": max_tokens,
        }

        data = await _pocharlies_post("/chat", payload, timeout=120.0)

        return json.dumps({
            "response": data.get("response", ""),
            "model": data.get("model", ""),
            "tools_used": data.get("tools_used", []),
        }, indent=2)

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"Chat failed: {str(e)[:300]}"})


# ── 7. Direct Qdrant Operations ───────────────────────────────────


@mcp.tool()
async def index_text(
    collection: str,
    text: str,
    ctx: Context[ServerSession, AppContext],
    title: str = "",
    url: str = "",
    metadata: dict | None = None,
) -> str:
    """Index arbitrary text directly into a Qdrant collection with hybrid embeddings (dense BGE + sparse BM25).
    Use this to store knowledge, notes, documentation, or any text for later RAG retrieval.
    The text is chunked (1200 chars, 150 overlap), embedded, and upserted to Qdrant."""

    app = _get_ctx(ctx)

    if not text or len(text.strip()) < 50:
        return json.dumps({"error": "Text must be at least 50 characters"})

    try:
        _ensure_collection(app.qdrant, collection, app.embed_dim)
    except Exception as e:
        return json.dumps({"error": f"Collection error: {str(e)[:200]}"})

    chunks = _chunk_text(text.strip())
    if not chunks:
        return json.dumps({"error": "No chunks produced from text"})

    await ctx.info(f"Indexing {len(chunks)} chunks into '{collection}'...")

    texts_list = [c["text"] for c in chunks]
    dense_embeddings = app.embed_model.encode(texts_list, normalize_embeddings=True, batch_size=len(texts_list))
    sparse_embeddings = _encode_sparse(app.bm25_model, texts_list)

    now = datetime.now(timezone.utc).isoformat()
    source_url = url or f"manual:{collection}:{now}"
    points = []
    for i, chunk in enumerate(chunks):
        payload = {
            "url": source_url,
            "title": title,
            "text": chunk["text"],
            "chunk_idx": chunk["chunk_idx"],
            "fetch_date": now,
            "source_type": "manual",
        }
        if metadata:
            payload.update(metadata)

        points.append(PointStruct(
            id=_generate_id(collection, source_url, chunk["chunk_idx"]),
            vector={
                "dense": dense_embeddings[i].tolist(),
                "sparse": sparse_embeddings[i],
            },
            payload=payload,
        ))

    try:
        app.qdrant.upsert(collection_name=collection, points=points)
    except Exception as e:
        return json.dumps({"error": f"Upsert failed: {str(e)[:200]}"})

    return json.dumps({
        "status": "indexed",
        "collection": collection,
        "title": title,
        "chunks_indexed": len(points),
        "text_length": len(text),
    })


@mcp.tool()
async def get_collection_info(
    name: str,
    ctx: Context[ServerSession, AppContext],
    sample_query: Optional[str] = None,
) -> str:
    """Get detailed information about a Qdrant collection including point count,
    vector configuration, and optionally a sample search to verify content.
    Use this to inspect what data is stored in a collection before searching."""

    app = _get_ctx(ctx)

    try:
        collections = [c.name for c in app.qdrant.get_collections().collections]
        if name not in collections:
            return json.dumps({"error": f"Collection '{name}' not found", "available": collections})

        info = app.qdrant.get_collection(name)
        result: Dict = {
            "name": name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": str(info.status),
        }

        if sample_query and info.points_count > 0:
            prefixed_query = f"{BGE_QUERY_PREFIX}{sample_query}"
            dense_vector = app.embed_model.encode(prefixed_query, normalize_embeddings=True).tolist()
            sparse_vector = _encode_sparse_query(app.bm25_model, sample_query)

            sample_results = app.qdrant.query_points(
                collection_name=name,
                prefetch=[
                    Prefetch(query=dense_vector, using="dense", limit=9),
                    Prefetch(query=sparse_vector, using="sparse", limit=9),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=3,
                with_payload=True,
            )

            result["sample_results"] = [
                {
                    "title": r.payload.get("title", ""),
                    "text": (r.payload.get("text", ""))[:200],
                    "url": r.payload.get("url", ""),
                    "score": round(r.score, 4),
                }
                for r in sample_results.points
            ]

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Failed to get collection info: {str(e)[:200]}"})


# ── 8. Health & Stats ─────────────────────────────────────────────


@mcp.tool()
async def rag_health(
    ctx: Context[ServerSession, AppContext],
) -> str:
    """Check the health of the entire RAG infrastructure: Qdrant connectivity, pocharlies service status,
    all collection stats (point counts), and available services (reranker, shopify, products).
    Use this to diagnose connectivity issues or verify the system is ready."""

    app = _get_ctx(ctx)
    result: Dict = {"qdrant": {}, "pocharlies": {}, "collections": []}

    # Check Qdrant directly
    try:
        collections_list = app.qdrant.get_collections().collections
        result["qdrant"]["status"] = "healthy"
        result["qdrant"]["url"] = QDRANT_URL
        for col in collections_list:
            try:
                info = app.qdrant.get_collection(col.name)
                result["collections"].append({
                    "name": col.name,
                    "points_count": info.points_count,
                    "status": str(info.status),
                })
            except Exception:
                result["collections"].append({"name": col.name, "status": "error"})
    except Exception as e:
        result["qdrant"]["status"] = "unreachable"
        result["qdrant"]["error"] = str(e)[:200]

    # Check pocharlies RAG service
    try:
        health = await _pocharlies_get("/health")
        result["pocharlies"]["status"] = health.get("status", "unknown")
        result["pocharlies"]["url"] = POCHARLIES_RAG_URL
        result["pocharlies"]["services"] = health.get("services", {})
        result["pocharlies"]["qdrant_version"] = health.get("qdrant", {}).get("version")
    except httpx.ConnectError:
        result["pocharlies"]["status"] = "unreachable"
        result["pocharlies"]["url"] = POCHARLIES_RAG_URL
    except Exception as e:
        result["pocharlies"]["status"] = "error"
        result["pocharlies"]["error"] = str(e)[:200]

    return json.dumps(result, indent=2)


@mcp.tool()
async def collection_stats(
    ctx: Context[ServerSession, AppContext],
) -> str:
    """Get statistics for ALL Qdrant collections: web_pages, product_catalog,
    competitor_products. Shows point counts, vector counts, and status for each."""

    try:
        data = await _pocharlies_get("/web/collections")
        return json.dumps(data, indent=2)

    except httpx.ConnectError:
        return json.dumps({"error": _connect_error_msg()})
    except Exception as e:
        return json.dumps({"error": f"Stats failed: {str(e)[:300]}"})


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)
