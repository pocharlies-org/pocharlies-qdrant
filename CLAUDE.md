# Pocharlies RAG — Brain Engine for Skirmshop

## Architecture

```
brain.e-dani.com (OAuth2) → nginx → Dashboard (:5001)
rag.e-dani.com/mcp/sse   → nginx → MCP Server (:8002)
                                  → RAG Service (:5000)
                                  → Picqer MCP  (:8003)
                                  → Qdrant      (:6333)
                                  → Redis       (internal)
```

All services run as Docker containers on the skirmshop server (`pocharlies-*`).

## Containers

| Container | Port | Purpose |
|-----------|------|---------|
| pocharlies-rag | 5000 | RAG API — search, crawl, translate, suppliers, chat |
| pocharlies-mcp | 8002 | MCP SSE server for OpenClaw/Claude |
| pocharlies-picqer-mcp | 8003 | Picqer WMS MCP bridge |
| pocharlies-qdrant | 6333 | Qdrant vector database |
| pocharlies-redis | internal | Cache, reranker, sync state |
| pocharlies-oauth2-proxy | 4180 | Google OAuth2 for *.e-dani.com |

## Qdrant Collections (15)

| Collection | Points | Purpose |
|------------|--------|---------|
| skirmshop_documents | 29,754 | Chunked documents from knowledge vault |
| skirmshop_concepts | 29,353 | Concept-level embeddings for semantic search |
| competitor_products_v2 | 19,936 | Competitor product data (8 competitors) |
| skirmshop_products_v2 | 2,701 | Shopify product catalog (synced via webhook) |
| product_catalog | 2,513 | Legacy product embeddings |
| competitor_products | 1,111 | Legacy competitor data |
| knowledge_brain | 307 | Knowledge notes (competitor, brand, market, strategy) |
| supplier_registry | 8 | Supplier profiles and metadata |
| supplier_products | 2 | Supplier product prices for margin analysis |
| web_pages | 0 | Crawled web content |
| emails_v2 | 0 | Indexed emails |
| product_pages | 0 | Product page HTML content |
| supplier_pages | 0 | Supplier website pages |
| skirmshop_vault | 0 | Vault storage |
| product_collections | 0 | Collection metadata |

## RAG Service Modules (35 Python files)

### Core
| File | Purpose |
|------|---------|
| app.py | FastAPI app — 87 endpoints, scheduler, startup |
| bgem3_encoder.py | BGE-M3 embedding encoder |
| sparse_encoder.py | Sparse vector encoding |
| reranker.py | BGE-reranker-v2-m3 for result ranking |
| qdrant_utils.py | Qdrant client wrapper |
| metrics.py | Prometheus metrics |

### Product Intelligence
| File | Purpose |
|------|---------|
| product_indexer.py | Shopify → Qdrant product sync |
| catalog_indexer.py | Full catalog indexing |
| shopify_client.py | Shopify REST API client |
| shopify_graphql.py | Shopify GraphQL client |
| product_classifier.py | AI product classification |
| fast_product_extractor.py | Extract products from HTML/markdown |
| compatibility_analyzer.py | Airsoft part compatibility analysis |
| compatibility_data.py | Compatibility rules database |
| webhook_handler.py | Shopify webhook processing |

### Competitive Intelligence
| File | Purpose |
|------|---------|
| web_indexer.py | Website crawling + indexing |
| firecrawl_client.py | Firecrawl API wrapper |

### Supplier Intelligence
| File | Purpose |
|------|---------|
| supplier_registry.py | Supplier CRUD + profile management |
| supplier_indexer.py | Index supplier products + prices |
| supplier_email_monitor.py | Monitor supplier emails for orders/promos |
| authenticated_crawler.py | Login + cookie-based B2B portal crawling |
| margin_analyzer.py | Margin analysis (wholesale vs retail vs competitor) |
| purchase_recommender.py | Purchase order generation + gap analysis |
| pricelist_parser.py | Parse supplier price list documents |

### Knowledge System
| File | Purpose |
|------|---------|
| vault_builder.py | Build knowledge vault from sources |
| vault_indexer.py | Index vault into Qdrant |
| content_learner.py | Learn from URLs and documents |
| knowledge_synthesizer.py | Synthesize knowledge across sources |
| deep_analyzer.py | Deep content analysis |
| research_agent.py | Multi-step research agent |

### Translation & Classification
| File | Purpose |
|------|---------|
| translator.py | Airsoft-aware translation (EN↔ES) with glossary |
| glossary_data.py | Airsoft terminology database |

### Utilities
| File | Purpose |
|------|---------|
| activity_logger.py | Activity logging |
| sync_state.py | Product sync state management |
| qdrant_overhaul.py | Collection migration/overhaul |

## API Endpoints (87)

### Products (8)
- `POST /products/search` — Semantic product search with filters
- `POST /products/sync` — Trigger Shopify→Qdrant sync
- `GET /products/stats` — Catalog statistics
- `GET /products/sync/history` — Sync job history
- `GET /products/sync/{job_id}` — Sync job status
- `GET /products/{id_or_handle}` — Product details
- `GET /products/{id_or_handle}/inventory` — Stock levels
- `POST /webhooks/shopify` — Shopify product webhooks

### Catalog & Collections (4)
- `POST /catalog/full-sync` — Full catalog reindex
- `POST /catalog/search` — Catalog search
- `POST /collections/search` — Collection search
- `GET /collections/{id_or_handle}/products` — Products in collection

### Competitors (4)
- `POST /competitor/index-url` — Crawl and index competitor
- `DELETE /competitor/source/{domain}` — Remove competitor
- `GET /competitor/sources` — List indexed competitors

### Suppliers (21)
- `POST /supplier/registry` — Register new supplier
- `GET /supplier/registry` — List all suppliers
- `GET /supplier/registry/{slug}` — Supplier details
- `DELETE /supplier/registry/{slug}` — Remove supplier
- `POST /supplier/crawl` — Crawl supplier website
- `POST /supplier/crawl-authenticated` — Login + crawl B2B portal
- `POST /supplier/upload-pricelist` — Upload price list document
- `POST /supplier/match/{supplier_slug}` — Match supplier products to catalog
- `POST /supplier/products/search` — Search supplier products
- `GET /supplier/products/stats` — Supplier product counts
- `DELETE /supplier/products/{supplier_slug}` — Remove supplier products
- `GET /supplier/margins` — Margin analysis
- `GET /supplier/margin-alerts` — Low margin alerts
- `GET /supplier/compare` — Cross-supplier price comparison
- `GET /supplier/recommendations` — Purchase recommendations
- `GET /supplier/recommendations/gaps` — Products competitors have, we don't
- `GET /supplier/recommendations/delist` — Products to consider delisting
- `POST /supplier/recommendations/purchase-order` — Generate purchase order
- `POST /supplier/email-check` — Check supplier emails
- `GET /supplier/email-digest` — Email summary digest
- `GET /supplier/stats` — Overall supplier stats
- `GET /supplier/search` — Search suppliers

### Knowledge Brain (11)
- `POST /knowledge/learn` — Learn from URL or document
- `POST /knowledge/research` — Multi-step research
- `POST /knowledge/synthesize` — Synthesize across sources
- `POST /knowledge/deep-analyze` — Deep content analysis
- `GET /knowledge/notes` — List all notes
- `GET /knowledge/note/{path}` — Read specific note
- `POST /knowledge/rebuild` — Rebuild all knowledge
- `POST /knowledge/rebuild/{domain}` — Rebuild domain
- `POST /knowledge/reindex` — Reindex vault
- `GET /knowledge/status` — Knowledge system status

### Web Crawler (8)
- `POST /web/index-url` — Crawl and index URL
- `POST /web/search` — Search crawled content
- `GET /web/sources` — List indexed domains
- `DELETE /web/source/{domain}` — Remove domain
- `GET /web/jobs` — List crawl jobs
- `GET /web/status/{job_id}` — Job status
- `POST /web/jobs/{job_id}/stop` — Stop job
- `POST /web/jobs/{job_id}/resume` — Resume job
- `GET /web/logs/{job_id}` — Job logs
- `GET /web/collections` — Web collections

### Translation (3)
- `POST /translate/batch` — Batch translate with airsoft glossary
- `POST /translate/normalize` — Normalize specs (FPS→m/s, oz→g)
- `GET /translate/status/{job_id}` — Translation job status

### Classification (4)
- `POST /classify/extract` — Extract product data from text
- `POST /classify/resolve` — Resolve classification conflicts
- `GET /classify/status/{job_id}` — Classification status
- `GET /classify/products/{job_id}` — Classification results

### Chat (1)
- `POST /chat` — RAG-powered chat (uses DGX LLM)

### Search (1)
- `POST /search` — Unified multi-collection search

### Orders (3)
- `GET /orders` — Recent orders
- `GET /orders/{id_or_name}` — Order details
- `GET /orders/{id_or_name}/fulfillments` — Fulfillment status

### Glossary (5)
- `GET /glossary` — List all terms
- `POST /glossary` — Add term
- `POST /glossary/bulk` — Bulk add terms
- `DELETE /glossary/{term}` — Remove term
- `GET /glossary/languages` — Supported languages
- `GET /glossary/test` — Test glossary

### Agent (A2A Protocol) (8)
- `POST /agent/task` — Create agent task
- `GET /agent/tasks` — List tasks
- `GET /agent/task/{task_id}` — Task details
- `POST /agent/task/{task_id}/message` — Send message to task
- `POST /agent/task/{task_id}/continue` — Continue task
- `GET /agent/task/{task_id}/steps` — Task steps
- `GET /agent/task/{task_id}/logs` — Task logs
- `GET /agent/task/{task_id}/stream` — SSE stream
- `GET /agent/task/{task_id}/history` — Task history
- `GET /agent/status` — Agent status

### Admin (3)
- `POST /admin/compatibility/analyze-all` — Analyze all product compatibility
- `POST /admin/compatibility/analyze-product/{shopify_id}` — Analyze single product
- `GET /admin/compatibility/stats` — Compatibility stats

### Dashboard (2)
- `GET /dashboard/api/timeline` — Activity timeline
- `GET /dashboard/api/collections` — Collections overview

### System (3)
- `GET /health` — Full health check
- `GET /metrics` — Prometheus metrics
- `GET /api` — API documentation

## Suppliers (8)

| Supplier | Status | Type | Country | Contact |
|----------|--------|------|---------|---------|
| Silverback Airsoft | active | Direct manufacturer | Hong Kong | kto@silverback-airsoft.com |
| 3P-Store | active | Wholesaler | China | service@3p-store.com |
| AGM | active | Distributor | Spain | — |
| Taiwangun | potential | Wholesaler | Poland | krakman@taiwangun.com |
| Gunfire | potential | Wholesaler | Poland | b2b@gfcorp.pl |
| AirsoftZone | potential | Competitor/supplier | Slovenia | — |
| Anareus | potential | Competitor/supplier | Czech Republic | — |
| VSGUN | potential | Competitor/supplier | Spain | — |

## Competitors Tracked (8)

Airsoft Quimera, Hobby Expert, VSGUN, Zulu Tactical, Airsoft Estartit, Airsoft Nation Store, Airsoft Yecla, Weapon762

## External Integrations

| System | Integration |
|--------|-------------|
| Shopify | REST + GraphQL API, webhooks |
| Picqer WMS | MCP bridge (port 8003) |
| DGX Server | vLLM (Qwen3.5-35B-A3B + LoRA) for chat/translation/classification |
| Firecrawl | Local instance (:3003) for web crawling |
| 1Password | CLI (`op`) for supplier credentials |
| Gmail | Supplier email monitoring |

## Key URLs

| URL | Purpose |
|-----|---------|
| brain.e-dani.com | Dashboard (OAuth2 protected) |
| rag.e-dani.com | RAG API + MCP SSE endpoint |
| qdrant.e-dani.com | Qdrant Web UI |

## Development

```bash
# SSH to server
ssh skirmshop

# Service directory
cd /home/ubuntu/skirmshop/pocharlies-qdrant

# Logs
docker logs pocharlies-rag -f --tail 100

# Quick code update (no rebuild)
docker cp rag-service/file.py pocharlies-rag:/app/file.py
docker restart pocharlies-rag

# Full rebuild
docker compose build rag-service && docker compose up -d rag-service

# Health check
curl http://localhost:5000/health | python3 -m json.tool
```
