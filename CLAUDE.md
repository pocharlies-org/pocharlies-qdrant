# pocharlies-qdrant — RAG Platform for Skirmshop

## What This Is

Complete RAG (Retrieval-Augmented Generation) platform for Skirmshop airsoft e-commerce. Provides semantic search, competitor intelligence, product catalog indexing, knowledge synthesis, and AI agent orchestration.

## Architecture

```
                    ┌──────────────────┐
                    │   Ingress/TLS    │
                    │ (nginx + cert-mg)│
                    └───┬──────────┬───┘
                        │          │
              rag.domain    agent.domain
                        │          │
                ┌───────▼──┐  ┌───▼──────────┐
                │rag-service│  │agent-service  │
                │  :5000    │  │  :8100        │
                │  FastAPI  │  │  LangGraph    │
                └──┬──┬──┬─┘  └──┬────────────┘
                   │  │  │       │ MCP (SSE)
          ┌────────┘  │  └───┐   │
          ▼           ▼      ▼   ▼
     ┌────────┐  ┌───────┐  ┌──────────┐  ┌──────────┐
     │ Qdrant │  │ Redis │  │mcp-server│  │picqer-mcp│
     │ :6333  │  │ :6379 │  │  :8000   │  │  :8001   │
     └────────┘  └───────┘  └──────────┘  └──────────┘
                                             │
                             ┌───────────┐   │
                             │ PostgreSQL │   │ Picqer WMS API
                             │  :5432    │   │
                             └───────────┘
                    ┌──────────────┐
                    │ openclaw-mem │  (optional)
                    │    :8080     │
                    └──────────────┘

    External dependency: LLM endpoint (vLLM/LiteLLM) — not in this stack
```

## Services

| Service | Port | Image | Purpose |
|---------|------|-------|---------|
| **qdrant** | 6333, 6334 | `qdrant/qdrant:latest` | Vector DB (hybrid dense+sparse search) |
| **redis** | 6379 | `redis:7-alpine` | Cache, sessions, sync state, glossary |
| **postgres** | 5432 | `postgres:16-alpine` | LangGraph checkpointer state |
| **rag-service** | 5000 | `rag-service/Dockerfile` | Core RAG: crawling, products, translation, search (60+ endpoints) |
| **mcp-server** | 8000 | `mcp-server/Dockerfile` | MCP tool bridge (25+ tools via SSE) |
| **picqer-mcp** | 8001 | `mcp-server/Dockerfile.picqer` | Picqer WMS MCP server |
| **agent-service** | 8100 | `agent-service/Dockerfile` | LangGraph supervisor agent |
| **openclaw-mem** | 8080 | `openclaw-mem/Dockerfile` | Persistent memory (SQLite + FTS5) |

## Deployment

### Docker Compose (single server)

```bash
cp .env.example .env
# Edit .env with real credentials
docker compose up -d
```

### Kubernetes (cloud)

All manifests in `k8s/`. Uses Kustomize.

```bash
# 1. Edit secrets (or use Sealed Secrets / External Secrets Operator)
vi k8s/secrets.yaml

# 2. Adjust storageClassName if needed (default: fast-ssd)
#    grep -r storageClassName k8s/

# 3. Configure LLM endpoint in configmap.yaml
#    LLM_BASE_URL should point to your vLLM/LiteLLM instance

# 4. Deploy
kubectl apply -k k8s/

# 5. Verify
kubectl get pods -n pocharlies-rag
curl https://rag.yourdomain.com/health
```

**Resources:** ~2.5 CPU / 7Gi RAM minimum. No GPU needed (LLM is external).

**Prerequisites:**
- Kubernetes cluster with nginx-ingress and cert-manager
- Storage class `fast-ssd` (or edit to match your provider)
- External LLM endpoint (vLLM, LiteLLM, or OpenAI-compatible API)
- DNS records pointing to cluster ingress

### CI/CD

GitHub Actions workflow at `.github/workflows/build-images.yaml`:
- Builds per-service on push to main (detects which services changed)
- Pushes to `ghcr.io/pocharlies/{service}:latest` + `:sha`
- Manual trigger via `workflow_dispatch`

## Environment Variables

### Required (Secrets)

| Variable | Description |
|----------|-------------|
| `QDRANT_API_KEY` | Qdrant vector DB authentication |
| `LLM_API_KEY` | LLM endpoint API key |
| `SHOPIFY_ACCESS_TOKEN` | Shopify Admin API token (`shpat_...`) |
| `SHOPIFY_WEBHOOK_SECRET` | Shopify webhook HMAC validation |
| `POSTGRES_PASSWORD` | PostgreSQL password |
| `PICQER_API_KEY` | Picqer WMS API key |

### Configuration (ConfigMap)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | — | OpenAI-compatible LLM endpoint (`http://host:4000/v1`) |
| `LLM_MODEL` | `smart` | Model name for smart operations |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Dense embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker |
| `RERANKER_ENABLED` | `true` | Enable/disable reranking |
| `SHOPIFY_SHOP_DOMAIN` | — | Shopify store domain |
| `UVICORN_WORKERS` | `4` | RAG service worker count |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection |

## Key Paths

```
pocharlies-qdrant/
├── rag-service/           # Core RAG API (FastAPI, 60+ endpoints)
│   ├── app.py             # Main application
│   ├── product_indexer.py # Shopify product sync
│   ├── web_indexer.py     # Web crawling
│   ├── translator.py      # Multi-language translation
│   ├── glossary_data.py   # Airsoft term glossary (104KB)
│   └── Dockerfile
├── mcp-server/            # MCP tool bridge
│   ├── server.py          # 25+ tools (SSE transport)
│   ├── picqer_server.py   # Picqer WMS tools
│   └── Dockerfile
├── agent-service/         # LangGraph agent orchestrator
│   ├── main.py            # FastAPI + WebSocket
│   ├── graphs/supervisor.py
│   ├── mcp_client/mcp_servers.json  # MCP server URLs
│   └── Dockerfile
├── k8s/                   # Kubernetes manifests (Kustomize)
│   ├── kustomization.yaml # Entry point: kubectl apply -k k8s/
│   ├── configmap.yaml     # Shared configuration
│   ├── secrets.yaml       # Credential template
│   ├── qdrant/            # StatefulSet + PVC 50Gi
│   ├── redis/             # Deployment + PVC 2Gi
│   ├── postgres/          # StatefulSet + PVC 10Gi
│   ├── rag-service/       # Deployment + HPA (1-3) + vault ConfigMap
│   ├── mcp-server/        # Deployment
│   ├── picqer-mcp/        # Deployment
│   ├── agent-service/     # Deployment + Ingress + MCP config
│   ├── openclaw-mem/      # Deployment + PVC 5Gi
│   └── brain-engine/      # CronJob (weekly retraining via SSH)
├── vault_config.yaml      # Competitor + knowledge source config
├── docker-compose.yml     # Single-server deployment
└── .env.example           # Environment variable template
```

## Technology Stack

- **Language:** Python 3.11
- **API:** FastAPI + Uvicorn
- **Vector DB:** Qdrant (hybrid: BM25 sparse + BGE dense)
- **Reranking:** BGE cross-encoder (BAAI/bge-reranker-v2-m3)
- **Agent:** LangGraph + LangChain
- **MCP:** FastMCP (SSE transport)
- **Cache:** Redis 7 (AOF persistence)
- **State:** PostgreSQL 16 (LangGraph checkpoints)
- **Shopify:** GraphQL + REST Admin API
- **LLM:** Any OpenAI-compatible endpoint (vLLM, LiteLLM, etc.)

## Brain-Engine (LoRA Retraining)

Separate repo: `pocharlies/pocharlies-lora`

Weekly CronJob in `k8s/brain-engine/cronjob.yaml` SSHs to the DGX server to run QLoRA fine-tuning on Qwen3.5-35B-A3B with domain data from this RAG system.

Requires `dgx-ssh-key` secret:
```bash
kubectl create secret generic dgx-ssh-key -n pocharlies-rag \
  --from-file=id_ed25519=/path/to/key \
  --from-file=known_hosts=/path/to/known_hosts
```

## Health Checks

```bash
# RAG service
curl http://rag-service:5000/health

# Agent service
curl http://agent-service:8100/health

# Qdrant
curl http://qdrant:6333/collections

# Full validation
curl http://rag-service:5000/health && \
curl http://agent-service:8100/health && \
echo "All services healthy"
```
