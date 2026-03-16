"""Prometheus metrics for RAG service — knowledge brain monitoring."""

import os
import tempfile

# Set up multiprocess mode BEFORE importing prometheus_client
# This allows metrics to be shared across uvicorn workers
if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
    prom_dir = os.path.join(tempfile.gettempdir(), "prometheus_multiproc")
    os.makedirs(prom_dir, exist_ok=True)
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = prom_dir

from prometheus_client import Counter, Gauge, Histogram, Info, CONTENT_TYPE_LATEST
from prometheus_client import multiprocess, CollectorRegistry, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import time

# ── Request metrics ──────────────────────────────────────
REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "rag_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# ── Knowledge brain metrics ──────────────────────────────
VAULT_NOTES_TOTAL = Gauge(
    "rag_vault_notes_total",
    "Total notes in the knowledge vault",
    multiprocess_mode="liveall",
)
VAULT_RECOMMENDATION_NOTES = Gauge(
    "rag_vault_recommendation_notes",
    "Total recommendation notes generated",
    multiprocess_mode="liveall",
)
QDRANT_COLLECTION_POINTS = Gauge(
    "rag_qdrant_collection_points",
    "Points count per Qdrant collection",
    ["collection"],
    multiprocess_mode="liveall",
)

# ── Synthesis metrics ────────────────────────────────────
SYNTHESIS_RUNS = Counter(
    "rag_synthesis_runs_total",
    "Total synthesis runs",
    ["status"],  # success, failed
)
SYNTHESIS_NOTES_GENERATED = Counter(
    "rag_synthesis_notes_generated_total",
    "Total recommendation notes generated across all runs",
)
SYNTHESIS_LLM_CALLS = Counter(
    "rag_synthesis_llm_calls_total",
    "Total LLM calls made during synthesis",
)
SYNTHESIS_DURATION = Histogram(
    "rag_synthesis_duration_seconds",
    "Time taken for a full synthesis run",
    buckets=[10, 30, 60, 120, 300, 600, 1200, 1800],
)

# ── Rebuild metrics ──────────────────────────────────────
REBUILD_RUNS = Counter(
    "rag_rebuild_runs_total",
    "Total vault rebuild runs",
    ["status"],  # success, failed
)
REBUILD_DURATION = Histogram(
    "rag_rebuild_duration_seconds",
    "Time taken for a full vault rebuild",
    buckets=[60, 300, 600, 1200, 1800, 3600, 7200],
)
REBUILD_COMPETITORS_CRAWLED = Counter(
    "rag_rebuild_competitors_crawled_total",
    "Total competitor sites crawled",
)
REBUILD_PRODUCTS_EXTRACTED = Counter(
    "rag_rebuild_products_extracted_total",
    "Total products extracted from competitors",
)

# ── Search metrics ───────────────────────────────────────
SEARCH_COUNT = Counter(
    "rag_search_total",
    "Total search requests",
    ["search_type"],  # product, catalog, recommendation
)
SEARCH_LATENCY = Histogram(
    "rag_search_duration_seconds",
    "Search latency in seconds",
    ["search_type"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# ── Crawl metrics ────────────────────────────────────────
CRAWL_PAGES = Counter(
    "rag_crawl_pages_total",
    "Total pages crawled from competitors",
    ["competitor"],
)
CRAWL_DURATION = Histogram(
    "rag_crawl_duration_seconds",
    "Crawl duration per competitor",
    ["competitor"],
    buckets=[60, 300, 600, 1200, 1800, 3600],
)

# ── Build info ───────────────────────────────────────────
BUILD_INFO = Info("rag_build", "RAG service build information")
BUILD_INFO.info({"version": "2.0.0", "service": "pocharlies-rag"})


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Track request count and latency for all endpoints."""

    async def dispatch(self, request: Request, call_next):
        # Skip metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        # Normalize path to avoid high cardinality
        path = request.url.path.rstrip("/")
        # Collapse dynamic segments
        parts = path.split("/")
        normalized = "/".join(
            "<id>" if (i > 0 and parts[i-1] in ("note", "products", "collections")) else p
            for i, p in enumerate(parts)
        )

        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        REQUEST_COUNT.labels(method=method, endpoint=normalized, status=response.status_code).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=normalized).observe(duration)

        return response


def metrics_response():
    """Generate Prometheus metrics response (multiprocess-safe)."""
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST,
    )
