"""
Supplier Product Indexer for Pocharlies RAG
Crawls B2B supplier sites and indexes products with wholesale prices into Qdrant.
Reuses WebIndexer for crawling and FastProductExtractor for product extraction.
"""

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable

from qdrant_client.http.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
    SparseVectorParams, SparseIndexParams,
    Prefetch, FusionQuery, Fusion,
)
import bgem3_encoder

from urllib.parse import urlparse
from qdrant_utils import make_qdrant_client

logger = logging.getLogger(__name__)

COLLECTION_NAME = "supplier_products"


def _product_point_id(supplier_slug: str, name: str, url: str = "") -> int:
    """Deterministic point ID for a supplier product."""
    key = f"supplier_product:{supplier_slug}:{name.lower().strip()}:{url}"
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:16], 16)


class SupplierIndexer:
    """Indexes supplier products into Qdrant with wholesale price data."""

    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        model=None,
        embedding_model: str = "BAAI/bge-m3",
        supplier_registry=None,
    ):
        self.client = make_qdrant_client(qdrant_url, qdrant_api_key)
        self._qdrant_url = qdrant_url.rstrip("/")
        self._qdrant_api_key = qdrant_api_key
        self.supplier_registry = supplier_registry

        self.dim = bgem3_encoder.DENSE_DIM
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME in collections:
            info = self.client.get_collection(COLLECTION_NAME)
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict) and "dense" in vectors_config:
                return
            logger.warning(f"Recreating {COLLECTION_NAME} with named vectors")
            self.client.delete_collection(COLLECTION_NAME)

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=self.dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        logger.info(f"Created collection: {COLLECTION_NAME}")

        for field_name, schema in [
            ("supplier_slug", "keyword"),
            ("brand", "keyword"),
            ("domain", "keyword"),
            ("source_type", "keyword"),
            ("category", "keyword"),
        ]:
            try:
                self.client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception:
                pass

    def index_products(self, products: List[dict], supplier_slug: str, source_type: str = "web_crawl") -> int:
        """Index a list of extracted products into supplier_products collection."""

        if not products:
            return 0

        texts = []
        payloads = []
        point_ids = []

        now = datetime.now(timezone.utc).isoformat()

        for p in products:
            name = p.get("name", "").strip()
            if not name:
                continue

            text = f"{name} | {p.get('brand', '')} | {p.get('category', '')}"
            texts.append(text)

            payload = {
                "text": text,
                "title": name,
                "url": p.get("source_url", ""),
                "domain": p.get("source_domain", ""),
                "supplier_slug": supplier_slug,
                "brand": p.get("brand", ""),
                "category": p.get("category", ""),
                "retail_price": p.get("price"),
                "wholesale_price": p.get("wholesale_price"),
                "currency": p.get("currency", "EUR"),
                "sku": p.get("sku", ""),
                "availability": "in_stock" if p.get("in_stock") else "unknown",
                "source_type": source_type,
                "indexed_at": now,
                "matched_our_product": "",
                "matched_competitor_product": "",
                "extraction_method": p.get("extraction_method", ""),
                "confidence": p.get("confidence", 0),
            }
            payloads.append(payload)
            point_ids.append(_product_point_id(supplier_slug, name, p.get("source_url", "")))

        if not texts:
            return 0

        # Embed via BGE-M3 TEI
        dense_embeddings, sparse_embeddings = bgem3_encoder.encode_both(texts)

        # Build points
        points = []
        for i, (payload, pid) in enumerate(zip(payloads, point_ids)):
            points.append(PointStruct(
                id=pid,
                vector={
                    "dense": dense_embeddings[i].tolist(),
                    "sparse": sparse_embeddings[i],
                },
                payload=payload,
            ))

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i:i + batch_size],
            )

        logger.info(f"Indexed {len(points)} supplier products for {supplier_slug}")

        # Update registry counter
        if self.supplier_registry:
            total = self._count_supplier_products(supplier_slug)
            self.supplier_registry.update_products_indexed(supplier_slug, total)

        return len(points)

    def extract_and_index(self, domain: str, supplier_slug: str, crawl_collection: str = "supplier_products") -> int:
        """Extract products from crawled pages and re-index with structured data."""
        from fast_product_extractor import FastProductExtractor

        extractor = FastProductExtractor(self.client, self._qdrant_api_key)
        products = extractor.extract_products(domain, collection=crawl_collection)

        if not products:
            logger.warning(f"No products extracted for {domain}")
            return 0

        return self.index_products(products, supplier_slug, source_type="web_crawl")


    def index_from_pages(self, pages: List[dict], supplier_slug: str, is_b2b: bool = True) -> int:
        """Index products extracted from scraped pages (authenticated crawl).

        Args:
            pages: List of dicts with 'markdown', 'html', 'url', 'title'
            supplier_slug: Supplier identifier
            is_b2b: If True, treat extracted prices as wholesale prices
        """
        from fast_product_extractor import (
            _extract_json_ld, _extract_og_meta, _extract_from_trafilatura_text,
            extract_wholesale_price, is_b2b_domain,
        )

        all_products = []
        seen_names = set()

        for page in pages:
            url = page.get("url", "")
            title = page.get("title", "")
            html = page.get("html", "")
            markdown = page.get("markdown", "")
            domain = urlparse(url).netloc if url else ""

            # Try structured extraction from HTML
            if html:
                for jp in _extract_json_ld(html):
                    if jp.get("name") and jp["name"].lower() not in seen_names:
                        jp["source_domain"] = domain
                        jp["source_url"] = jp.get("source_url") or url
                        # On B2B sites, price IS wholesale price
                        if is_b2b and jp.get("price"):
                            jp["wholesale_price"] = jp["price"]
                        all_products.append(jp)
                        seen_names.add(jp["name"].lower())

                og = _extract_og_meta(html, url)
                if og and og.get("name") and og["name"].lower() not in seen_names:
                    og["source_domain"] = domain
                    if is_b2b and og.get("price"):
                        og["wholesale_price"] = og["price"]
                    all_products.append(og)
                    seen_names.add(og["name"].lower())

            # Try trafilatura-style extraction from markdown
            if markdown:
                p = _extract_from_trafilatura_text(markdown, url, title)
                if p and p.get("name") and p["name"].lower() not in seen_names:
                    p["source_domain"] = domain
                    if is_b2b and p.get("price"):
                        p["wholesale_price"] = p["price"]
                    # Also try explicit wholesale price detection
                    wp = extract_wholesale_price(markdown, url)
                    if wp:
                        p["wholesale_price"] = wp
                    all_products.append(p)
                    seen_names.add(p["name"].lower())

        if not all_products:
            logger.warning(f"No products extracted from {len(pages)} pages for {supplier_slug}")
            return 0

        logger.info(f"Extracted {len(all_products)} products from {len(pages)} pages for {supplier_slug}")
        return self.index_products(all_products, supplier_slug, source_type="authenticated_crawl")

    def match_to_our_catalog(self, supplier_slug: str, our_collection: str = "skirmshop_products_v2", threshold: float = 0.82) -> int:
        """Match supplier products to our catalog by embedding similarity."""
        supplier_products = self._scroll_supplier_products(supplier_slug)
        if not supplier_products:
            return 0

        our_client = self.client
        matched = 0

        for sp in supplier_products:
            name = sp.get("title", "")
            brand = sp.get("brand", "")
            query = f"{brand} {name}".strip()
            if not query:
                continue

            dense_emb = bgem3_encoder.encode_dense_query(query)

            try:
                results = our_client.search(
                    collection_name=our_collection,
                    query_vector=("dense", dense_emb),
                    limit=1,
                    with_payload=True,
                )
                if results and results[0].score >= threshold:
                    match = results[0]
                    handle = match.payload.get("handle", "")
                    point_id = _product_point_id(
                        supplier_slug,
                        sp.get("title", ""),
                        sp.get("url", ""),
                    )
                    self.client.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={"matched_our_product": handle},
                        points=[point_id],
                    )
                    matched += 1
            except Exception as e:
                logger.debug(f"Match error for '{query}': {e}")
                continue

        logger.info(f"Matched {matched}/{len(supplier_products)} supplier products to our catalog for {supplier_slug}")
        return matched

    def match_to_competitors(self, supplier_slug: str, competitor_collection: str = "competitor_products_v2", threshold: float = 0.80) -> int:
        """Match supplier products to competitor products."""
        supplier_products = self._scroll_supplier_products(supplier_slug)
        if not supplier_products:
            return 0

        matched = 0
        for sp in supplier_products:
            name = sp.get("title", "")
            brand = sp.get("brand", "")
            query = f"{brand} {name}".strip()
            if not query:
                continue

            dense_emb = bgem3_encoder.encode_dense_query(query)

            try:
                results = self.client.search(
                    collection_name=competitor_collection,
                    query_vector=("dense", dense_emb),
                    limit=1,
                    with_payload=True,
                )
                if results and results[0].score >= threshold:
                    match = results[0]
                    comp_url = match.payload.get("url", match.payload.get("source_url", ""))
                    point_id = _product_point_id(
                        supplier_slug,
                        sp.get("title", ""),
                        sp.get("url", ""),
                    )
                    self.client.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={"matched_competitor_product": comp_url},
                        points=[point_id],
                    )
                    matched += 1
            except Exception as e:
                logger.debug(f"Competitor match error for '{query}': {e}")
                continue

        logger.info(f"Matched {matched}/{len(supplier_products)} to competitors for {supplier_slug}")
        return matched

    def search(
        self,
        query: str,
        top_k: int = 10,
        supplier_slug: Optional[str] = None,
        brand: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[dict]:
        """Hybrid search on supplier products."""
        try:
            dense_emb = bgem3_encoder.encode_dense_query(query)
            sparse_emb = bgem3_encoder.encode_sparse_query(query)

            conditions = []
            if supplier_slug:
                conditions.append(FieldCondition(key="supplier_slug", match=MatchValue(value=supplier_slug)))
            if brand:
                conditions.append(FieldCondition(key="brand", match=MatchValue(value=brand)))

            search_filter = Filter(must=conditions) if conditions else None

            results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    Prefetch(query=dense_emb, using="dense", filter=search_filter, limit=top_k * 3),
                    Prefetch(query=sparse_emb, using="sparse", filter=search_filter, limit=top_k * 3),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
            return [
                {
                    "title": r.payload.get("title", ""),
                    "supplier_slug": r.payload.get("supplier_slug", ""),
                    "brand": r.payload.get("brand", ""),
                    "category": r.payload.get("category", ""),
                    "wholesale_price": r.payload.get("wholesale_price"),
                    "retail_price": r.payload.get("retail_price"),
                    "currency": r.payload.get("currency", "EUR"),
                    "url": r.payload.get("url", ""),
                    "sku": r.payload.get("sku", ""),
                    "availability": r.payload.get("availability", ""),
                    "source_type": r.payload.get("source_type", ""),
                    "matched_our_product": r.payload.get("matched_our_product", ""),
                    "matched_competitor_product": r.payload.get("matched_competitor_product", ""),
                    "score": round(r.score, 4),
                }
                for r in results.points
                if r.score >= min_score
            ]
        except Exception as e:
            logger.error(f"Supplier product search error: {e}")
            return []

    def _scroll_supplier_products(self, supplier_slug: str) -> List[dict]:
        """Get all products for a supplier."""
        all_products = []
        offset = None
        while True:
            try:
                results = self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="supplier_slug", match=MatchValue(value=supplier_slug))
                    ]),
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points, next_offset = results
                for p in points:
                    all_products.append(p.payload or {})
                if not next_offset:
                    break
                offset = next_offset
            except Exception as e:
                logger.error(f"Scroll error for {supplier_slug}: {e}")
                break
        return all_products

    def _count_supplier_products(self, supplier_slug: str) -> int:
        """Count products for a supplier."""
        try:
            result = self.client.count(
                collection_name=COLLECTION_NAME,
                count_filter=Filter(must=[
                    FieldCondition(key="supplier_slug", match=MatchValue(value=supplier_slug))
                ]),
            )
            return result.count
        except Exception:
            return 0

    def get_stats(self) -> dict:
        """Collection stats."""
        try:
            info = self.client.get_collection(COLLECTION_NAME)
            return {
                "name": COLLECTION_NAME,
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception:
            return {"name": COLLECTION_NAME, "points_count": 0, "status": "not_found"}

    def delete_supplier_products(self, supplier_slug: str) -> bool:
        """Delete all products for a supplier."""
        try:
            self.client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(must=[
                    FieldCondition(key="supplier_slug", match=MatchValue(value=supplier_slug))
                ]),
            )
            logger.info(f"Deleted all products for supplier: {supplier_slug}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete products for {supplier_slug}: {e}")
            return False
