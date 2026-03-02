"""
Product Catalog Indexer for RAG Pipeline
Indexes Shopify products into Qdrant and provides product search.
"""

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional, Callable

from qdrant_client.http.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
    SparseVectorParams, SparseIndexParams,
    Prefetch, FusionQuery, Fusion,
)
from sentence_transformers import SentenceTransformer

from qdrant_utils import make_qdrant_client

logger = logging.getLogger(__name__)


@dataclass
class ProductSyncJob:
    """Tracks a product catalog sync job. Follows CrawlJob pattern."""

    job_id: str
    status: str = "running"  # running | completed | failed
    sync_type: str = "full"  # full | incremental
    products_found: int = 0
    products_indexed: int = 0
    chunks_indexed: int = 0
    current_product: str = ""
    errors: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None

    MAX_LOGS = 3000

    def log(self, msg: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {msg}")
        if len(self.logs) > self.MAX_LOGS:
            self.logs = self.logs[:50] + self.logs[-(self.MAX_LOGS - 50):]

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "sync_type": self.sync_type,
            "products_found": self.products_found,
            "products_indexed": self.products_indexed,
            # Aliases expected by the frontend
            "total_products": self.products_found,
            "products_synced": self.products_indexed,
            "chunks_indexed": self.chunks_indexed,
            "current_product": self.current_product,
            "errors": self.errors[-10:],
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


class ProductIndexer:
    COLLECTION_NAME = "product_catalog"
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        model: Optional[SentenceTransformer] = None,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
    ):
        self.client = make_qdrant_client(qdrant_url, qdrant_api_key)
        self._qdrant_url = qdrant_url.rstrip("/")
        self._qdrant_api_key = qdrant_api_key

        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(embedding_model)

        self.dim = self.model.get_sentence_embedding_dimension()
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.COLLECTION_NAME in collections:
            info = self.client.get_collection(self.COLLECTION_NAME)
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict) and "dense" in vectors_config:
                return
            logger.warning(f"Recreating {self.COLLECTION_NAME} with named vectors")
            self.client.delete_collection(self.COLLECTION_NAME)

        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=self.dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        logger.info(f"Created hybrid collection: {self.COLLECTION_NAME}")

    @staticmethod
    def _generate_id(shopify_id: int, chunk_idx: int) -> int:
        key = f"product:{shopify_id}:{chunk_idx}"
        h = hashlib.sha256(key.encode()).hexdigest()
        return int(h[:16], 16)

    # ── Full sync ─────────────────────────────────────────────────

    async def index_all_products(
        self,
        shopify_client,
        progress_callback: Optional[Callable] = None,
        content_hash_store=None,
    ) -> ProductSyncJob:
        """Full catalog sync: fetch all products from Shopify, embed, store."""
        job = ProductSyncJob(
            job_id=uuid.uuid4().hex[:12],
            sync_type="full",
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            job.log("Fetching all products from Shopify...")
            if progress_callback:
                progress_callback(job)

            products = await shopify_client.list_products()
            job.products_found = len(products)
            job.log(f"Found {len(products)} products")

            if not products:
                job.status = "completed"
                job.ended_at = datetime.now(timezone.utc).isoformat()
                return job

            # Process in batches
            batch_size = 20
            for i in range(0, len(products), batch_size):
                batch = products[i:i + batch_size]
                points, skipped = await self._process_product_batch(batch, shopify_client, job, content_hash_store)

                if points:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda pts=points: self.client.upsert(
                            collection_name=self.COLLECTION_NAME,
                            points=pts,
                        )
                    )
                    job.chunks_indexed += len(points)
                    job.log(f"Upserted {len(points)} chunks, skipped {skipped} (total: {job.chunks_indexed})")

                if progress_callback:
                    progress_callback(job)

            job.status = "completed"
            job.log(f"COMPLETED: {job.products_indexed} products, {job.chunks_indexed} chunks indexed")

        except Exception as e:
            job.status = "failed"
            job.errors.append(f"Sync failed: {str(e)[:200]}")
            job.log(f"FAILED: {str(e)[:200]}")
            logger.error(f"Product sync failed: {e}")

        job.ended_at = datetime.now(timezone.utc).isoformat()
        if progress_callback:
            progress_callback(job)
        return job

    # ── Incremental sync ──────────────────────────────────────────

    async def incremental_sync(
        self,
        shopify_client,
        since: str,
        progress_callback: Optional[Callable] = None,
        content_hash_store=None,
    ) -> ProductSyncJob:
        """Incremental sync: only re-index products updated since `since`."""
        job = ProductSyncJob(
            job_id=uuid.uuid4().hex[:12],
            sync_type="incremental",
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            job.log(f"Fetching products updated since {since}...")
            products = await shopify_client.get_updated_since(since)
            job.products_found = len(products)
            job.log(f"Found {len(products)} updated products")

            if not products:
                job.status = "completed"
                job.ended_at = datetime.now(timezone.utc).isoformat()
                return job

            # Delete old points for these products, then re-index
            for product in products:
                shopify_id = product.get("id")
                if shopify_id:
                    try:
                        self.client.delete(
                            collection_name=self.COLLECTION_NAME,
                            points_selector=Filter(
                                must=[FieldCondition(
                                    key="shopify_id",
                                    match=MatchValue(value=shopify_id),
                                )]
                            ),
                        )
                    except Exception:
                        pass  # may not exist yet

            # Re-index
            batch_size = 20
            for i in range(0, len(products), batch_size):
                batch = products[i:i + batch_size]
                points, skipped = await self._process_product_batch(batch, shopify_client, job, content_hash_store)

                if points:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda pts=points: self.client.upsert(
                            collection_name=self.COLLECTION_NAME,
                            points=pts,
                        )
                    )
                    job.chunks_indexed += len(points)

                if progress_callback:
                    progress_callback(job)

            job.status = "completed"
            job.log(f"COMPLETED: {job.products_indexed} products, {job.chunks_indexed} chunks")

        except Exception as e:
            job.status = "failed"
            job.errors.append(str(e)[:200])
            job.log(f"FAILED: {str(e)[:200]}")

        job.ended_at = datetime.now(timezone.utc).isoformat()
        if progress_callback:
            progress_callback(job)
        return job

    # ── Batch processing ──────────────────────────────────────────

    async def _process_product_batch(
        self,
        products: List[dict],
        shopify_client,
        job: ProductSyncJob,
        content_hash_store=None,
    ) -> tuple:
        """Process a batch of products: extract text, embed, build points. Returns (points, skipped_count)."""
        from sparse_encoder import encode_sparse

        texts = []
        metadatas = []
        hash_items = []
        skipped = 0

        for product in products:
            text = shopify_client.extract_product_text(product)
            if not text or len(text) < 20:
                continue

            metadata = shopify_client.extract_metadata(product)

            # Content hash dedup
            item_key = f"product:{metadata.get('shopify_id', '')}"
            if content_hash_store:
                if not await content_hash_store.has_changed(item_key, text):
                    skipped += 1
                    continue

            texts.append(text)
            metadatas.append(metadata)
            hash_items.append((item_key, text))

            job.current_product = metadata.get("title", "")
            job.products_indexed += 1

        if not texts:
            return [], skipped

        # Embed (dense + sparse)
        loop = asyncio.get_event_loop()
        dense_embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=len(texts),
            )
        )
        sparse_embeddings = await loop.run_in_executor(
            None,
            lambda: encode_sparse(texts)
        )

        points = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            shopify_id = metadata.get("shopify_id", 0)
            point_id = self._generate_id(shopify_id, 0)

            payload = {
                "text": text,
                "source_type": "product",
                **metadata,
            }

            points.append(PointStruct(
                id=point_id,
                vector={
                    "dense": dense_embeddings[i].tolist(),
                    "sparse": sparse_embeddings[i],
                },
                payload=payload,
            ))

        # Update content hashes after successful embedding
        if content_hash_store:
            for item_key, text in hash_items:
                await content_hash_store.set_hash(item_key, text)

        return points, skipped

    # ── Search ────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        brand_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
    ) -> List[Dict]:
        """Hybrid search on product_catalog with structured filters."""
        from sparse_encoder import encode_sparse_query

        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.COLLECTION_NAME not in collections:
                return []

            prefixed_query = f"{self.BGE_QUERY_PREFIX}{query}"
            dense_embedding = self.model.encode(
                prefixed_query, normalize_embeddings=True
            ).tolist()
            sparse_embedding = encode_sparse_query(query)

            # Build filters
            conditions = []
            if brand_filter:
                conditions.append(
                    FieldCondition(key="brand", match=MatchValue(value=brand_filter))
                )
            if category_filter:
                conditions.append(
                    FieldCondition(key="category", match=MatchValue(value=category_filter))
                )
            if min_price is not None or max_price is not None:
                range_params = {}
                if min_price is not None:
                    range_params["gte"] = min_price
                if max_price is not None:
                    range_params["lte"] = max_price
                conditions.append(
                    FieldCondition(key="price", range=Range(**range_params))
                )

            search_filter = Filter(must=conditions) if conditions else None

            results = self.client.query_points(
                collection_name=self.COLLECTION_NAME,
                prefetch=[
                    Prefetch(query=dense_embedding, using="dense", filter=search_filter, limit=top_k * 3),
                    Prefetch(query=sparse_embedding, using="sparse", filter=search_filter, limit=top_k * 3),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )

            return [
                {
                    "title": r.payload.get("title", ""),
                    "handle": r.payload.get("handle", ""),
                    "brand": r.payload.get("brand", ""),
                    "category": r.payload.get("category", ""),
                    "price": r.payload.get("price", 0),
                    "url": r.payload.get("url", ""),
                    "sku": r.payload.get("sku", ""),
                    "text": r.payload.get("text", ""),
                    "fps": r.payload.get("fps"),
                    "material": r.payload.get("material"),
                    "inventory_quantity": r.payload.get("inventory_quantity", 0),
                    "status": r.payload.get("status", ""),
                    "score": round(r.score, 4),
                    "source_type": "product",
                }
                for r in results.points
            ]

        except Exception as e:
            logger.error(f"Product search error: {e}")
            return []

    # ── Stats ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get product_catalog collection stats."""
        import httpx as _httpx

        headers = {}
        if self._qdrant_api_key:
            headers["api-key"] = self._qdrant_api_key

        try:
            resp = _httpx.get(
                f"{self._qdrant_url}/collections/{self.COLLECTION_NAME}",
                headers=headers,
                timeout=5.0,
            )
            if resp.status_code == 200:
                data = resp.json().get("result", {})
                points = data.get("points_count", 0)
                vectors = data.get("indexed_vectors_count", 0)
                return {
                    "name": self.COLLECTION_NAME,
                    "points_count": points,
                    "vectors_count": vectors,
                    "total_products": points,
                    "total_indexed": vectors,
                    "status": data.get("status", "unknown"),
                }
        except Exception as e:
            logger.warning(f"Failed to get product stats: {e}")

        return {
            "name": self.COLLECTION_NAME,
            "points_count": 0,
            "vectors_count": 0,
            "total_products": 0,
            "total_indexed": 0,
            "status": "not_found",
        }

    def delete_all(self) -> bool:
        """Delete all products from the collection."""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
            self._ensure_collection()
            logger.info("Deleted all products and recreated collection")
            return True
        except Exception as e:
            logger.error(f"Error deleting products: {e}")
            return False
