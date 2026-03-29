"""
Catalog Indexer - extends ProductIndexer with collections, pages, and sync orchestration.
Uses GraphQL bulk operations and content hash dedup for efficient syncing.
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from qdrant_client.http.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
    SparseVectorParams, SparseIndexParams,
    Prefetch, FusionQuery, Fusion,
)

from qdrant_utils import make_qdrant_client
import bgem3_encoder

logger = logging.getLogger(__name__)

COLLECTIONS_NAME = "product_collections"
PAGES_NAME = "product_pages"


class CatalogIndexer:
    """Indexes collections and pages into Qdrant. Works alongside ProductIndexer."""

    def __init__(self, qdrant_url, qdrant_api_key=None, model=None, embedding_model="BAAI/bge-m3"):
        self.client = make_qdrant_client(qdrant_url, qdrant_api_key)
        self._qdrant_url = qdrant_url.rstrip("/")
        self._qdrant_api_key = qdrant_api_key
        self.dim = bgem3_encoder.DENSE_DIM
        self._ensure_collections()

    def _ensure_collections(self):
        """Create product_collections and product_pages if they don't exist."""
        existing = [c.name for c in self.client.get_collections().collections]

        for coll_name in (COLLECTIONS_NAME, PAGES_NAME):
            if coll_name in existing:
                info = self.client.get_collection(coll_name)
                vectors_config = info.config.params.vectors
                if isinstance(vectors_config, dict) and "dense" in vectors_config:
                    continue
                logger.warning(f"Recreating {coll_name} with named vectors")
                self.client.delete_collection(coll_name)

            self.client.create_collection(
                collection_name=coll_name,
                vectors_config={
                    "dense": VectorParams(size=self.dim, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False),
                    ),
                },
            )
            logger.info(f"Created hybrid collection: {coll_name}")

    @staticmethod
    def _generate_id(prefix: str, shopify_id: str) -> int:
        key = f"{prefix}:{shopify_id}"
        h = hashlib.sha256(key.encode()).hexdigest()
        return int(h[:16], 16)

    # ── Index collections ─────────────────────────────────────────

    async def index_collections(self, collections_data: List[dict], shopify_client,
                                 content_hash_store=None) -> dict:
        """Index a list of flattened collections into Qdrant."""


        indexed = 0
        skipped = 0

        texts = []
        metadatas = []
        items = []

        for coll in collections_data:
            text = shopify_client.extract_collection_text(coll)
            if not text or len(text) < 10:
                continue

            # Content hash dedup
            item_key = f"collection:{coll.get('id', '')}"
            if content_hash_store:
                if not await content_hash_store.has_changed(item_key, text):
                    skipped += 1
                    continue

            metadata = shopify_client.extract_collection_metadata(coll)
            texts.append(text)
            metadatas.append(metadata)
            items.append((item_key, text))

        if not texts:
            return {"indexed": indexed, "skipped": skipped}

        # Embed via BGE-M3
        loop = asyncio.get_event_loop()
        dense_embeddings, sparse_embeddings = await loop.run_in_executor(
            None, lambda: bgem3_encoder.encode_both(texts)
        )

        points = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            shopify_id = str(metadata.get("shopify_id", ""))
            point_id = self._generate_id("collection", shopify_id)
            payload = {"text": text, "source_type": "collection", **metadata}
            points.append(PointStruct(
                id=point_id,
                vector={"dense": dense_embeddings[i].tolist(), "sparse": sparse_embeddings[i]},
                payload=payload,
            ))

        await loop.run_in_executor(
            None,
            lambda: self.client.upsert(collection_name=COLLECTIONS_NAME, points=points)
        )

        # Update content hashes
        if content_hash_store:
            for item_key, text in items:
                await content_hash_store.set_hash(item_key, text)

        indexed = len(points)
        logger.info(f"Indexed {indexed} collections, skipped {skipped}")
        return {"indexed": indexed, "skipped": skipped}

    # ── Index pages ───────────────────────────────────────────────

    @staticmethod
    def _chunk_page_html(title: str, body_html: str, max_chunk_chars: int = 1500) -> List[Dict]:
        """Split a page into chunks by HTML headings (h1-h4).

        Each chunk contains:
          - section_heading: the heading text (or "Introduction" for content before first heading)
          - text: the section content
          - chunk_index: position in the page

        If a section is longer than max_chunk_chars, it's split into paragraphs.
        """
        from bs4 import BeautifulSoup
        import re

        if not body_html:
            return [{"section_heading": title, "text": title, "chunk_index": 0}] if title else []

        soup = BeautifulSoup(body_html, "html.parser")

        # Split by headings (h1, h2, h3, h4)
        sections: List[Dict] = []
        current_heading = "Introduction"
        current_parts: List[str] = []

        for element in soup.children:
            tag_name = getattr(element, "name", None)
            if tag_name in ("h1", "h2", "h3", "h4"):
                # Save previous section
                if current_parts:
                    text = "\n".join(current_parts).strip()
                    if text:
                        sections.append({"heading": current_heading, "text": text})
                current_heading = element.get_text(strip=True)
                current_parts = []
            else:
                text = element.get_text(separator="\n", strip=True) if hasattr(element, "get_text") else str(element).strip()
                if text:
                    current_parts.append(text)

        # Don't forget last section
        if current_parts:
            text = "\n".join(current_parts).strip()
            if text:
                sections.append({"heading": current_heading, "text": text})

        # If no sections found, treat entire content as one chunk
        if not sections:
            full_text = soup.get_text(separator="\n", strip=True)
            if full_text:
                sections = [{"heading": title, "text": full_text}]

        # Build chunks with page title prefix for context
        chunks = []
        for section in sections:
            heading = section["heading"]
            text = section["text"]
            # Prefix with page title + section heading for better retrieval
            chunk_text = f"{title} — {heading}\n{text}"

            if len(chunk_text) <= max_chunk_chars:
                chunks.append({
                    "section_heading": heading,
                    "text": chunk_text,
                    "chunk_index": len(chunks),
                })
            else:
                # Split long sections by double newline (paragraphs)
                paragraphs = re.split(r"\n{2,}", text)
                buffer = ""
                for para in paragraphs:
                    candidate = f"{buffer}\n{para}".strip() if buffer else para
                    if len(f"{title} — {heading}\n{candidate}") > max_chunk_chars and buffer:
                        chunks.append({
                            "section_heading": heading,
                            "text": f"{title} — {heading}\n{buffer}",
                            "chunk_index": len(chunks),
                        })
                        buffer = para
                    else:
                        buffer = candidate
                if buffer:
                    chunks.append({
                        "section_heading": heading,
                        "text": f"{title} — {heading}\n{buffer}",
                        "chunk_index": len(chunks),
                    })

        return chunks

    async def index_pages(self, pages_data: List[dict], shopify_client,
                           content_hash_store=None) -> dict:
        """Index pages as chunked sections into Qdrant for fine-grained retrieval."""


        indexed = 0
        skipped = 0

        texts = []
        metadatas = []
        items = []

        for page in pages_data:
            page_title = page.get("title", "")
            body_html = page.get("body_html", "") or page.get("body_summary", "")
            if not body_html and not page_title:
                continue

            full_text = shopify_client.extract_page_text(page)
            item_key = f"page:{page.get('id', '')}"
            if content_hash_store:
                if not await content_hash_store.has_changed(item_key, full_text or ""):
                    skipped += 1
                    continue

            base_metadata = shopify_client.extract_page_metadata(page)
            chunks = self._chunk_page_html(page_title, body_html)

            for chunk in chunks:
                texts.append(chunk["text"])
                metadatas.append({
                    **base_metadata,
                    "section_heading": chunk["section_heading"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": len(chunks),
                })
                items.append((item_key, full_text or ""))

        if not texts:
            return {"indexed": indexed, "skipped": skipped}

        loop = asyncio.get_event_loop()
        dense_embeddings, sparse_embeddings = await loop.run_in_executor(
            None, lambda: bgem3_encoder.encode_both(texts)
        )

        points = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            shopify_id = str(metadata.get("shopify_id", ""))
            chunk_idx = metadata.get("chunk_index", 0)
            point_id = self._generate_id("page_chunk", f"{shopify_id}:{chunk_idx}")
            payload = {"text": text, "source_type": "page", **metadata}
            points.append(PointStruct(
                id=point_id,
                vector={"dense": dense_embeddings[i].tolist(), "sparse": sparse_embeddings[i]},
                payload=payload,
            ))

        await loop.run_in_executor(
            None,
            lambda: self.client.upsert(collection_name=PAGES_NAME, points=points)
        )

        if content_hash_store:
            seen_keys = set()
            for item_key, text in items:
                if item_key not in seen_keys:
                    await content_hash_store.set_hash(item_key, text)
                    seen_keys.add(item_key)

        indexed = len(points)
        logger.info(f"Indexed {indexed} page chunks from {len(pages_data)} pages, skipped {skipped}")
        return {"indexed": indexed, "skipped": skipped}

    # ── Search ────────────────────────────────────────────────────

    def search_collections(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search on product_collections."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if COLLECTIONS_NAME not in collections:
                return []

            dense = bgem3_encoder.encode_dense_query(query)
            sparse = bgem3_encoder.encode_sparse_query(query)

            results = self.client.query_points(
                collection_name=COLLECTIONS_NAME,
                prefetch=[
                    Prefetch(query=dense, using="dense", limit=top_k * 3),
                    Prefetch(query=sparse, using="sparse", limit=top_k * 3),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )

            return [
                {
                    "title": r.payload.get("title", ""),
                    "handle": r.payload.get("handle", ""),
                    "url": r.payload.get("url", ""),
                    "image_url": r.payload.get("image_url", ""),
                    "products_count": r.payload.get("products_count", 0),
                    "score": round(r.score, 4),
                    "source_type": "collection",
                    "shopify_id": r.payload.get("shopify_id", ""),
                }
                for r in results.points
            ]
        except Exception as e:
            logger.error(f"Collection search error: {e}")
            return []

    def search_pages(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search on product_pages."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if PAGES_NAME not in collections:
                return []

            dense = bgem3_encoder.encode_dense_query(query)
            sparse = bgem3_encoder.encode_sparse_query(query)

            results = self.client.query_points(
                collection_name=PAGES_NAME,
                prefetch=[
                    Prefetch(query=dense, using="dense", limit=top_k * 3),
                    Prefetch(query=sparse, using="sparse", limit=top_k * 3),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )

            return [
                {
                    "title": r.payload.get("title", ""),
                    "handle": r.payload.get("handle", ""),
                    "url": r.payload.get("url", ""),
                    "text": r.payload.get("text", ""),
                    "section_heading": r.payload.get("section_heading", ""),
                    "chunk_index": r.payload.get("chunk_index", 0),
                    "total_chunks": r.payload.get("total_chunks", 1),
                    "score": round(r.score, 4),
                    "source_type": "page",
                    "shopify_id": r.payload.get("shopify_id", ""),
                }
                for r in results.points
            ]
        except Exception as e:
            logger.error(f"Page search error: {e}")
            return []

    def get_product_by_id_or_handle(self, product_indexer, id_or_handle: str) -> Optional[Dict]:
        """Look up a single product by GID or handle from the product_catalog collection."""
        try:
            # Try handle first (more common in user queries)
            results, _ = product_indexer.client.scroll(
                collection_name=product_indexer.COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key="handle", match=MatchValue(value=id_or_handle))
                ]),
                limit=1,
                with_payload=True,
            )
            if not results:
                # Try shopify_id
                results, _ = product_indexer.client.scroll(
                    collection_name=product_indexer.COLLECTION_NAME,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="shopify_id", match=MatchValue(value=id_or_handle))
                    ]),
                    limit=1,
                    with_payload=True,
                )
            if results:
                p = results[0].payload
                return {**p, "source_type": "product"}
            return None
        except Exception as e:
            logger.error(f"Product lookup error: {e}")
            return None

    def get_collection_products(self, product_indexer, id_or_handle: str, limit: int = 20) -> List[Dict]:
        """Get products belonging to a collection (by filtering on collection_ids payload)."""
        try:
            # First find the collection
            results, _ = self.client.scroll(
                collection_name=COLLECTIONS_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key="handle", match=MatchValue(value=id_or_handle))
                ]),
                limit=1,
                with_payload=True,
            )
            if not results:
                results, _ = self.client.scroll(
                    collection_name=COLLECTIONS_NAME,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="shopify_id", match=MatchValue(value=id_or_handle))
                    ]),
                    limit=1,
                    with_payload=True,
                )

            if not results:
                return []

            collection_id = results[0].payload.get("shopify_id", "")
            if not collection_id:
                return []

            # Search products that have this collection_id in their collection_ids array
            prod_results, _ = product_indexer.client.scroll(
                collection_name=product_indexer.COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key="collection_ids", match=MatchValue(value=collection_id))
                ]),
                limit=limit,
                with_payload=True,
            )

            return [
                {**r.payload, "source_type": "product"}
                for r in prod_results
            ]
        except Exception as e:
            logger.error(f"Collection products lookup error: {e}")
            return []

    def get_inventory(self, product_indexer, id_or_handle_or_sku: str) -> Optional[Dict]:
        """Get inventory info for a product by GID, handle, or SKU."""
        product = self.get_product_by_id_or_handle(product_indexer, id_or_handle_or_sku)
        if not product:
            # Try SKU search
            try:
                results, _ = product_indexer.client.scroll(
                    collection_name=product_indexer.COLLECTION_NAME,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="sku", match=MatchValue(value=id_or_handle_or_sku))
                    ]),
                    limit=1,
                    with_payload=True,
                )
                if results:
                    product = {**results[0].payload, "source_type": "product"}
            except Exception:
                pass
        if not product:
            return None

        return {
            "title": product.get("title", ""),
            "shopify_id": product.get("shopify_id", ""),
            "handle": product.get("handle", ""),
            "sku": product.get("sku", ""),
            "inventory_quantity": product.get("inventory_quantity", 0),
            "variants_count": product.get("variants_count", 0),
            "status": product.get("status", ""),
        }

    def get_stats(self) -> Dict:
        """Get stats for collections and pages."""
        import httpx as _httpx

        headers = {}
        if self._qdrant_api_key:
            headers["api-key"] = self._qdrant_api_key

        stats = {}
        for coll_name in (COLLECTIONS_NAME, PAGES_NAME):
            try:
                resp = _httpx.get(
                    f"{self._qdrant_url}/collections/{coll_name}",
                    headers=headers, timeout=5.0,
                )
                if resp.status_code == 200:
                    data = resp.json().get("result", {})
                    stats[coll_name] = {
                        "points_count": data.get("points_count", 0),
                        "status": data.get("status", "unknown"),
                    }
                else:
                    stats[coll_name] = {"points_count": 0, "status": "not_found"}
            except Exception:
                stats[coll_name] = {"points_count": 0, "status": "error"}

        return stats
