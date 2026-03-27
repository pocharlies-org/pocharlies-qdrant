"""
Product Classifier for RAG Pipeline
Uses LLM to extract structured product data from crawled web content,
resolve entities against the product catalog, and generate price comparisons.
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional, Callable

from qdrant_client.http.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)


@dataclass
class ProductSpec:
    """Structured product data extracted by LLM."""
    name: str
    category: str  # gbb, aeg, sniper, pistol, shotgun, smg, accessory, gear, etc.
    brand: str = ""
    model_number: str = ""
    caliber: Optional[str] = None
    fps: Optional[int] = None
    material: Optional[str] = None
    weight_grams: Optional[float] = None
    price: Optional[float] = None
    currency: str = "EUR"
    compatibility: List[str] = field(default_factory=list)
    source_url: str = ""
    source_domain: str = ""
    raw_description: str = ""
    confidence: float = 0.0
    extracted_at: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "brand": self.brand,
            "model_number": self.model_number,
            "caliber": self.caliber,
            "fps": self.fps,
            "material": self.material,
            "weight_grams": self.weight_grams,
            "price": self.price,
            "currency": self.currency,
            "compatibility": self.compatibility,
            "source_url": self.source_url,
            "source_domain": self.source_domain,
            "confidence": self.confidence,
            "extracted_at": self.extracted_at,
        }


@dataclass
class PriceMatch:
    """Entity resolution match between competitor and catalog product."""
    competitor_product: dict
    catalog_product_id: Optional[int] = None
    catalog_product_name: str = ""
    similarity_score: float = 0.0
    price_difference: Optional[float] = None
    price_ratio: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "competitor": self.competitor_product,
            "catalog_match": self.catalog_product_name,
            "catalog_id": self.catalog_product_id,
            "similarity": round(self.similarity_score, 4),
            "price_difference": self.price_difference,
            "price_ratio": round(self.price_ratio, 4) if self.price_ratio else None,
        }


@dataclass
class ClassificationJob:
    """Tracks a product classification job."""
    job_id: str
    domain: str
    status: str = "running"
    chunks_processed: int = 0
    chunks_total: int = 0
    products_found: int = 0
    errors: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    results: List[dict] = field(default_factory=list)
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
            "domain": self.domain,
            "status": self.status,
            "chunks_processed": self.chunks_processed,
            "chunks_total": self.chunks_total,
            "products_found": self.products_found,
            "errors": self.errors[-10:],
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


# ── Extraction prompt ───────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are a product data extraction assistant for an airsoft/tactical equipment store.
Extract structured product information from the provided text passages.
Each passage may contain one or more products, or no products at all (skip those).

Return a JSON array of products. For each product found:
{
  "name": "product name",
  "category": "gbb|aeg|sniper|pistol|shotgun|smg|launcher|magazine|battery|optic|ammunition|gear|protection|accessory",
  "brand": "brand name",
  "model_number": "model number if found",
  "caliber": "6mm or 8mm or null",
  "fps": integer or null,
  "material": "metal|polymer|ABS|wood|nylon|fiber|null",
  "weight_grams": float or null,
  "price": float or null,
  "currency": "EUR|USD|GBP",
  "compatibility": ["list of compatible items/systems"],
  "confidence": 0.0-1.0
}

Rules:
- Only extract actual products, not categories or navigation items
- If a field is not found, use null
- confidence should reflect how complete the extraction is (1.0 = all fields found)
- Return [] if no products are found in the text
- Return ONLY the JSON array, no explanation"""


class ProductClassifier:
    def __init__(self, qdrant_client, model, llm_client):
        self.qdrant_client = qdrant_client
        self.model = model
        self.llm_client = llm_client

    async def extract_products_from_collection(
        self,
        domain: str,
        collection_name: str = "web_pages",
        batch_size: int = 5,
        progress_callback: Optional[Callable] = None,
    ) -> ClassificationJob:
        """Scroll through collection chunks for a domain, extract products via LLM."""
        job = ClassificationJob(
            job_id=uuid.uuid4().hex[:12],
            domain=domain,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            # Count chunks for this domain
            job.log(f"Scanning {collection_name} for domain: {domain}")
            chunks = self._scroll_domain_chunks(domain, collection_name)
            job.chunks_total = len(chunks)
            job.log(f"Found {len(chunks)} chunks to process")

            if not chunks:
                job.status = "completed"
                job.ended_at = datetime.now(timezone.utc).isoformat()
                return job

            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                products = await self._classify_batch(batch)

                for p in products:
                    p["source_domain"] = domain
                    job.results.append(p)

                job.chunks_processed = min(i + batch_size, len(chunks))
                job.products_found = len(job.results)
                job.log(f"Processed {job.chunks_processed}/{job.chunks_total} chunks, {job.products_found} products found")

                if progress_callback:
                    progress_callback(job)

            job.status = "completed"
            job.log(f"COMPLETED: {job.products_found} products extracted from {job.chunks_total} chunks")

        except Exception as e:
            job.status = "failed"
            job.errors.append(str(e)[:200])
            job.log(f"FAILED: {str(e)[:200]}")
            logger.error(f"Classification failed for {domain}: {e}")

        job.ended_at = datetime.now(timezone.utc).isoformat()
        if progress_callback:
            progress_callback(job)
        return job

    def _scroll_domain_chunks(self, domain: str, collection_name: str) -> List[dict]:
        """Scroll all chunks for a domain from a collection."""
        chunks = []
        offset = None

        while True:
            results = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
                ),
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = results

            for point in points:
                payload = point.payload or {}
                chunks.append({
                    "text": payload.get("text", ""),
                    "url": payload.get("url", ""),
                    "title": payload.get("title", ""),
                })

            if next_offset is None:
                break
            offset = next_offset

        return chunks

    async def _classify_batch(self, chunks: List[dict]) -> List[dict]:
        """Send batch of text chunks to LLM for product extraction."""
        combined = "\n\n---\n\n".join(
            f"Source: {c['url']}\nTitle: {c['title']}\n{c['text']}"
            for c in chunks
        )

        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.llm_client.chat.completions.create(
                    model=os.environ.get("CLASSIFIER_MODEL", "local"),
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Extract products from these text passages:\n\n{combined}"},
                    ],
                    max_tokens=4096,
                    temperature=0.1,
                    timeout=300,
                    user="rag:competitor-extract",
                ),
            )

            response_text = response.choices[0].message.content
            products = self._parse_products(response_text)

            # Attach source info
            now = datetime.now(timezone.utc).isoformat()
            for p in products:
                p["extracted_at"] = now
                # Try to match source URL from chunks
                if not p.get("source_url") and chunks:
                    p["source_url"] = chunks[0].get("url", "")

            return products

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return []

    @staticmethod
    def _parse_products(response_text: str) -> List[dict]:
        """Parse LLM response into product dicts."""
        text = response_text.strip()

        # Strip markdown fences
        if "```" in text:
            lines = text.split("\n")
            inside = False
            json_lines = []
            for line in lines:
                if line.strip().startswith("```"):
                    inside = not inside
                    continue
                if inside:
                    json_lines.append(line)
            if json_lines:
                text = "\n".join(json_lines)

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError:
            # Try to find JSON array in the text
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                    if isinstance(data, list):
                        return data
                except json.JSONDecodeError:
                    pass
            return []

    # ── Entity Resolution ─────────────────────────────────────────

    async def resolve_entities(
        self,
        extracted_products: List[dict],
        product_indexer,
        top_k: int = 3,
    ) -> List[PriceMatch]:
        """Match extracted competitor products to catalog using embedding similarity."""
        matches = []

        for product in extracted_products:
            name = product.get("name", "")
            brand = product.get("brand", "")
            query = f"{brand} {name}".strip()

            if not query:
                continue

            results = product_indexer.search(query=query, top_k=top_k)

            if results and results[0]["score"] > 0.3:
                best = results[0]
                comp_price = product.get("price")
                cat_price = best.get("price")

                price_diff = None
                price_ratio = None
                if comp_price and cat_price and cat_price > 0:
                    price_diff = round(comp_price - cat_price, 2)
                    price_ratio = comp_price / cat_price

                matches.append(PriceMatch(
                    competitor_product=product,
                    catalog_product_id=best.get("shopify_id"),
                    catalog_product_name=best.get("title", ""),
                    similarity_score=best["score"],
                    price_difference=price_diff,
                    price_ratio=price_ratio,
                ))

        return matches

    @staticmethod
    def generate_price_report(matches: List[PriceMatch]) -> dict:
        """Generate a structured price comparison report."""
        cheaper = []
        pricier = []
        matched = []

        for m in matches:
            entry = m.to_dict()
            matched.append(entry)
            if m.price_difference is not None:
                if m.price_difference < -1:  # competitor is cheaper
                    cheaper.append(entry)
                elif m.price_difference > 1:  # competitor is more expensive
                    pricier.append(entry)

        return {
            "total_matches": len(matches),
            "competitor_cheaper": len(cheaper),
            "competitor_pricier": len(pricier),
            "details": matched,
            "cheaper_items": sorted(cheaper, key=lambda x: x["price_difference"] or 0),
            "pricier_items": sorted(pricier, key=lambda x: -(x["price_difference"] or 0)),
        }
