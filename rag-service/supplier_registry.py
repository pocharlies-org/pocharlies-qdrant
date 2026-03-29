"""
Supplier Registry for Pocharlies RAG
Manages supplier metadata in Qdrant with CRUD operations and search.
"""

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
import bgem3_encoder

from qdrant_utils import make_qdrant_client

logger = logging.getLogger(__name__)

COLLECTION_NAME = "supplier_registry"

# Known suppliers to seed on first run
SEED_SUPPLIERS = [
    {
        "name": "Silverback Airsoft",
        "slug": "silverback",
        "type": "supplier",
        "status": "active",
        "country": "Hong Kong",
        "website_retail": "https://www.silverback-airsoft.com",
        "website_b2b": "",
        "contact_email": "kto@silverback-airsoft.com",
        "payment_terms": "Prepayment",
        "shipping_terms": "10 business days after payment",
        "minimum_order": 0,
        "currency": "USD",
        "brands_carried": ["Silverback"],
        "notes": "Direct manufacturer. SRS, HTI, MDRX specialist. Contact: Kasel To.",
    },
    {
        "name": "3P-Store",
        "slug": "3p-store",
        "type": "supplier",
        "status": "active",
        "country": "China",
        "website_retail": "",
        "website_b2b": "",
        "contact_email": "service@3p-store.com",
        "payment_terms": "",
        "shipping_terms": "UPS",
        "minimum_order": 0,
        "currency": "USD",
        "brands_carried": [],
        "notes": "Chinese wholesaler. Contact: Chris.",
    },
    {
        "name": "AGM",
        "slug": "agm",
        "type": "supplier",
        "status": "active",
        "country": "Spain",
        "website_retail": "",
        "website_b2b": "",
        "contact_email": "",
        "payment_terms": "",
        "shipping_terms": "",
        "minimum_order": 0,
        "currency": "EUR",
        "brands_carried": [],
        "notes": "Spanish distributor. Thermal scopes, accessories.",
    },
    {
        "name": "Taiwangun",
        "slug": "taiwangun",
        "type": "supplier",
        "status": "potential",
        "country": "Poland",
        "website_retail": "https://www.taiwangun.com",
        "website_b2b": "https://taiwangun.biz",
        "contact_email": "krakman@taiwangun.com",
        "payment_terms": "",
        "shipping_terms": "",
        "minimum_order": 0,
        "currency": "EUR",
        "brands_carried": [
            "CYMA", "Double Eagle", "Specna Arms", "S&T", "Modify",
            "Maple Leaf", "Gate", "Nuprol", "ASG",
        ],
        "notes": "Large Polish wholesaler with B2B portal at taiwangun.biz. "
                 "Sends regular promotional emails. Never purchased from them yet.",
    },
    {
        "name": "Gunfire",
        "slug": "gunfire",
        "type": "supplier",
        "status": "potential",
        "country": "Poland",
        "website_retail": "https://www.gunfire.com",
        "website_b2b": "https://b2b.gunfire.com",
        "contact_email": "b2b@gfcorp.pl",
        "payment_terms": "",
        "shipping_terms": "",
        "minimum_order": 0,
        "currency": "EUR",
        "brands_carried": [
            "Specna Arms", "Evolution", "G&G", "Modify",
        ],
        "notes": "Polish wholesaler. Owns Specna Arms brand. B2B wholesale offers via email.",
    },
    {
        "name": "AirsoftZone",
        "slug": "airsoftzone",
        "type": "competitor_supplier",
        "status": "potential",
        "country": "Slovenia",
        "website_retail": "https://www.airsoftzone.com",
        "website_b2b": "",
        "contact_email": "",
        "payment_terms": "",
        "shipping_terms": "",
        "minimum_order": 0,
        "currency": "EUR",
        "brands_carried": [],
        "notes": "Slovenian retailer that also operates as B2B supplier. "
                 "Overlapping catalog with Taiwangun and Anareus.",
    },
    {
        "name": "Anareus",
        "slug": "anareus",
        "type": "competitor_supplier",
        "status": "potential",
        "country": "Czech Republic",
        "website_retail": "https://www.anareus.com",
        "website_b2b": "",
        "contact_email": "",
        "payment_terms": "",
        "shipping_terms": "",
        "minimum_order": 0,
        "currency": "EUR",
        "brands_carried": [],
        "notes": "Czech retailer/wholesaler. Overlapping catalog with AirsoftZone and VSGUN.",
    },
    {
        "name": "VSGUN",
        "slug": "vsgun",
        "type": "competitor_supplier",
        "status": "potential",
        "country": "Spain",
        "website_retail": "https://www.vsgun.es",
        "website_b2b": "",
        "contact_email": "",
        "payment_terms": "",
        "shipping_terms": "",
        "minimum_order": 0,
        "currency": "EUR",
        "brands_carried": [],
        "notes": "Spanish competitor that could also serve as supplier. "
                 "Shares many products with AirsoftZone and Anareus.",
    },
]


def _slug_to_point_id(slug: str) -> int:
    """Deterministic point ID from slug."""
    h = hashlib.sha256(f"supplier:{slug}".encode()).hexdigest()
    return int(h[:16], 16)


def _build_text(supplier: dict) -> str:
    """Build searchable text from supplier metadata."""
    parts = [
        supplier.get("name", ""),
        supplier.get("country", ""),
        supplier.get("type", ""),
        supplier.get("status", ""),
    ]
    brands = supplier.get("brands_carried", [])
    if brands:
        parts.append("Brands: " + ", ".join(brands))
    notes = supplier.get("notes", "")
    if notes:
        parts.append(notes)
    return " | ".join(p for p in parts if p)


class SupplierRegistry:
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        model=None,
        embedding_model: str = "BAAI/bge-m3",
    ):
        self.client = make_qdrant_client(qdrant_url, qdrant_api_key)
        self._qdrant_url = qdrant_url.rstrip("/")
        self._qdrant_api_key = qdrant_api_key

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
            ("slug", "keyword"),
            ("status", "keyword"),
            ("type", "keyword"),
            ("country", "keyword"),
        ]:
            try:
                self.client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception:
                pass

    def seed_if_empty(self):
        """Seed with known suppliers if collection is empty."""
        info = self.client.get_collection(COLLECTION_NAME)
        if info.points_count > 0:
            logger.info(f"Supplier registry has {info.points_count} entries, skipping seed")
            return

        logger.info(f"Seeding supplier registry with {len(SEED_SUPPLIERS)} suppliers...")
        for supplier in SEED_SUPPLIERS:
            self.upsert_supplier(supplier)
        logger.info("Supplier registry seeded")

    def upsert_supplier(self, supplier: dict) -> dict:
        """Add or update a supplier."""
        slug = supplier["slug"]
        now = datetime.now(timezone.utc).isoformat()

        payload = {
            "name": supplier.get("name", ""),
            "slug": slug,
            "type": supplier.get("type", "supplier"),
            "status": supplier.get("status", "potential"),
            "country": supplier.get("country", ""),
            "website_retail": supplier.get("website_retail", ""),
            "website_b2b": supplier.get("website_b2b", ""),
            "contact_email": supplier.get("contact_email", ""),
            "payment_terms": supplier.get("payment_terms", ""),
            "shipping_terms": supplier.get("shipping_terms", ""),
            "minimum_order": supplier.get("minimum_order", 0),
            "currency": supplier.get("currency", "EUR"),
            "brands_carried": supplier.get("brands_carried", []),
            "notes": supplier.get("notes", ""),
            "last_crawled": supplier.get("last_crawled", ""),
            "last_pricelist": supplier.get("last_pricelist", ""),
            "products_indexed": supplier.get("products_indexed", 0),
            "created_at": supplier.get("created_at", now),
            "updated_at": now,
        }

        text = _build_text(payload)
        payload["text"] = text

        dense_emb = bgem3_encoder.encode_dense_query(text)
        sparse_emb = bgem3_encoder.encode_sparse([text])[0]

        point_id = _slug_to_point_id(slug)
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=point_id,
                vector={"dense": dense_emb, "sparse": sparse_emb},
                payload=payload,
            )],
        )
        logger.info(f"Upserted supplier: {slug}")
        return payload

    def get_supplier(self, slug: str) -> Optional[dict]:
        """Get a single supplier by slug."""
        results = self.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key="slug", match=MatchValue(value=slug))
            ]),
            limit=1,
            with_payload=True,
        )
        points = results[0]
        if not points:
            return None
        return points[0].payload

    def list_suppliers(self, status: Optional[str] = None, type_filter: Optional[str] = None) -> List[dict]:
        """List all suppliers, optionally filtered by status or type."""
        conditions = []
        if status:
            conditions.append(FieldCondition(key="status", match=MatchValue(value=status)))
        if type_filter:
            conditions.append(FieldCondition(key="type", match=MatchValue(value=type_filter)))

        scroll_filter = Filter(must=conditions) if conditions else None
        results = self.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=100,
            with_payload=True,
        )
        return [p.payload for p in results[0]]

    def delete_supplier(self, slug: str) -> bool:
        """Delete a supplier by slug."""
        point_id = _slug_to_point_id(slug)
        try:
            self.client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=[point_id],
            )
            logger.info(f"Deleted supplier: {slug}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete supplier {slug}: {e}")
            return False

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Hybrid search across suppliers."""
        try:
            dense_emb = bgem3_encoder.encode_dense_query(query)
            sparse_emb = bgem3_encoder.encode_sparse_query(query)

            results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    Prefetch(query=dense_emb, using="dense", limit=top_k * 3),
                    Prefetch(query=sparse_emb, using="sparse", limit=top_k * 3),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
            return [
                {**r.payload, "score": round(r.score, 4)}
                for r in results.points
            ]
        except Exception as e:
            logger.error(f"Supplier search error: {e}")
            return []

    def update_products_indexed(self, slug: str, count: int):
        """Update the products_indexed counter for a supplier."""
        point_id = _slug_to_point_id(slug)
        try:
            self.client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={
                    "products_indexed": count,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                points=[point_id],
            )
        except Exception as e:
            logger.error(f"Failed to update products count for {slug}: {e}")

    def update_last_crawled(self, slug: str):
        """Mark supplier as recently crawled."""
        point_id = _slug_to_point_id(slug)
        try:
            self.client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={
                    "last_crawled": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                points=[point_id],
            )
        except Exception as e:
            logger.error(f"Failed to update last_crawled for {slug}: {e}")

    def update_last_pricelist(self, slug: str):
        """Mark supplier as having a recent pricelist upload."""
        point_id = _slug_to_point_id(slug)
        try:
            self.client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={
                    "last_pricelist": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                points=[point_id],
            )
        except Exception as e:
            logger.error(f"Failed to update last_pricelist for {slug}: {e}")

    def get_email_domains(self) -> Dict[str, str]:
        """Return mapping of email domains to supplier slugs for email monitoring."""
        suppliers = self.list_suppliers()
        domain_map = {}
        for s in suppliers:
            email = s.get("contact_email", "")
            if email and "@" in email:
                domain = email.split("@")[1].lower()
                domain_map[domain] = s["slug"]
        return domain_map

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
