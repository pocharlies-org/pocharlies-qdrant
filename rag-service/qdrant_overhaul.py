"""
OpenClaw RAG — Qdrant Overhaul
Creates new collections with nomic-embed-text (Ollama), structured payloads,
augmented embedding text, and payload indexes.

Collections:
  - skirmshop_products_v2  (Shopify catalog)
  - competitor_products_v2 (scraped competitor data)
  - emails_v2              (Gmail ingestion)

Usage:
  python qdrant_overhaul.py create-collections
  python qdrant_overhaul.py ingest-shopify
  python qdrant_overhaul.py ingest-competitors
  python qdrant_overhaul.py ingest-emails
  python qdrant_overhaul.py benchmark
  python qdrant_overhaul.py promote  # rename v2 → final, delete old
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "b8-CVVZ7z5ODHVfYLG60N061c6R6XAiOb5cQL6CqP6o")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768

SHOPIFY_DOMAIN = "skirmshop-spain.myshopify.com"
SHOPIFY_TOKEN = os.environ.get("SHOPIFY_TOKEN", "")

COLLECTIONS = {
    "products": "skirmshop_products_v2",
    "competitors": "competitor_products_v2",
    "emails": "emails_v2",
}

QDRANT_HEADERS = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}


# ── Embedding via Ollama ────────────────────────────────────────────

def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Embed texts using nomic-embed-text via Ollama. Batches for efficiency."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()
        embeddings = resp.json()["embeddings"]
        all_embeddings.extend(embeddings)
    return all_embeddings


def embed_single(text: str) -> List[float]:
    """Embed a single text."""
    return embed_texts([text])[0]


# ── Qdrant helpers ──────────────────────────────────────────────────

def qdrant_request(method: str, path: str, json_data=None):
    """Make authenticated Qdrant API request."""
    url = f"{QDRANT_URL}{path}"
    resp = requests.request(method, url, headers=QDRANT_HEADERS, json=json_data, timeout=30)
    resp.raise_for_status()
    return resp.json()


def generate_point_id(prefix: str, unique_key: str) -> int:
    """Deterministic point ID from prefix + key."""
    h = hashlib.sha256(f"{prefix}:{unique_key}".encode()).hexdigest()
    return int(h[:16], 16)


# ── Phase 3: Create Collections ────────────────────────────────────

def create_collections():
    """Create the 3 new collections with proper schema."""
    existing = [c["name"] for c in qdrant_request("GET", "/collections")["result"]["collections"]]

    for key, name in COLLECTIONS.items():
        if name in existing:
            logger.info(f"Collection {name} already exists, skipping")
            continue

        qdrant_request("PUT", f"/collections/{name}", {
            "vectors": {
                "dense": {"size": EMBED_DIM, "distance": "Cosine"}
            }
        })
        logger.info(f"Created collection: {name}")

    # Create payload indexes
    _create_indexes()
    logger.info("All collections and indexes created")


def _create_indexes():
    """Create payload indexes for filtered search performance."""
    product_indexes = [
        ("vendor", "keyword"),
        ("brand", "keyword"),
        ("product_type", "keyword"),
        ("in_stock", "bool"),
        ("price", "float"),
        ("sku", "keyword"),
    ]
    competitor_indexes = [
        ("source", "keyword"),
        ("brand", "keyword"),
        ("in_stock", "bool"),
        ("price", "float"),
        ("domain", "keyword"),
    ]
    email_indexes = [
        ("from_address", "keyword"),
        ("labels", "keyword"),
    ]

    for field_name, schema in product_indexes:
        _safe_create_index(COLLECTIONS["products"], field_name, schema)

    for field_name, schema in competitor_indexes:
        _safe_create_index(COLLECTIONS["competitors"], field_name, schema)

    for field_name, schema in email_indexes:
        _safe_create_index(COLLECTIONS["emails"], field_name, schema)


def _safe_create_index(collection: str, field_name: str, field_schema: str):
    """Create index, ignoring if already exists."""
    try:
        qdrant_request("PUT", f"/collections/{collection}/index", {
            "field_name": field_name,
            "field_schema": field_schema,
        })
        logger.info(f"  Index: {collection}.{field_name} ({field_schema})")
    except Exception as e:
        if "already exists" in str(e).lower() or "409" in str(e):
            pass
        else:
            logger.warning(f"  Index {field_name} on {collection}: {e}")


# ── Phase 4a: Shopify Ingestion ────────────────────────────────────

def build_product_embedding_text(product: dict) -> str:
    """Build enriched text for embedding a Shopify product."""
    variants = product.get("variants", [])
    main_variant = variants[0] if variants else {}

    inventory = sum(v.get("inventory_quantity", 0) or 0 for v in variants)
    in_stock = inventory > 0

    stock_text = f"disponible en stock ({inventory} unidades)" if in_stock else "agotado sin stock"

    # Clean HTML from body
    body_html = product.get("body_html", "")
    body_text = ""
    if body_html:
        soup = BeautifulSoup(body_html, "html.parser")
        body_text = soup.get_text(separator=" ", strip=True)

    title = product.get("title", "")
    vendor = product.get("vendor", "")
    product_type = product.get("product_type", "")
    sku = main_variant.get("sku", "")
    price = main_variant.get("price", "0")
    tags = product.get("tags", "")
    if isinstance(tags, list):
        tags = ", ".join(tags)

    return f"""Producto: {title}
Marca: {vendor}
Categoría: {product_type}
SKU: {sku}
Precio: {price}€
Stock: {stock_text}
Descripción: {body_text[:500]}
Tags: {tags}
Términos de búsqueda: {product_type.lower()}, réplica airsoft, {vendor.lower()}""".strip()


def extract_product_payload(product: dict) -> dict:
    """Extract structured payload for a Shopify product."""
    variants = product.get("variants", [])
    main_variant = variants[0] if variants else {}
    handle = product.get("handle", "")
    image = product.get("image") or {}
    inventory = sum(v.get("inventory_quantity", 0) or 0 for v in variants)

    return {
        "source": "shopify",
        "shopify_id": product.get("id"),
        "sku": main_variant.get("sku", ""),
        "title": product.get("title", ""),
        "vendor": main_variant.get("vendor", product.get("vendor", "")),
        "brand": product.get("vendor", ""),
        "product_type": product.get("product_type", ""),
        "tags": [t.strip() for t in (product.get("tags", "") if isinstance(product.get("tags", ""), str) else ",".join(product.get("tags", []))).split(",") if t.strip()],
        "price": float(main_variant.get("price", 0) or 0),
        "compare_at_price": float(main_variant.get("compare_at_price", 0) or 0) or None,
        "currency": "EUR",
        "inventory_quantity": inventory,
        "in_stock": inventory > 0,
        "url": f"https://www.skirmshop.es/products/{handle}" if handle else "",
        "handle": handle,
        "image_url": image.get("src", ""),
        "variants_count": len(variants),
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_all_shopify_products() -> List[dict]:
    """Fetch all products from Shopify Admin API (paginated)."""
    products = []
    url = f"https://{SHOPIFY_DOMAIN}/admin/api/2024-01/products.json?limit=250&status=active"
    headers = {"X-Shopify-Access-Token": SHOPIFY_TOKEN}

    while url:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("products", [])
        products.extend(batch)
        logger.info(f"  Fetched {len(products)} products so far...")

        # Pagination via Link header
        url = None
        link = resp.headers.get("Link", "")
        if 'rel="next"' in link:
            for part in link.split(","):
                if 'rel="next"' in part:
                    url = part.split("<")[1].split(">")[0]
                    break

    return products


def ingest_shopify():
    """Full Shopify product ingestion into skirmshop_products_v2."""
    collection = COLLECTIONS["products"]
    logger.info(f"Starting Shopify ingestion into {collection}")

    products = fetch_all_shopify_products()
    logger.info(f"Total products from Shopify: {len(products)}")

    batch_size = 20
    total_upserted = 0

    for i in range(0, len(products), batch_size):
        batch = products[i:i + batch_size]

        texts = []
        payloads = []
        point_ids = []

        for product in batch:
            text = build_product_embedding_text(product)
            if len(text) < 30:
                continue

            payload = extract_product_payload(product)
            payload["text_for_embedding"] = text
            pid = generate_point_id("shopify", str(product["id"]))

            texts.append(text)
            payloads.append(payload)
            point_ids.append(pid)

        if not texts:
            continue

        # Embed batch
        embeddings = embed_texts(texts)

        # Build points
        points = []
        for j, (emb, payload, pid) in enumerate(zip(embeddings, payloads, point_ids)):
            points.append({
                "id": pid,
                "vector": {"dense": emb},
                "payload": payload,
            })

        # Upsert
        qdrant_request("PUT", f"/collections/{collection}/points", {"points": points})
        total_upserted += len(points)
        logger.info(f"  Upserted batch {i // batch_size + 1}: {len(points)} points (total: {total_upserted})")

    logger.info(f"Shopify ingestion complete: {total_upserted} products indexed")


# ── Phase 4b: Competitor Ingestion ─────────────────────────────────

def build_competitor_embedding_text(item: dict) -> str:
    """Build enriched text for embedding a competitor product."""
    source = item.get("source", item.get("domain", "unknown"))
    title = item.get("title", "")
    brand = item.get("brand", "")
    price = item.get("price")
    in_stock = item.get("in_stock")
    category = item.get("category_raw", item.get("category_path", ""))
    url = item.get("url", "")

    price_text = f"{price}€" if price else "precio no disponible"
    stock_text = "disponible" if in_stock else "sin stock" if in_stock is not None else "stock desconocido"

    return f"""Competidor: {source}
Producto: {title}
Marca: {brand}
Precio competidor: {price_text}
Stock: {stock_text}
Categoría: {category}
URL: {url}""".strip()


def migrate_competitors():
    """Migrate existing competitor_products to v2 with better embedding text.

    Reads from old collection, re-embeds with nomic-embed-text, upserts to v2.
    Attempts to extract structured fields from raw text where possible.
    """
    old_collection = "competitor_products"
    new_collection = COLLECTIONS["competitors"]
    logger.info(f"Migrating {old_collection} → {new_collection}")

    # Scroll through all points in old collection
    total_migrated = 0
    next_offset = None
    batch_size = 50

    while True:
        scroll_body = {
            "limit": batch_size,
            "with_payload": True,
            "with_vector": False,
        }
        if next_offset is not None:
            scroll_body["offset"] = next_offset

        resp = qdrant_request("POST", f"/collections/{old_collection}/points/scroll", scroll_body)
        points = resp["result"]["points"]
        next_offset = resp["result"].get("next_page_offset")

        if not points:
            break

        texts = []
        payloads = []
        point_ids = []

        for point in points:
            old_payload = point.get("payload", {})

            # Extract structured data from old payload
            url = old_payload.get("url", "")
            domain = old_payload.get("domain", "")
            title = old_payload.get("title", "")
            raw_text = old_payload.get("text", "")

            # Try to extract brand/price from raw text or title
            brand = _extract_brand_from_text(title, raw_text)
            price = _extract_price_from_text(raw_text)
            in_stock = _extract_stock_from_text(raw_text)
            category = old_payload.get("category_path", "")

            # Source = domain without www
            source = domain.replace("www.", "").replace(".com", "").replace(".es", "")

            new_payload = {
                "source": source,
                "domain": domain,
                "url": url,
                "title": title,
                "brand": brand,
                "sku_raw": "",
                "price": price,
                "currency": "EUR" if price else None,
                "in_stock": in_stock,
                "category_raw": category,
                "scraped_at": old_payload.get("fetch_date", datetime.now(timezone.utc).isoformat()),
            }

            embed_text = build_competitor_embedding_text(new_payload)
            # Also include some of the original text for richer semantics
            if raw_text and len(raw_text) > 50:
                # Clean up navigation junk
                clean_text = _clean_web_text(raw_text)
                if clean_text:
                    embed_text += f"\nDetalle: {clean_text[:300]}"

            new_payload["text_for_embedding"] = embed_text

            pid = generate_point_id("competitor", url or str(point["id"]))

            texts.append(embed_text)
            payloads.append(new_payload)
            point_ids.append(pid)

        if not texts:
            if next_offset is None:
                break
            continue

        # Embed batch
        embeddings = embed_texts(texts)

        # Build points
        new_points = []
        for emb, payload, pid in zip(embeddings, payloads, point_ids):
            new_points.append({
                "id": pid,
                "vector": {"dense": emb},
                "payload": payload,
            })

        qdrant_request("PUT", f"/collections/{new_collection}/points", {"points": new_points})
        total_migrated += len(new_points)
        logger.info(f"  Migrated {total_migrated} points...")

        if next_offset is None:
            break

    logger.info(f"Competitor migration complete: {total_migrated} points")


# ── Text extraction helpers ─────────────────────────────────────────

KNOWN_BRANDS = [
    "VFC", "Tokyo Marui", "G&G", "Cyma", "Specna Arms", "ICS", "LCT", "WE",
    "KWA", "ASG", "Ares", "Krytac", "Silverback", "Novritsch", "Maple Leaf",
    "Action Army", "AGM", "Umarex", "KJ Works", "WELL", "JG", "A&K",
    "Double Bell", "E&L", "King Arms", "SRS", "Lancer Tactical", "Modify",
    "Prometheus", "PDI", "Guarder", "Nine Ball", "Lonex", "SHS", "ZCI",
    "Maxx Model", "Gate", "Jefftron", "Perun", "BTC", "Titan",
    "Nuprol", "Nimrod", "Evolution", "Amoeba", "ARES", "S&T",
    "Double Eagle", "Arcturus", "EMG", "Strike Industries",
    "Leapers", "Vector Optics", "Element", "Night Evolution",
    "5KU", "CowCow", "CTM", "TTI", "Tapp", "SPEED",
    "Acetech", "Xcortech", "FMA", "Emerson", "TMC",
    "Condor", "Invader Gear", "Helikon", "Direct Action",
    "STALKER", "CONCAMO", "Clawgear",
]

STOCK_POSITIVE = ["en stock", "disponible", "in stock", "available", "hay stock", "últimas unidades"]
STOCK_NEGATIVE = ["agotado", "sin stock", "out of stock", "no disponible", "sold out", "sin existencias"]


def _extract_brand_from_text(title: str, text: str) -> str:
    """Try to extract a known brand from title or text."""
    combined = f"{title} {text}".lower()
    for brand in KNOWN_BRANDS:
        if brand.lower() in combined:
            return brand
    return ""


def _extract_price_from_text(text: str) -> Optional[float]:
    """Try to extract a price from raw text."""
    import re
    # Match patterns like "375,00€", "€375.00", "375.00 EUR", "Precio: 375€"
    patterns = [
        r'(\d+[.,]\d{2})\s*€',
        r'€\s*(\d+[.,]\d{2})',
        r'(\d+[.,]\d{2})\s*EUR',
        r'precio[:\s]*(\d+[.,]\d{2})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            price_str = match.group(1).replace(",", ".")
            try:
                return float(price_str)
            except ValueError:
                continue
    return None


def _extract_stock_from_text(text: str) -> Optional[bool]:
    """Try to determine stock status from text."""
    text_lower = text.lower()
    for indicator in STOCK_NEGATIVE:
        if indicator in text_lower:
            return False
    for indicator in STOCK_POSITIVE:
        if indicator in text_lower:
            return True
    return None


def _clean_web_text(text: str) -> str:
    """Remove navigation junk from scraped web text."""
    import re
    # Remove lines that are just navigation links or very short
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        line = line.strip()
        if len(line) < 5:
            continue
        # Skip common nav patterns
        if any(nav in line.lower() for nav in [
            "menú", "menu", "inicio", "home", "carrito", "cart",
            "mi cuenta", "my account", "buscar", "search", "newsletter",
            "envío gratis", "registro", "login", "contraseña",
        ]):
            continue
        clean_lines.append(line)

    return " ".join(clean_lines[:10])  # Keep first 10 meaningful lines


# ── Phase 4c: Gmail Ingestion (stub) ───────────────────────────────

def ingest_emails():
    """Gmail ingestion — requires Gmail API setup. Placeholder for now."""
    logger.info("Gmail ingestion not yet implemented — requires Gmail API OAuth setup")
    logger.info("Collection emails_v2 is ready. Implement when Gmail MCP is configured.")


# ── Phase 6: Benchmark ─────────────────────────────────────────────

def run_benchmark():
    """Run 10 benchmark queries against new collections."""
    queries = [
        ("rifles eléctricos VFC disponibles menos de 400 euros", COLLECTIONS["products"], {
            "must": [{"key": "brand", "match": {"value": "VFC"}}, {"key": "in_stock", "match": {"value": True}}]
        }),
        ("pistolas GBB glock en stock", COLLECTIONS["products"], {
            "must": [{"key": "in_stock", "match": {"value": True}}]
        }),
        ("precio competidor del VFC M4 CQBR", COLLECTIONS["competitors"], {
            "must": [{"key": "brand", "match": {"value": "VFC"}}]
        }),
        ("clientes preguntando por problemas de envío", COLLECTIONS["products"], None),
        ("productos de francotirador sniper disponibles", COLLECTIONS["products"], {
            "must": [{"key": "in_stock", "match": {"value": True}}]
        }),
        ("cuánto cobra VSGun por el Tokyo Marui MWS", COLLECTIONS["competitors"], {
            "must": [{"key": "source", "match": {"value": "vsgun"}}]
        }),
        ("réplicas agotadas de HK416", COLLECTIONS["products"], {
            "must": [{"key": "in_stock", "match": {"value": False}}]
        }),
        ("emails de proveedores con facturas pendientes", COLLECTIONS["emails"], None),
        ("productos con descuento compare_at_price mayor que price", COLLECTIONS["products"], None),
        ("airsoft pistol under 100 euros in stock", COLLECTIONS["products"], {
            "must": [{"key": "in_stock", "match": {"value": True}}, {"key": "price", "range": {"lte": 100.0}}]
        }),
    ]

    results = []
    all_scores = []

    for i, (query_text, collection, query_filter) in enumerate(queries, 1):
        # Check collection exists and has points
        try:
            info = qdrant_request("GET", f"/collections/{collection}")
            if info["result"]["points_count"] == 0:
                print(f"Q{i}: \"{query_text}\" → {collection} [EMPTY COLLECTION]")
                results.append({"query": query_text, "collection": collection, "top_3": [], "note": "empty"})
                continue
        except Exception:
            print(f"Q{i}: \"{query_text}\" → {collection} [COLLECTION NOT FOUND]")
            results.append({"query": query_text, "collection": collection, "top_3": [], "note": "not found"})
            continue

        vec = embed_single(query_text)

        search_body = {
            "vector": {"name": "dense", "vector": vec},
            "limit": 3,
            "with_payload": True,
            "with_vector": False,
        }
        if query_filter:
            search_body["filter"] = query_filter

        try:
            resp = qdrant_request("POST", f"/collections/{collection}/points/search", search_body)
            hits = resp.get("result", [])
        except Exception as e:
            print(f"Q{i}: ERROR — {e}")
            results.append({"query": query_text, "collection": collection, "top_3": [], "error": str(e)})
            continue

        print(f"\nQ{i}: \"{query_text}\" → {collection}")
        entry = {"query": query_text, "collection": collection, "top_3": []}

        for j, hit in enumerate(hits):
            score = hit["score"]
            payload = hit.get("payload", {})
            title = payload.get("title", "")[:60]
            price = payload.get("price", "N/A")
            in_stock = payload.get("in_stock", "N/A")
            brand = payload.get("brand", "N/A")

            icon = "Y" if score > 0.70 else "~" if score > 0.50 else "X"
            print(f"  [{icon}] #{j+1} score={score:.4f} | {title} | brand={brand} | ${price} | stock={in_stock}")

            entry["top_3"].append({"score": round(score, 4), "title": title, "price": price, "in_stock": in_stock, "brand": brand})
            all_scores.append(score)

        if not hits:
            print("  [X] No results (filter too strict?)")

        results.append(entry)

    # Summary
    if all_scores:
        avg = sum(all_scores) / len(all_scores)
        above_70 = sum(1 for s in all_scores if s > 0.70)
        print(f"\n--- NEW BENCHMARK ---")
        print(f"Avg cosine: {avg:.4f}")
        print(f">0.70: {above_70}/{len(all_scores)} | >0.50: {sum(1 for s in all_scores if s > 0.50)}/{len(all_scores)}")

    with open("benchmark_v2.json", "w") as f:
        json.dump({"avg_score": round(avg, 4) if all_scores else 0, "results": results}, f, indent=2, ensure_ascii=False)
    print("Saved benchmark_v2.json")


# ── Promote (swap old → new) ───────────────────────────────────────

def promote():
    """After validation, this would rename v2 collections. For now just reports."""
    logger.info("Promotion: old collections preserved, new v2 collections are active.")
    logger.info("To switch OpenClaw agents, update collection names in their configs.")
    logger.info("Old collections to delete after full validation:")
    logger.info("  - product_catalog")
    logger.info("  - competitor_products")
    logger.info("  - web_pages")
    logger.info("  - product_collections")
    logger.info("  - product_pages")


# ── Query wrapper (Phase 5) ────────────────────────────────────────

def query_products(
    text: str,
    source: Optional[str] = None,
    brand: Optional[str] = None,
    max_price: Optional[float] = None,
    min_price: Optional[float] = None,
    in_stock_only: bool = False,
    limit: int = 10,
) -> List[dict]:
    """Standard query wrapper for all OpenClaw agents."""
    collection = COLLECTIONS["competitors"] if source and source != "shopify" else COLLECTIONS["products"]

    vec = embed_single(text)

    conditions = []
    if source:
        conditions.append({"key": "source", "match": {"value": source}})
    if brand:
        conditions.append({"key": "brand", "match": {"value": brand}})
    if max_price is not None:
        conditions.append({"key": "price", "range": {"lte": max_price}})
    if min_price is not None:
        conditions.append({"key": "price", "range": {"gte": min_price}})
    if in_stock_only:
        conditions.append({"key": "in_stock", "match": {"value": True}})

    search_body = {
        "vector": {"name": "dense", "vector": vec},
        "limit": limit,
        "with_payload": True,
        "with_vector": False,
    }
    if conditions:
        search_body["filter"] = {"must": conditions}

    resp = qdrant_request("POST", f"/collections/{collection}/points/search", search_body)
    return [
        {**hit.get("payload", {}), "score": hit["score"]}
        for hit in resp.get("result", [])
    ]


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qdrant_overhaul.py <command>")
        print("Commands: create-collections, ingest-shopify, ingest-competitors, ingest-emails, benchmark, promote")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "create-collections":
        create_collections()
    elif cmd == "ingest-shopify":
        ingest_shopify()
    elif cmd == "ingest-competitors":
        migrate_competitors()
    elif cmd == "ingest-emails":
        ingest_emails()
    elif cmd == "benchmark":
        run_benchmark()
    elif cmd == "promote":
        promote()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
