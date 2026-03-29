"""
Fast Product Extractor — extracts product data from crawled web pages
using structured data (JSON-LD, meta tags) and regex patterns.

No LLM calls needed — 1000x faster than the LLM-based product classifier.
Falls back to basic text pattern matching when structured data is unavailable.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional
from collections import defaultdict

from qdrant_client.http.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

# Common airsoft brand names for regex matching
BRAND_PATTERNS = [
    "Tokyo Marui", "VFC", "G&G", "KWA", "CYMA", "Classic Army", "ICS", "Krytac",
    "Specna Arms", "Double Eagle", "LCT", "E&L", "Ares", "Amoeba", "King Arms",
    "A&K", "JG", "S&T", "ASG", "KJW", "WE", "Umarex", "Elite Force", "Novritsch",
    "Silverback", "Action Army", "Modify", "Gate", "Wolverine", "Polar Star",
    "Maple Leaf", "Prometheus", "PDI", "Lonex", "Maxx Model", "Retro Arms",
    "Nuprol", "Acetech", "Helikon", "Invader Gear", "Emerson", "TMC",
    "Warrior Assault", "Condor", "Clawgear", "Evolution", "HFC", "Saigo",
    "Rossi", "EmersonGear", "Lancer Tactical", "Valken", "Redline",
    "SHS", "ZCI", "Perun", "Jefftron", "CNC Production", "Real Sword",
    "Systema", "Mechanix", "5.11", "Oakley", "Conquer",
]

# Price pattern (EUR context)
PRICE_RE = re.compile(
    r"(?:€|EUR)\s*(\d{1,5}[.,]\d{2})|(\d{1,5}[.,]\d{2})\s*(?:€|EUR)",
    re.IGNORECASE,
)

# Category detection keywords
CATEGORY_MAP = {
    "aeg": ["aeg", "eléctrica", "electrica", "electric", "airsoft gun aeg"],
    "gbb": ["gbb", "gas blowback", "gas blow back"],
    "gbbr": ["gbbr", "gas blowback rifle"],
    "pistol": ["pistola", "pistol", "handgun", "sidearm"],
    "sniper": ["sniper", "francotirador", "bolt action", "muelle"],
    "shotgun": ["escopeta", "shotgun"],
    "smg": ["smg", "subfusil", "submachine"],
    "dmr": ["dmr", "designated marksman"],
    "lmg": ["lmg", "ametralladora", "machine gun", "support weapon"],
    "optic": ["mira", "optic", "red dot", "scope", "holographic", "visor"],
    "magazine": ["cargador", "magazine", "midcap", "hicap", "lowcap"],
    "battery": ["batería", "battery", "lipo", "nimh", "charger", "cargador batería"],
    "ammunition": ["bbs", "bb", "munición", "ammunition", "bolas"],
    "gear": ["chaleco", "vest", "plate carrier", "chest rig", "molle"],
    "protection": ["gafas", "goggles", "máscara", "mask", "protección", "casco", "helmet"],
    "accessory": ["accesorio", "accessory", "rail", "grip", "stock", "silenciador",
                   "suppressor", "tracer", "linterna", "flashlight", "laser", "bipod"],
    "clothing": ["uniforme", "uniform", "pantalón", "pants", "camisa", "shirt",
                  "botas", "boots", "guantes", "gloves", "gorra", "cap"],
    "upgrade": ["upgrade", "mejora", "inner barrel", "hop up", "hop-up",
                "motor", "gearbox", "piston", "spring", "nozzle", "cylinder"],
}


def _extract_json_ld(text: str) -> List[dict]:
    """Extract JSON-LD Product schema from HTML text."""
    products = []
    # Find JSON-LD blocks
    for match in re.finditer(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', text, re.DOTALL):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, list):
                for item in data:
                    if item.get("@type") == "Product":
                        products.append(_jsonld_to_product(item))
            elif isinstance(data, dict):
                if data.get("@type") == "Product":
                    products.append(_jsonld_to_product(data))
                elif "@graph" in data:
                    for item in data["@graph"]:
                        if item.get("@type") == "Product":
                            products.append(_jsonld_to_product(item))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return [p for p in products if p.get("name")]


def _jsonld_to_product(data: dict) -> dict:
    """Convert JSON-LD Product to our product dict."""
    offers = data.get("offers", {})
    if isinstance(offers, list):
        offers = offers[0] if offers else {}

    price = None
    currency = "EUR"
    if isinstance(offers, dict):
        price = offers.get("price")
        currency = offers.get("priceCurrency", "EUR")
        if price:
            try:
                price = float(str(price).replace(",", "."))
            except (ValueError, TypeError):
                price = None

    brand_data = data.get("brand", {})
    brand = ""
    if isinstance(brand_data, dict):
        brand = brand_data.get("name", "")
    elif isinstance(brand_data, str):
        brand = brand_data

    return {
        "name": data.get("name", ""),
        "brand": brand,
        "price": price,
        "currency": currency,
        "category": _detect_category(data.get("name", "") + " " + data.get("description", "")),
        "source_url": data.get("url", ""),
        "image_url": data.get("image", ""),
        "sku": data.get("sku", ""),
        "description": (data.get("description", "") or "")[:200],
        "confidence": 0.9,
        "extraction_method": "json-ld",
    }


def _extract_og_meta(text: str, url: str = "") -> Optional[dict]:
    """Extract product info from Open Graph meta tags."""
    og = {}
    for match in re.finditer(r'<meta\s+(?:property|name)=["\']og:(\w+)["\']\s+content=["\']([^"\']*)["\']', text, re.IGNORECASE):
        og[match.group(1)] = match.group(2)

    # Also check product: meta tags
    for match in re.finditer(r'<meta\s+(?:property|name)=["\']product:(\w+)["\']\s+content=["\']([^"\']*)["\']', text, re.IGNORECASE):
        og[f"product_{match.group(1)}"] = match.group(2)

    if og.get("type") == "product" or og.get("product_price"):
        price = None
        price_str = og.get("product_price", og.get("product_amount", ""))
        if price_str:
            try:
                price = float(price_str.replace(",", "."))
            except (ValueError, TypeError):
                pass

        return {
            "name": og.get("title", ""),
            "brand": og.get("product_brand", ""),
            "price": price,
            "currency": og.get("product_currency", "EUR"),
            "category": _detect_category(og.get("title", "") + " " + og.get("description", "")),
            "source_url": og.get("url", url),
            "image_url": og.get("image", ""),
            "description": (og.get("description", "") or "")[:200],
            "confidence": 0.7,
            "extraction_method": "open-graph",
        }
    return None


def _extract_from_text(text: str, url: str = "") -> List[dict]:
    """Extract product data from raw text using regex patterns."""
    products = []

    # Find prices
    prices = PRICE_RE.findall(text)
    if not prices:
        return []

    # Find brand mentions
    found_brands = set()
    text_lower = text.lower()
    for brand in BRAND_PATTERNS:
        if brand.lower() in text_lower:
            found_brands.add(brand)

    # If we found prices and brands, it's likely a product page
    if found_brands:
        # Extract the first price
        price_match = PRICE_RE.search(text)
        if price_match:
            price_str = price_match.group(1) or price_match.group(2)
            try:
                price = float(price_str.replace(",", "."))
            except (ValueError, TypeError):
                price = None

            # Try to extract product name from title-like text
            title_match = re.search(r'<h1[^>]*>(.*?)</h1>', text, re.IGNORECASE | re.DOTALL)
            name = ""
            if title_match:
                name = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()

            if name and price:
                products.append({
                    "name": name,
                    "brand": list(found_brands)[0] if found_brands else "",
                    "price": price,
                    "currency": "EUR",
                    "category": _detect_category(name + " " + text[:500]),
                    "source_url": url,
                    "confidence": 0.5,
                    "extraction_method": "text-pattern",
                })

    return products


def _detect_category(text: str) -> str:
    """Detect product category from text using keyword matching."""
    text_lower = text.lower()
    scores = defaultdict(int)

    for category, keywords in CATEGORY_MAP.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                scores[category] += 1

    if scores:
        return max(scores, key=scores.get)
    return "other"


def _extract_from_trafilatura_text(text: str, url: str = "", title: str = "", category_path: str = "") -> Optional[dict]:
    """Extract product from trafilatura-extracted text that contains schema.org data.

    Many e-commerce sites (PrestaShop, WooCommerce) emit structured data that
    trafilatura captures as plain text. Pattern:
      URL
      SKU
      Product Name
      Image URL
      Price (float)
      InStock/OutOfStock
      Category > Path
    """
    lines = text.strip().split("\n")
    if len(lines) < 4:
        return None

    # Detect product pages by URL pattern (common e-commerce URL patterns)
    is_product_url = bool(re.search(r'-p-\d|/producto/|/product/|\.html$', url))

    # Look for price-like float in the text
    price = None
    name = title.split(" - ")[0].strip() if title else ""
    stock_status = ""
    sku = ""
    brand = ""

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # SKU detection (pure number on its own line)
        if re.match(r'^\d{3,8}$', line) and not sku:
            sku = line
            continue

        # Price detection (float that looks like a price)
        if re.match(r'^\d{1,5}\.\d+$', line):
            try:
                p = float(line)
                if 0.1 < p < 50000:  # Reasonable price range
                    price = round(p, 2)
            except ValueError:
                pass
            continue

        # Stock status
        if line.lower() in ("instock", "in stock", "outofstock", "out of stock", "preorder"):
            stock_status = line
            continue

        # Product name — usually the longest meaningful line early in the text
        if not name and len(line) > 10 and not line.startswith("http"):
            name = line

    if not name and title:
        name = title.split(" - ")[0].strip()

    # Must have a name and either a price or be a product URL
    if not name or (not price and not is_product_url):
        return None

    # Detect brand from name + text
    text_lower = (name + " " + text[:500]).lower()
    for b in BRAND_PATTERNS:
        if b.lower() in text_lower:
            brand = b
            break

    # Detect category
    category = _detect_category(name + " " + (category_path or "") + " " + text[:300])

    return {
        "name": name,
        "brand": brand,
        "price": price,
        "currency": "EUR",
        "category": category,
        "source_url": url,
        "sku": sku,
        "in_stock": stock_status.lower() in ("instock", "in stock"),
        "confidence": 0.8 if price else 0.4,
        "extraction_method": "trafilatura-structured",
    }


class FastProductExtractor:
    """Extracts products from crawled Qdrant chunks using structured data, not LLM."""

    def __init__(self, qdrant_client, qdrant_api_key: str = None):
        self.qdrant = qdrant_client
        self.api_key = qdrant_api_key

    def extract_products(self, domain: str, collection: str = "competitor_products_v2") -> List[dict]:
        """Extract products for a domain from Qdrant chunks."""
        chunks = self._scroll_chunks(domain, collection)
        logger.info(f"FastExtract: {domain} → {len(chunks)} chunks to scan")

        products = []
        seen_names = set()

        for chunk in chunks:
            text = chunk.get("text", "") or ""
            url = chunk.get("url", "")
            title = chunk.get("title", "") or ""
            category_path = chunk.get("category_path", "") or ""

            # Primary: extract from trafilatura-extracted text (schema.org data captured as text)
            p = _extract_from_trafilatura_text(text, url, title, category_path)
            if p and p["name"] and p["name"].lower() not in seen_names:
                p["source_domain"] = domain
                products.append(p)
                seen_names.add(p["name"].lower())
                continue

            # Fallback: try HTML-based extraction if raw_html is available
            raw_html = chunk.get("raw_html", "")
            if raw_html:
                for jp in _extract_json_ld(raw_html):
                    if jp["name"] and jp["name"].lower() not in seen_names:
                        jp["source_domain"] = domain
                        jp["source_url"] = jp.get("source_url") or url
                        products.append(jp)
                        seen_names.add(jp["name"].lower())

                og = _extract_og_meta(raw_html, url)
                if og and og["name"] and og["name"].lower() not in seen_names:
                    og["source_domain"] = domain
                    products.append(og)
                    seen_names.add(og["name"].lower())

            # Extract brand mentions from category/listing pages (for brand intelligence)
            found_brands = set()
            text_lower = (text + " " + title).lower()
            for b in BRAND_PATTERNS:
                if b.lower() in text_lower:
                    found_brands.add(b)

        logger.info(f"FastExtract: {domain} → {len(products)} products found")
        return products

    def _scroll_chunks(self, domain: str, collection: str) -> List[dict]:
        """Scroll all chunks for a domain."""
        # Try both with and without www
        domains = [domain]
        if not domain.startswith("www."):
            domains.append(f"www.{domain}")
        else:
            domains.append(domain[4:])

        all_chunks = []
        for d in domains:
            offset = None
            while True:
                try:
                    results = self.qdrant.scroll(
                        collection_name=collection,
                        scroll_filter=Filter(
                            must=[FieldCondition(key="domain", match=MatchValue(value=d))]
                        ),
                        limit=100,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    points, next_offset = results
                    for point in points:
                        all_chunks.append(point.payload or {})
                    if not next_offset:
                        break
                    offset = next_offset
                except Exception as e:
                    logger.warning(f"Scroll failed for {d}: {e}")
                    break

        return all_chunks

    def get_brand_summary(self, products: List[dict]) -> Dict[str, dict]:
        """Aggregate products by brand."""
        brands = defaultdict(lambda: {"count": 0, "products": [], "avg_price": 0, "categories": set()})
        for p in products:
            brand = p.get("brand", "Unknown")
            if not brand or brand.lower() in ("unknown", "n/a", ""):
                brand = "Unknown"
            b = brands[brand]
            b["count"] += 1
            b["products"].append(p["name"])
            if p.get("price"):
                prices = [pp.get("price", 0) for pp in products if pp.get("brand") == brand and pp.get("price")]
                b["avg_price"] = round(sum(prices) / len(prices), 2) if prices else 0
            if p.get("category"):
                b["categories"].add(p["category"])

        # Convert sets to lists
        for b in brands.values():
            b["categories"] = sorted(b["categories"])
            b["products"] = b["products"][:20]  # Cap at 20 product names

        return dict(brands)


# ── Wholesale Price Detection ─────────────────────────────────

# B2B / wholesale price patterns
WHOLESALE_PRICE_RE = re.compile(
    r"(?:b2b|wholesale|dealer|trade|netto?|net\s*price|prix\s*net|precio\s*neto|hurtow)"
    r"[^€$\d]{0,30}"
    r"(?:€|EUR|USD|\$)\s*(\d{1,5}[.,]\d{2})"
    r"|"
    r"(\d{1,5}[.,]\d{2})\s*(?:€|EUR|USD|\$)"
    r"[^a-zA-Z]{0,30}"
    r"(?:b2b|wholesale|dealer|trade|netto?|net\s*price|prix\s*net|precio\s*neto|hurtow)",
    re.IGNORECASE,
)

QUANTITY_BREAK_RE = re.compile(
    r"(\d+)\+?\s*(?:pcs?|units?|szt|pieces?)?\s*[:=]\s*"
    r"(?:€|EUR|USD|\$)?\s*(\d{1,5}[.,]\d{2})\s*(?:€|EUR|USD|\$)?",
    re.IGNORECASE,
)


def extract_wholesale_price(text: str, url: str = "") -> Optional[float]:
    """Detect B2B/wholesale price from text.

    Checks for:
    1. Explicit B2B/wholesale price labels
    2. Quantity break pricing (picks lowest per-unit price)
    3. URL patterns suggesting B2B portal
    """
    # Check explicit B2B price labels
    match = WHOLESALE_PRICE_RE.search(text)
    if match:
        price_str = match.group(1) or match.group(2)
        if price_str:
            try:
                return float(price_str.replace(",", "."))
            except (ValueError, TypeError):
                pass

    # Check quantity breaks — pick the single-unit or lowest quantity price
    breaks = QUANTITY_BREAK_RE.findall(text)
    if breaks:
        # Sort by quantity, pick the lowest-quantity price as wholesale base
        parsed = []
        for qty_str, price_str in breaks:
            try:
                qty = int(qty_str)
                price = float(price_str.replace(",", "."))
                if 0.1 < price < 50000:
                    parsed.append((qty, price))
            except (ValueError, TypeError):
                continue
        if parsed:
            parsed.sort(key=lambda x: x[0])
            # Return the single-unit price (qty=1) or lowest qty
            return parsed[0][1]

    # If URL suggests B2B portal, treat the regular price as wholesale
    b2b_url_patterns = ["b2b.", "biz.", "/wholesale", "/b2b", "/dealer", "/trade", "hurtow"]
    if url and any(p in url.lower() for p in b2b_url_patterns):
        # Return None here — caller should use retail_price as wholesale_price
        # when on a B2B domain
        return None

    return None


def is_b2b_domain(url: str) -> bool:
    """Check if URL belongs to a B2B/wholesale portal."""
    if not url:
        return False
    url_lower = url.lower()
    return any(p in url_lower for p in [
        "b2b.", "biz.", "/wholesale", "/b2b", "/dealer",
        "/trade", "hurtow", "grossiste", "mayorista",
    ])
