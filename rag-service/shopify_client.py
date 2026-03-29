"""
Shopify Admin API Client for product catalog indexing.
Pulls products from skirmshop.es Shopify store and extracts structured metadata.
"""

import asyncio
import logging
import re
from typing import List, Dict, Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ShopifyClient:
    """Client for Shopify Admin REST API (2024-01 version)."""

    API_VERSION = "2024-10"

    def __init__(self, shop_domain: str, access_token: str):
        self.shop_domain = shop_domain
        self.base_url = f"https://{shop_domain}/admin/api/{self.API_VERSION}"
        self.headers = {
            "X-Shopify-Access-Token": access_token,
            "Content-Type": "application/json",
        }
        self._graphql = None

    @property
    def graphql(self):
        """Lazy-init GraphQL client for bulk operations and single-item fetches."""
        if self._graphql is None:
            from shopify_graphql import ShopifyGraphQL
            self._graphql = ShopifyGraphQL(self.shop_domain, self.headers["X-Shopify-Access-Token"])
        return self._graphql

    async def get_product_count(self) -> int:
        """Get total product count."""
        async with httpx.AsyncClient(headers=self.headers, timeout=15.0) as client:
            resp = await client.get(f"{self.base_url}/products/count.json")
            resp.raise_for_status()
            return resp.json()["count"]

    async def list_products(
        self,
        limit: int = 250,
        since_id: Optional[int] = None,
    ) -> List[dict]:
        """Paginate through all products via the Admin API.

        Shopify rate limit: ~2 req/sec (leaky bucket). We sleep 0.5s between pages.
        """
        all_products = []
        params = {"limit": min(limit, 250)}
        if since_id:
            params["since_id"] = since_id

        async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
            url = f"{self.base_url}/products.json"

            while url:
                resp = await client.get(url, params=params)
                resp.raise_for_status()

                data = resp.json()
                products = data.get("products", [])
                all_products.extend(products)

                if not products:
                    break

                # Pagination via Link header
                url = None
                params = None  # subsequent pages use URL params from Link header
                link_header = resp.headers.get("link", "")
                if 'rel="next"' in link_header:
                    for part in link_header.split(","):
                        if 'rel="next"' in part:
                            url = part.split("<")[1].split(">")[0]
                            break

                await asyncio.sleep(0.5)  # rate limit

        logger.info(f"Fetched {len(all_products)} products from Shopify")
        return all_products

    async def get_product(self, product_id: int) -> dict:
        """Fetch a single product with variants and metafields."""
        async with httpx.AsyncClient(headers=self.headers, timeout=15.0) as client:
            resp = await client.get(f"{self.base_url}/products/{product_id}.json")
            resp.raise_for_status()
            return resp.json()["product"]

    async def get_updated_since(self, since: str) -> List[dict]:
        """Fetch products updated after a given ISO timestamp."""
        all_products = []
        params = {"limit": 250, "updated_at_min": since}

        async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
            url = f"{self.base_url}/products.json"

            while url:
                resp = await client.get(url, params=params)
                resp.raise_for_status()

                products = resp.json().get("products", [])
                all_products.extend(products)

                if not products:
                    break

                url = None
                params = None  # subsequent pages use URL params from Link header
                link_header = resp.headers.get("link", "")
                if 'rel="next"' in link_header:
                    for part in link_header.split(","):
                        if 'rel="next"' in part:
                            url = part.split("<")[1].split(">")[0]
                            break

                await asyncio.sleep(0.5)

        logger.info(f"Fetched {len(all_products)} updated products since {since}")
        return all_products

    # ── Text & metadata extraction ──────────────────────────────

    def extract_product_text(self, product: dict) -> str:
        """Combine product fields into a single embeddable text string."""
        parts = []

        title = product.get("title", "")
        if title:
            parts.append(title)

        vendor = product.get("vendor", "")
        if vendor:
            parts.append(f"Brand: {vendor}")

        product_type = product.get("product_type", "")
        if product_type:
            parts.append(f"Type: {product_type}")

        # Strip HTML from body
        body_html = product.get("body_html", "")
        if body_html:
            soup = BeautifulSoup(body_html, "html.parser")
            body_text = soup.get_text(separator=" ", strip=True)
            if body_text:
                parts.append(body_text)

        tags = product.get("tags", "")
        if tags:
            parts.append(f"Tags: {tags}")

        # Variant info (price, SKU)
        variants = product.get("variants", [])
        if variants:
            v = variants[0]
            price = v.get("price", "")
            if price:
                parts.append(f"Price: {price}")

        return "\n".join(parts)

    def extract_metadata(self, product: dict) -> dict:
        """Extract structured metadata from a Shopify product."""
        variants = product.get("variants", [])
        main_variant = variants[0] if variants else {}

        handle = product.get("handle", "")
        image = product.get("image") or {}

        metadata = {
            "shopify_id": product.get("id"),
            "title": product.get("title", ""),
            "handle": handle,
            "url": f"https://{self.shop_domain}/products/{handle}" if handle else "",
            "sku": main_variant.get("sku", ""),
            "price": float(main_variant.get("price", 0) or 0),
            "compare_at_price": float(main_variant.get("compare_at_price", 0) or 0) or None,
            "brand": product.get("vendor", ""),
            "product_type": product.get("product_type", ""),
            "tags": [t.strip() for t in product.get("tags", "").split(",") if t.strip()],
            "inventory_quantity": sum(
                v.get("inventory_quantity", 0) or 0 for v in variants
            ),
            "variants_count": len(variants),
            "image_url": image.get("src", ""),
            "created_at": product.get("created_at", ""),
            "updated_at": product.get("updated_at", ""),
            "status": product.get("status", "active"),
        }

        # Metafields (from GraphQL flattened format)
        metafields = product.get("metafields")
        if isinstance(metafields, dict):
            metadata["metafields"] = metafields

        # Collection IDs (from GraphQL)
        collection_ids = product.get("collection_ids")
        if collection_ids:
            metadata["collection_ids"] = collection_ids

        # SEO (from GraphQL)
        if product.get("seo_title"):
            metadata["seo_title"] = product["seo_title"]
        if product.get("seo_description"):
            metadata["seo_description"] = product["seo_description"]

        # Parse airsoft-specific specs
        specs = self.parse_airsoft_specs(product)
        metadata.update(specs)

        return metadata

    def extract_collection_text(self, collection: dict) -> str:
        """Combine collection fields into embeddable text."""
        parts = []
        title = collection.get("title", "")
        if title:
            parts.append(title)
        body_html = collection.get("body_html", "")
        if body_html:
            soup = BeautifulSoup(body_html, "html.parser")
            body_text = soup.get_text(separator=" ", strip=True)
            if body_text:
                parts.append(body_text)
        seo_desc = collection.get("seo_description", "")
        if seo_desc:
            parts.append(seo_desc)
        return "\n".join(parts)

    def extract_collection_metadata(self, collection: dict) -> dict:
        """Extract structured metadata from a collection."""
        handle = collection.get("handle", "")
        return {
            "shopify_id": collection.get("id"),
            "title": collection.get("title", ""),
            "handle": handle,
            "url": f"https://{self.shop_domain}/collections/{handle}" if handle else "",
            "image_url": collection.get("image_url", ""),
            "products_count": collection.get("products_count", 0),
            "updated_at": collection.get("updated_at", ""),
            "seo_title": collection.get("seo_title", ""),
            "seo_description": collection.get("seo_description", ""),
        }

    def extract_page_text(self, page: dict) -> str:
        """Combine page fields into embeddable text, stripping HTML."""
        parts = []
        title = page.get("title", "")
        if title:
            parts.append(title)
        body_html = page.get("body_html", "") or page.get("body_summary", "")
        if body_html:
            soup = BeautifulSoup(body_html, "html.parser")
            body = soup.get_text(separator="\n", strip=True)
            if body:
                parts.append(body)
        return "\n".join(parts)

    def extract_page_metadata(self, page: dict) -> dict:
        """Extract structured metadata from a page."""
        handle = page.get("handle", "")
        return {
            "shopify_id": page.get("id"),
            "title": page.get("title", ""),
            "handle": handle,
            "url": f"https://{self.shop_domain}/pages/{handle}" if handle else "",
            "created_at": page.get("created_at", ""),
            "updated_at": page.get("updated_at", ""),
        }

    def parse_airsoft_specs(self, product: dict) -> dict:
        """Regex-based extraction of airsoft specs from product text.

        Parses FPS, caliber, material, weight, and category from
        product body, title, and tags.
        """
        # Combine all text sources
        body_html = product.get("body_html", "")
        body_text = ""
        if body_html:
            soup = BeautifulSoup(body_html, "html.parser")
            body_text = soup.get_text(separator=" ", strip=True)

        title = product.get("title", "")
        tags = product.get("tags", "")
        combined = f"{title} {body_text} {tags}".lower()

        specs = {
            "fps": None,
            "caliber": None,
            "material": None,
            "weight_grams": None,
            "category": self._classify_category(combined, product.get("product_type", "")),
        }

        # FPS extraction (e.g., "350 fps", "350fps", "350 FPS")
        fps_match = re.search(r"(\d{2,4})\s*fps", combined, re.IGNORECASE)
        if fps_match:
            specs["fps"] = int(fps_match.group(1))

        # Joule extraction as fallback (e.g., "1.2j", "1.2 joules")
        if not specs["fps"]:
            joule_match = re.search(r"(\d+\.?\d*)\s*(?:j(?:oules?)?)", combined, re.IGNORECASE)
            if joule_match:
                joules = float(joule_match.group(1))
                # Approximate: FPS = sqrt(joules / 0.00034) for 0.20g BB
                specs["fps"] = int((joules / 0.00034) ** 0.5)

        # Caliber
        if "6mm" in combined or "6 mm" in combined:
            specs["caliber"] = "6mm"
        elif "8mm" in combined or "8 mm" in combined:
            specs["caliber"] = "8mm"

        # Material
        material_patterns = [
            (r"\bmetal\b", "metal"),
            (r"\bfull\s*metal\b", "full metal"),
            (r"\bpolymer\b", "polymer"),
            (r"\babs\b", "ABS"),
            (r"\bnylon\b", "nylon"),
            (r"\bwood\b", "wood"),
            (r"\bfiber\b", "fiber"),
            (r"\balumini?um\b", "aluminum"),
            (r"\bzinc\b", "zinc alloy"),
        ]
        for pattern, mat in material_patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                specs["material"] = mat
                break

        # Weight (grams or kg)
        weight_g = re.search(r"(\d+\.?\d*)\s*(?:g(?:rams?)?)\b", combined, re.IGNORECASE)
        weight_kg = re.search(r"(\d+\.?\d*)\s*kg\b", combined, re.IGNORECASE)
        if weight_g:
            w = float(weight_g.group(1))
            if 50 < w < 10000:  # reasonable airsoft gun weight in grams
                specs["weight_grams"] = w
        elif weight_kg:
            w = float(weight_kg.group(1))
            if 0.1 < w < 10:  # reasonable weight in kg
                specs["weight_grams"] = w * 1000

        return specs

    @staticmethod
    def _classify_category(text: str, product_type: str) -> str:
        """Classify airsoft product category from text."""
        text = f"{text} {product_type}".lower()

        # Order matters: more specific first
        categories = [
            (["gbb", "gas blowback", "gas blow back"], "gbb"),
            (["aeg", "electric", "electrica", "automatica"], "aeg"),
            (["sniper", "bolt action", "bolt-action", "cerrojo"], "sniper"),
            (["pistol", "pistola", "handgun", "sidearm"], "pistol"),
            (["shotgun", "escopeta"], "shotgun"),
            (["smg", "submachine", "subfusil"], "smg"),
            (["grenade", "launcher", "lanzagranadas", "granada"], "launcher"),
            (["magazine", "cargador", "mag "], "magazine"),
            (["battery", "bateria", "charger", "cargador bater"], "battery"),
            (["scope", "sight", "optic", "mira", "visor", "red dot", "holographic"], "optic"),
            (["bb", "bola", "municion", "ammo"], "ammunition"),
            (["vest", "plate carrier", "chaleco", "gear", "tactical"], "gear"),
            (["mask", "goggle", "proteccion", "protection", "gafa"], "protection"),
        ]

        for keywords, cat in categories:
            if any(kw in text for kw in keywords):
                return cat

        return "accessory"
