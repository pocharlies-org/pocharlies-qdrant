"""
Price List Parser for Pocharlies RAG
Parses PDF, Excel, and CSV price lists from suppliers.
Extracts product names, SKUs, and wholesale prices.
"""

import csv
import io
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _normalize_price(value) -> Optional[float]:
    """Convert various price formats to float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if 0.01 < float(value) < 100000 else None
    s = str(value).strip().replace(" ", "")
    s = re.sub(r"[€$£]", "", s)
    s = re.sub(r"EUR|USD|GBP", "", s, flags=re.IGNORECASE)
    s = s.strip()
    if not s:
        return None
    # Handle European format: 1.234,56 → 1234.56
    if re.match(r"^\d{1,3}(\.\d{3})+,\d{2}$", s):
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        p = float(s)
        return p if 0.01 < p < 100000 else None
    except (ValueError, TypeError):
        return None


# Column header patterns for auto-detection
HEADER_PATTERNS = {
    "name": re.compile(r"product|name|article|description|descripci[oó]n|nombre|titel|produkt", re.I),
    "sku": re.compile(r"sku|code|c[oó]digo|ref|reference|article.?n|item.?n|ean|barcode", re.I),
    "brand": re.compile(r"brand|marca|marque|manufacturer|fabricant", re.I),
    "price": re.compile(r"price|precio|prix|preis|cena|cost|nett?o?|wholesale|b2b|pvd|dealer", re.I),
    "retail_price": re.compile(r"retail|pvp|rrp|msrp|recommended|suggested|venta", re.I),
    "category": re.compile(r"category|categor[ií]a|cat[eé]gorie|type|tipo", re.I),
    "stock": re.compile(r"stock|qty|quantity|cantidad|disponib|available|inventory", re.I),
}


def _detect_columns(headers: List[str]) -> Dict[str, int]:
    """Map column names to their indices using pattern matching."""
    mapping = {}
    for i, header in enumerate(headers):
        header_str = str(header).strip()
        if not header_str:
            continue
        for field, pattern in HEADER_PATTERNS.items():
            if pattern.search(header_str) and field not in mapping:
                mapping[field] = i
                break
    return mapping


def _row_to_product(row: list, col_map: Dict[str, int]) -> Optional[dict]:
    """Convert a spreadsheet row to a product dict using column mapping."""
    def get(field: str) -> str:
        idx = col_map.get(field)
        if idx is not None and idx < len(row):
            val = row[idx]
            return str(val).strip() if val is not None else ""
        return ""

    name = get("name")
    if not name or len(name) < 3:
        return None

    price = _normalize_price(get("price"))
    retail = _normalize_price(get("retail_price"))

    return {
        "name": name,
        "sku": get("sku"),
        "brand": get("brand"),
        "wholesale_price": price,
        "price": retail or price,
        "category": get("category"),
        "in_stock": bool(get("stock")),
        "source_type": "pricelist_upload",
        "confidence": 0.9 if price else 0.5,
        "extraction_method": "pricelist",
    }


class PricelistParser:
    """Parses supplier price lists in PDF, Excel, and CSV formats."""

    def __init__(self, llm_client=None, llm_model: str = "smart"):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def parse(self, file_bytes: bytes, filename: str, supplier_slug: str = "") -> List[dict]:
        """Auto-detect format and parse."""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext == "pdf":
            return self.parse_pdf(file_bytes, supplier_slug)
        elif ext in ("xlsx", "xls"):
            return self.parse_excel(file_bytes, supplier_slug)
        elif ext == "csv":
            return self.parse_csv(file_bytes, supplier_slug)
        else:
            logger.warning(f"Unknown file type: {ext}")
            return []

    def parse_csv(self, file_bytes: bytes, supplier_slug: str = "") -> List[dict]:
        """Parse CSV price list."""
        text = file_bytes.decode("utf-8", errors="replace")
        # Try different delimiters
        for delimiter in [",", ";", "\t"]:
            reader = csv.reader(io.StringIO(text), delimiter=delimiter)
            rows = list(reader)
            if len(rows) < 2:
                continue

            # Find header row (first row with enough columns)
            header_idx = 0
            for i, row in enumerate(rows[:5]):
                if len([c for c in row if c.strip()]) >= 3:
                    header_idx = i
                    break

            col_map = _detect_columns(rows[header_idx])
            if "name" not in col_map and "sku" not in col_map:
                continue

            products = []
            for row in rows[header_idx + 1:]:
                p = _row_to_product(row, col_map)
                if p:
                    p["source_domain"] = supplier_slug
                    products.append(p)

            if products:
                logger.info(f"CSV: parsed {len(products)} products (delimiter='{delimiter}')")
                return products

        return []

    def parse_excel(self, file_bytes: bytes, supplier_slug: str = "") -> List[dict]:
        """Parse Excel price list."""
        try:
            import openpyxl
        except ImportError:
            logger.error("openpyxl not installed — cannot parse Excel files")
            return []

        try:
            wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
        except Exception as e:
            logger.error(f"Failed to open Excel file: {e}")
            return []

        all_products = []
        for sheet in wb.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                rows.append(list(row))
            if len(rows) < 2:
                continue

            # Find header row
            header_idx = 0
            for i, row in enumerate(rows[:10]):
                non_empty = [c for c in row if c is not None and str(c).strip()]
                if len(non_empty) >= 3:
                    header_idx = i
                    break

            col_map = _detect_columns([str(c) if c else "" for c in rows[header_idx]])
            if "name" not in col_map and "sku" not in col_map:
                continue

            for row in rows[header_idx + 1:]:
                p = _row_to_product(row, col_map)
                if p:
                    p["source_domain"] = supplier_slug
                    all_products.append(p)

        wb.close()
        logger.info(f"Excel: parsed {len(all_products)} products")
        return all_products

    def parse_pdf(self, file_bytes: bytes, supplier_slug: str = "") -> List[dict]:
        """Parse PDF price list using PyMuPDF for tables, LLM fallback for unstructured."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("PyMuPDF not installed — cannot parse PDF files")
            return []

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            return []

        all_products = []

        # Try table extraction first (PyMuPDF 1.23+)
        for page_num in range(len(doc)):
            page = doc[page_num]
            try:
                tables = page.find_tables()
                for table in tables:
                    data = table.extract()
                    if len(data) < 2:
                        continue

                    col_map = _detect_columns([str(c) if c else "" for c in data[0]])
                    if "name" not in col_map and "sku" not in col_map:
                        continue

                    for row in data[1:]:
                        p = _row_to_product(row, col_map)
                        if p:
                            p["source_domain"] = supplier_slug
                            all_products.append(p)
            except Exception:
                pass  # Table extraction not supported or failed

        if all_products:
            doc.close()
            logger.info(f"PDF tables: parsed {len(all_products)} products")
            return all_products

        # Fallback: extract text and use LLM
        if self.llm_client:
            full_text = ""
            for page_num in range(min(len(doc), 10)):  # Cap at 10 pages
                page = doc[page_num]
                full_text += page.get_text() + "\n"
            doc.close()

            if len(full_text.strip()) < 50:
                return []

            return self._llm_extract_products(full_text, supplier_slug)

        doc.close()
        return []

    def _llm_extract_products(self, text: str, supplier_slug: str) -> List[dict]:
        """Use LLM to extract products from unstructured text."""
        if not self.llm_client:
            return []

        # Truncate to ~4000 chars to fit in context
        text = text[:4000]

        prompt = f"""Extract product data from this supplier price list text. Return a JSON array of objects with these fields:
- name: product name
- sku: SKU/article number (if available)
- brand: brand name (if available)
- wholesale_price: the wholesale/B2B/net price as a number
- currency: EUR, USD, etc.
- category: product category (if detectable)

Only include entries that have at least a name and a price. Return ONLY the JSON array, no other text.

Text:
{text}"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=4000,
            )
            content = response.choices[0].message.content.strip()

            # Extract JSON from response
            if content.startswith("```"):
                content = re.sub(r"```\w*\n?", "", content).strip()

            products = json.loads(content)
            if not isinstance(products, list):
                return []

            result = []
            for p in products:
                if not p.get("name"):
                    continue
                result.append({
                    "name": p.get("name", ""),
                    "sku": p.get("sku", ""),
                    "brand": p.get("brand", ""),
                    "wholesale_price": _normalize_price(p.get("wholesale_price")),
                    "price": _normalize_price(p.get("wholesale_price")),
                    "currency": p.get("currency", "EUR"),
                    "category": p.get("category", ""),
                    "source_domain": supplier_slug,
                    "source_type": "pricelist_upload",
                    "confidence": 0.7,
                    "extraction_method": "llm-pdf",
                })

            logger.info(f"LLM PDF extraction: {len(result)} products")
            return result

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []
