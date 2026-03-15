"""
Vault Builder — Crawls competitors and generates Obsidian-compatible markdown notes.

Orchestrates: crawl → classify → resolve → aggregate → render templates → write vault.
Uses existing rag-service endpoints for crawling and classification.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from jinja2 import Environment, FileSystemLoader
from fast_product_extractor import FastProductExtractor
from firecrawl_client import FirecrawlClient

logger = logging.getLogger(__name__)

VAULT_HASH_PREFIX = "vault:hash"


@dataclass
class CompetitorConfig:
    url: str
    name: str
    slug: str
    max_pages: int = 500
    max_depth: int = 3
    crawl_delay: float = 1.0
    url_include_patterns: List[str] = field(default_factory=list)
    use_firecrawl: bool = False  # Use Firecrawl (headless browser) for Cloudflare-protected sites


@dataclass
class BrandAggregate:
    name: str
    slug: str
    origin: str = "Unknown"
    competitors: List[dict] = field(default_factory=list)  # [{slug, count}]
    categories: List[str] = field(default_factory=list)
    total_competitor_products: int = 0
    avg_competitor_price: float = 0.0
    our_product_count: int = 0
    category_breakdown: List[dict] = field(default_factory=list)
    overview: str = ""
    opportunity_notes: str = ""


@dataclass
class CategoryAggregate:
    name: str
    slug: str
    competitors: List[dict] = field(default_factory=list)  # [{slug, count, avg_price}]
    brands: List[str] = field(default_factory=list)
    brand_breakdown: List[dict] = field(default_factory=list)
    our_products: int = 0
    our_avg_price: float = 0.0
    competitor_avg_products: int = 0
    avg_price: float = 0.0
    market_overview: str = ""
    translation_notes: str = ""


@dataclass
class CompetitorResult:
    config: CompetitorConfig
    products: List[dict] = field(default_factory=list)
    matched: List[dict] = field(default_factory=list)
    unmatched: List[dict] = field(default_factory=list)
    crawl_job_id: str = ""
    error: Optional[str] = None
    overview: str = ""
    strengths_weaknesses: str = ""
    seo_notes: str = ""


@dataclass
class VaultBuildResult:
    started_at: str = ""
    completed_at: str = ""
    competitors_processed: int = 0
    competitors_failed: int = 0
    notes_written: int = 0
    errors: List[str] = field(default_factory=list)
    competitor_results: Dict[str, dict] = field(default_factory=dict)


# ── Slug helpers ──────────────────────────────────────────────


def _slugify(text: str) -> str:
    """Convert text to a vault-safe slug."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def _categorize_slug(cat: str) -> str:
    """Normalize category names to slugs."""
    mapping = {
        "aeg": "aeg-rifles",
        "gbb": "gbb-pistols",
        "gbbr": "gbb-rifles",
        "sniper": "sniper-rifles",
        "pistol": "gbb-pistols",
        "shotgun": "shotguns",
        "smg": "smg",
        "accessory": "accessories",
        "gear": "tactical-gear",
        "optics": "optics-sights",
        "battery": "batteries-chargers",
        "bb": "bbs-ammunition",
        "magazine": "magazines",
        "upgrade": "upgrade-parts",
    }
    return mapping.get(cat.lower(), _slugify(cat))


# ── Jinja2 filters ───────────────────────────────────────────


def _format_wikilink(slug: str) -> str:
    return f"[[{slug}]]"


# ── Human-edit preservation ──────────────────────────────────

_HUMAN_BLOCK_RE = re.compile(
    r"(<!-- human-start -->.*?<!-- human-end -->)",
    re.DOTALL,
)


def _extract_human_blocks(content: str) -> List[str]:
    """Extract all human-preserved blocks from existing note."""
    return _HUMAN_BLOCK_RE.findall(content)


def _inject_human_blocks(new_content: str, old_blocks: List[str]) -> str:
    """Replace placeholder human blocks with previously preserved content."""
    new_blocks = _HUMAN_BLOCK_RE.findall(new_content)
    for i, old_block in enumerate(old_blocks):
        if i < len(new_blocks):
            new_content = new_content.replace(new_blocks[i], old_block, 1)
    return new_content


# ── Main Builder ─────────────────────────────────────────────


class VaultBuilder:
    """Builds an Obsidian knowledge vault from competitor crawl data."""

    def __init__(
        self,
        vault_path: str,
        config_path: str = None,
        rag_base_url: str = "http://localhost:5000",
        llm_client=None,
        redis_client=None,
        glossary: dict = None,
        competitor_indexer=None,
        product_classifier=None,
        crawl_jobs: dict = None,
        crawl_queue: list = None,
    ):
        self.vault_path = Path(vault_path)
        self.rag_base_url = rag_base_url.rstrip("/")
        self.llm_client = llm_client
        self.redis = redis_client
        self.glossary = glossary or {}
        # Direct references to in-process objects (avoid HTTP self-calls)
        self._competitor_indexer = competitor_indexer
        self._product_classifier = product_classifier
        self._crawl_jobs = crawl_jobs if crawl_jobs is not None else {}
        self._crawl_queue = crawl_queue if crawl_queue is not None else []

        # Fast product extractor (structured data, no LLM needed)
        self._fast_extractor = None
        if competitor_indexer and hasattr(competitor_indexer, 'client'):
            self._fast_extractor = FastProductExtractor(
                qdrant_client=competitor_indexer.client,
            )
            logger.info("Fast product extractor initialized (no-LLM extraction)")

        # Firecrawl client (headless browser for Cloudflare-protected sites)
        self._firecrawl = FirecrawlClient()
        logger.info(f"Firecrawl client: {self._firecrawl.base_url}")

        # Load config
        self.competitors: List[CompetitorConfig] = []
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            for c in cfg.get("competitors", []):
                self.competitors.append(CompetitorConfig(**c))
            logger.info(f"Loaded {len(self.competitors)} competitors from config")

        # Jinja2 environment
        templates_dir = self.vault_path / "_templates"
        if templates_dir.exists():
            self.jinja = Environment(
                loader=FileSystemLoader(str(templates_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
            )
            self.jinja.filters["format_wikilink"] = _format_wikilink
        else:
            self.jinja = None
            logger.warning(f"Templates directory not found: {templates_dir}")

    # ── Redis lock ────────────────────────────────────────────

    LOCK_KEY = "knowledge:rebuild:lock"
    LOCK_TTL = 7200  # 2 hours

    async def _acquire_lock(self) -> bool:
        if not self.redis:
            return True
        result = await self.redis.set(self.LOCK_KEY, "locked", ex=self.LOCK_TTL, nx=True)
        return result is not None

    async def _release_lock(self):
        if self.redis:
            await self.redis.delete(self.LOCK_KEY)

    # ── Crawl deduplication ──────────────────────────────────

    def _dedup_crawl_results(self, chunks: List[dict], domain: str) -> tuple:
        """Remove duplicate pages from crawl results using content hash.

        Tracks seen URLs via Redis (key: crawl:seen:{domain}:{url_hash}).
        Returns (deduplicated_chunks, stats_dict).
        """
        total = len(chunks)
        if not total:
            return chunks, {"total": 0, "new": 0, "duplicate": 0, "duplicate_pct": 0}

        seen_urls = set()
        new_chunks = []
        duplicate_count = 0

        for chunk in chunks:
            url = chunk.get("url", "")
            if not url:
                new_chunks.append(chunk)
                continue

            # Normalize URL (strip trailing slash, query params for dedup)
            normalized = url.rstrip("/").split("?")[0].split("#")[0]
            if normalized in seen_urls:
                duplicate_count += 1
                continue
            seen_urls.add(normalized)
            new_chunks.append(chunk)

        dup_pct = (duplicate_count / total * 100) if total > 0 else 0

        return new_chunks, {
            "total": total,
            "new": len(new_chunks),
            "duplicate": duplicate_count,
            "duplicate_pct": dup_pct,
        }

    # ── Core build methods ────────────────────────────────────

    async def build_full(self, competitors: List[CompetitorConfig] = None) -> VaultBuildResult:
        """Full rebuild: crawl all competitors, regenerate all notes."""
        if not await self._acquire_lock():
            logger.warning("Rebuild already in progress (Redis lock held)")
            result = VaultBuildResult()
            result.errors.append("Rebuild already in progress")
            return result

        try:
            return await self._do_build(competitors or self.competitors)
        finally:
            await self._release_lock()

    async def build_incremental(self, domain: str) -> VaultBuildResult:
        """Re-crawl a single competitor and update affected notes."""
        if not await self._acquire_lock():
            result = VaultBuildResult()
            result.errors.append("Rebuild already in progress")
            return result

        try:
            targets = [c for c in self.competitors if c.slug == domain or domain in c.url]
            if not targets:
                result = VaultBuildResult()
                result.errors.append(f"Unknown competitor: {domain}")
                return result
            return await self._do_build(targets)
        finally:
            await self._release_lock()

    async def _do_build(self, competitors: List[CompetitorConfig]) -> VaultBuildResult:
        result = VaultBuildResult(
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        # Phase 1: Crawl + extract + resolve each competitor
        all_results: Dict[str, CompetitorResult] = {}
        for config in competitors:
            logger.info(f"Processing competitor: {config.name} ({config.url})")
            try:
                cr = await self._process_competitor(config)
                all_results[config.slug] = cr
                result.competitors_processed += 1
                result.competitor_results[config.slug] = {
                    "name": config.name,
                    "products": len(cr.products),
                    "matched": len(cr.matched),
                    "unmatched": len(cr.unmatched),
                    "error": cr.error,
                }
            except Exception as e:
                logger.error(f"Failed to process {config.name}: {e}")
                result.competitors_failed += 1
                result.errors.append(f"{config.name}: {str(e)[:200]}")
                result.competitor_results[config.slug] = {
                    "name": config.name,
                    "error": str(e)[:200],
                }

        if not all_results:
            result.errors.append("No competitors processed successfully")
            result.completed_at = datetime.now(timezone.utc).isoformat()
            self._write_meta(result)
            return result

        # Phase 2: Aggregate by brand and category
        brands = self._aggregate_brands(all_results)
        categories = self._aggregate_categories(all_results)

        # Phase 3: Detect glossary gaps
        glossary_gaps = self._detect_glossary_gaps(all_results)

        # Phase 4: Generate LLM summaries (if llm_client available)
        if self.llm_client:
            await self._generate_summaries(all_results, brands, categories)

        # Phase 5: Render and write notes
        notes_written = 0

        # Competitor notes
        for slug, cr in all_results.items():
            config = next(c for c in competitors if c.slug == slug)
            notes_written += self._write_competitor_note(config, cr, brands, categories)

        # Brand notes
        for slug, brand in brands.items():
            notes_written += self._write_brand_note(brand)

        # Category notes
        for slug, cat in categories.items():
            notes_written += self._write_category_note(cat)

        # Market gap note
        notes_written += self._write_market_gap_note(all_results)

        # Glossary gaps note
        notes_written += self._write_glossary_gaps_note(glossary_gaps, all_results)

        result.notes_written = notes_written
        result.completed_at = datetime.now(timezone.utc).isoformat()
        self._write_meta(result)

        logger.info(
            f"Vault build complete: {result.competitors_processed} competitors, "
            f"{result.notes_written} notes written, {result.competitors_failed} failed"
        )
        return result

    # ── Competitor processing ─────────────────────────────────

    async def _process_competitor(self, config: CompetitorConfig) -> CompetitorResult:
        """Crawl a competitor, extract products, resolve against catalog.

        Uses in-process references to competitor_indexer and product_classifier
        to avoid HTTP self-call issues (SSE endpoints, job dict isolation).
        Falls back to HTTP calls if direct references not available.
        """
        cr = CompetitorResult(config=config)
        domain = config.url.replace("https://", "").replace("http://", "").rstrip("/")

        # ── 1. Crawl ──
        firecrawl_used = False

        # Use Firecrawl directly for sites marked as needing headless browser
        if config.use_firecrawl and self._firecrawl:
            logger.info(f"Using Firecrawl (headless browser) for {config.url}")
            if await self._firecrawl.is_available():
                fc_chunks = await self._firecrawl.crawl_to_chunks(
                    config.url,
                    max_pages=config.max_pages,
                    max_depth=config.max_depth,
                )
                if fc_chunks:
                    # Dedup: remove pages already seen in previous crawls
                    fc_chunks, dedup_stats = self._dedup_crawl_results(fc_chunks, domain)
                    logger.info(
                        f"Firecrawl {config.url}: {dedup_stats['total']} pages crawled, "
                        f"{dedup_stats['new']} new, {dedup_stats['duplicate']} duplicates "
                        f"({dedup_stats['duplicate_pct']:.0f}% repeat)"
                    )
                    firecrawl_used = True
                    cr._firecrawl_chunks = fc_chunks
                    cr.crawl_job_id = f"firecrawl-{config.slug}"
                else:
                    logger.warning(f"Firecrawl returned 0 pages for {config.url}")
                    cr.error = "Firecrawl crawl returned no results"
                    return cr
            else:
                logger.warning(f"Firecrawl not available, skipping {config.url}")
                cr.error = "Firecrawl not available for Cloudflare-protected site"
                return cr

        elif self._competitor_indexer:
            logger.info(f"Starting in-process crawl of {config.url}")
            from web_indexer import CrawlJob
            job = await self._competitor_indexer.crawl_and_index(
                start_url=config.url,
                max_depth=config.max_depth,
                max_pages=config.max_pages,
            )
            cr.crawl_job_id = job.job_id
            logger.info(f"Crawl completed: {job.pages_indexed} pages indexed")

            # Firecrawl fallback: if standard crawler got 0 pages (Cloudflare/JS site)
            if job.pages_indexed == 0 and self._firecrawl:
                logger.info(f"Standard crawler got 0 pages for {config.url}, trying Firecrawl (headless browser)...")
                if await self._firecrawl.is_available():
                    fc_chunks = await self._firecrawl.crawl_to_chunks(
                        config.url,
                        max_pages=config.max_pages,
                        max_depth=config.max_depth,
                    )
                    if fc_chunks:
                        firecrawl_used = True
                        cr._firecrawl_chunks = fc_chunks  # Store for direct product extraction
                        logger.info(f"Firecrawl success: {len(fc_chunks)} chunks from {config.url}")
                        job.pages_indexed = len(fc_chunks)
                    else:
                        logger.warning(f"Firecrawl also returned 0 results for {config.url}")
                else:
                    logger.warning("Firecrawl not available, skipping fallback")

            if job.status == "failed" and not firecrawl_used:
                cr.error = f"Crawl failed: {job.error or 'unknown'}"
                return cr
        else:
            # Fallback: HTTP call + poll crawl_jobs dict
            logger.info(f"Starting HTTP crawl of {config.url}")
            async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0)) as client:
                resp = await client.post(
                    f"{self.rag_base_url}/competitor/index-url",
                    json={
                        "url": config.url,
                        "max_depth": config.max_depth,
                        "max_pages": config.max_pages,
                    },
                )
                resp.raise_for_status()
                job_id = resp.json()["job_id"]
                cr.crawl_job_id = job_id

                # Poll the in-memory crawl_jobs dict directly
                for _ in range(180):
                    await asyncio.sleep(10)
                    job = self._crawl_jobs.get(job_id)
                    if job and job.status in ("completed", "failed", "stopped"):
                        logger.info(f"Crawl {job_id} {job.status}: {job.pages_indexed} pages")
                        break
                else:
                    cr.error = "Crawl timeout after 30 minutes"
                    return cr

                if job and job.status == "failed":
                    cr.error = f"Crawl failed"
                    return cr

        # ── 2. Extract products ──
        # Primary: Fast structured-data extraction (JSON-LD, Open Graph, regex) — instant, no LLM
        # Fallback: LLM-based extraction (slow, rate-limited)
        domain_variants = [domain]
        if not domain.startswith("www."):
            domain_variants.append(f"www.{domain}")
        else:
            domain_variants.append(domain[4:])

        logger.info(f"Extracting products from {config.slug} (domains: {domain_variants})")

        # If we used Firecrawl, extract products directly from markdown chunks
        if hasattr(cr, '_firecrawl_chunks') and cr._firecrawl_chunks:
            from fast_product_extractor import BRAND_PATTERNS, _detect_category
            seen = set()
            for chunk in cr._firecrawl_chunks:
                text = chunk.get("text", "")
                url = chunk.get("url", "")
                title = chunk.get("title", "")
                # Extract brand mentions from markdown content
                text_lower = (text + " " + title).lower()
                brand = ""
                for b in BRAND_PATTERNS:
                    if b.lower() in text_lower:
                        brand = b
                        break
                # Try to find price
                import re as _pricere
                price_match = _pricere.search(r'(\d{1,5}[.,]\d{2})\s*€|€\s*(\d{1,5}[.,]\d{2})', text)
                price = None
                if price_match:
                    ps = price_match.group(1) or price_match.group(2)
                    try:
                        price = float(ps.replace(",", "."))
                    except (ValueError, TypeError):
                        pass
                if title and title.lower() not in seen:
                    cr.products.append({
                        "name": title.split(" - ")[0].strip() if " - " in title else title,
                        "brand": brand,
                        "price": price,
                        "currency": "EUR",
                        "category": _detect_category(title + " " + text[:500]),
                        "source_url": url,
                        "source_domain": domain,
                        "confidence": 0.6,
                        "extraction_method": "firecrawl",
                    })
                    seen.add(title.lower())
            logger.info(f"Firecrawl extraction: {len(cr.products)} products from {config.slug}")

        # Try fast extraction first (no LLM, instant)
        if not cr.products and self._fast_extractor:
            for d in domain_variants:
                cr.products = self._fast_extractor.extract_products(d, "competitor_products")
                if cr.products:
                    logger.info(f"Fast extraction: {len(cr.products)} products from {d}")
                    break
            if not cr.products:
                logger.info(f"Fast extraction found 0 products for {config.slug}, trying web_pages collection")
                for d in domain_variants:
                    cr.products = self._fast_extractor.extract_products(d, "web_pages")
                    if cr.products:
                        logger.info(f"Fast extraction (web_pages): {len(cr.products)} products from {d}")
                        break

        # If fast extraction found nothing and LLM classifier is available, use it as fallback
        # (disabled by default to avoid rate limiting — enable with VAULT_LLM_EXTRACT=true)
        if not cr.products and self._product_classifier and os.getenv("VAULT_LLM_EXTRACT", "").lower() == "true":
            logger.info(f"Falling back to LLM extraction for {config.slug}")
            for try_domain in domain_variants:
                try:
                    ej = await self._product_classifier.extract_products_from_collection(
                        domain=try_domain,
                        collection_name="competitor_products",
                        batch_size=10,
                    )
                    if ej.results:
                        cr.products = [p.to_dict() if hasattr(p, 'to_dict') else p for p in ej.results]
                        logger.info(f"LLM extraction: {len(cr.products)} products from {try_domain}")
                        break
                except Exception as e:
                    logger.warning(f"LLM extraction with domain {try_domain} failed: {e}")

        logger.info(f"Total: {len(cr.products)} products extracted from {config.slug}")

        # ── 3. Resolve against our catalog ──
        if cr.products and self._product_classifier:
            try:
                matches = await self._product_classifier.resolve_entities(
                    extracted_products=cr.products if isinstance(cr.products[0], dict) else [p.to_dict() for p in cr.products],
                    top_k=3,
                )
                cr.matched = matches.get("matched", [])
                cr.unmatched = matches.get("unmatched", [])
            except Exception as e:
                logger.warning(f"Entity resolution failed for {config.slug}: {e}")
                cr.unmatched = cr.products
        elif cr.products:
            # All unmatched if no classifier
            cr.unmatched = cr.products

        return cr

    # ── Aggregation ───────────────────────────────────────────

    def _aggregate_brands(self, all_results: Dict[str, CompetitorResult]) -> Dict[str, BrandAggregate]:
        """Aggregate brand data across all competitors."""
        brands: Dict[str, BrandAggregate] = {}

        for slug, cr in all_results.items():
            brand_counts: Dict[str, list] = defaultdict(list)
            for p in cr.products:
                brand_name = p.get("brand", "Unknown")
                if not brand_name or brand_name.lower() in ("unknown", "n/a", ""):
                    continue
                brand_counts[brand_name].append(p)

            for brand_name, products in brand_counts.items():
                brand_slug = _slugify(brand_name)
                if brand_slug not in brands:
                    brands[brand_slug] = BrandAggregate(
                        name=brand_name,
                        slug=brand_slug,
                    )

                b = brands[brand_slug]
                b.competitors.append({"slug": slug, "count": len(products)})
                b.total_competitor_products += len(products)

                prices = [p.get("price", 0) for p in products if p.get("price")]
                if prices:
                    b.avg_competitor_price = round(
                        (b.avg_competitor_price * (b.total_competitor_products - len(products)) +
                         sum(prices)) / b.total_competitor_products, 2
                    )

                for p in products:
                    cat = _categorize_slug(p.get("category", "other"))
                    if cat not in b.categories:
                        b.categories.append(cat)

        return brands

    def _aggregate_categories(self, all_results: Dict[str, CompetitorResult]) -> Dict[str, CategoryAggregate]:
        """Aggregate category data across all competitors."""
        categories: Dict[str, CategoryAggregate] = {}

        for slug, cr in all_results.items():
            cat_counts: Dict[str, list] = defaultdict(list)
            for p in cr.products:
                cat = _categorize_slug(p.get("category", "other"))
                cat_counts[cat].append(p)

            for cat_name, products in cat_counts.items():
                if cat_name not in categories:
                    categories[cat_name] = CategoryAggregate(
                        name=cat_name.replace("-", " ").title(),
                        slug=cat_name,
                    )

                c = categories[cat_name]
                prices = [p.get("price", 0) for p in products if p.get("price")]
                avg_p = round(sum(prices) / len(prices), 2) if prices else 0
                c.competitors.append({"slug": slug, "count": len(products), "avg_price": avg_p})

                for p in products:
                    brand_slug = _slugify(p.get("brand", ""))
                    if brand_slug and brand_slug not in c.brands:
                        c.brands.append(brand_slug)

        # Compute averages
        for c in categories.values():
            if c.competitors:
                c.competitor_avg_products = round(
                    sum(comp["count"] for comp in c.competitors) / len(c.competitors)
                )
                all_prices = [comp["avg_price"] for comp in c.competitors if comp["avg_price"]]
                c.avg_price = round(sum(all_prices) / len(all_prices), 2) if all_prices else 0

        return categories

    # ── Glossary gap detection ────────────────────────────────

    def _detect_glossary_gaps(self, all_results: Dict[str, CompetitorResult]) -> List[dict]:
        """Find Spanish terms in competitor content not in our glossary."""
        if not self.glossary:
            return []

        # Build set of known ES terms
        known_es = set()
        for en_term, translations in self.glossary.items():
            es_term = translations.get("es", "")
            if es_term:
                known_es.add(es_term.lower())

        # Collect Spanish terms from competitor products
        term_freq: Dict[str, Dict[str, Any]] = {}  # term -> {count, sources}
        for slug, cr in all_results.items():
            for p in cr.products:
                # Extract potential glossary terms from product names/descriptions
                text = f"{p.get('name', '')} {p.get('raw_description', '')}"
                words = re.findall(r"\b[a-záéíóúüñ]{4,}\b", text.lower())
                # Also extract 2-word phrases
                text_lower = text.lower()
                bigrams = re.findall(r"\b([a-záéíóúüñ]+ [a-záéíóúüñ]+)\b", text_lower)

                for term in bigrams:
                    if term not in known_es and len(term) > 5:
                        if term not in term_freq:
                            term_freq[term] = {"count": 0, "sources": set()}
                        term_freq[term]["count"] += 1
                        term_freq[term]["sources"].add(slug)

        # Filter to meaningful gaps (frequency > 3)
        gaps = []
        for term, data in sorted(term_freq.items(), key=lambda x: -x[1]["count"]):
            if data["count"] >= 3:
                gaps.append({
                    "term_es": term,
                    "suggested_en": "",  # Would need LLM to translate
                    "sources": list(data["sources"]),
                    "frequency": data["count"],
                    "source": list(data["sources"])[0] if data["sources"] else "",
                })
        return gaps[:50]  # top 50

    # ── LLM summaries ─────────────────────────────────────────

    async def _generate_summaries(
        self,
        all_results: Dict[str, CompetitorResult],
        brands: Dict[str, BrandAggregate],
        categories: Dict[str, CategoryAggregate],
    ):
        """Generate deep LLM analysis for competitors, brands, and categories."""

        # ── Per-competitor deep analysis ──
        for slug, cr in all_results.items():
            try:
                # Build rich context from extracted products
                brand_counts = defaultdict(int)
                category_counts = defaultdict(int)
                prices = []
                sample_products = []
                for p in cr.products:
                    b = p.get("brand", "")
                    if b and b.lower() not in ("unknown", ""):
                        brand_counts[b] += 1
                    category_counts[p.get("category", "other")] += 1
                    if p.get("price"):
                        prices.append(p["price"])
                    if len(sample_products) < 30:
                        sample_products.append(
                            f"- {p.get('name', '?')[:80]} | {b or '?'} | €{p.get('price', '?')} | {p.get('category', '?')}"
                        )

                top_brands = sorted(brand_counts.items(), key=lambda x: -x[1])[:15]
                top_cats = sorted(category_counts.items(), key=lambda x: -x[1])[:10]
                price_info = ""
                if prices:
                    price_info = f"Price range: €{min(prices):.2f} - €{max(prices):.2f}, avg €{sum(prices)/len(prices):.2f}"

                # Also include raw page content samples if available (from Firecrawl)
                content_samples = ""
                if hasattr(cr, '_firecrawl_chunks') and cr._firecrawl_chunks:
                    # Pick 3 diverse page contents for the LLM to analyze
                    fc_chunks = cr._firecrawl_chunks
                    step = max(1, len(fc_chunks) // 3)
                    for i in range(0, min(len(fc_chunks), 3 * step), step):
                        if i < len(fc_chunks):
                            chunk = fc_chunks[i]
                            content_samples += f"\n--- Page: {chunk.get('title', '?')} ({chunk.get('url', '')}) ---\n"
                            content_samples += chunk.get("text", "")[:800] + "\n"

                analysis_prompt = f"""Analyze the airsoft retailer "{cr.config.name}" ({cr.config.url}) as a competitor to Skirmshop.es (our store).

DATA EXTRACTED:
- Total products found: {len(cr.products)}
- {price_info}
- Top brands: {', '.join(f'{b} ({c})' for b, c in top_brands)}
- Categories: {', '.join(f'{c} ({n})' for c, n in top_cats)}

SAMPLE PRODUCTS:
{chr(10).join(sample_products[:20])}
{f'''
WEBSITE CONTENT SAMPLES:
{content_samples[:2000]}''' if content_samples else ''}

Provide a detailed competitive analysis in this EXACT format (no other text):

OVERVIEW:
[2-3 sentences: market positioning, target customer, unique value proposition]

STRENGTHS:
- [strength 1]
- [strength 2]
- [strength 3]

WEAKNESSES:
- [weakness 1]
- [weakness 2]
- [weakness 3]

SEO_NOTES:
[2-3 sentences about their content strategy, SEO approach, what keywords they target]

PRICING_STRATEGY:
[1-2 sentences about their pricing vs market]

OPPORTUNITY:
[2-3 sentences: what Skirmshop can learn from or exploit against this competitor]"""

                logger.info(f"LLM deep analysis for {cr.config.name} ({len(cr.products)} products, {len(top_brands)} brands)...")
                analysis = await self._llm_analyze(analysis_prompt)

                if analysis:
                    # Parse structured response
                    cr.overview = self._extract_section(analysis, "OVERVIEW")
                    cr.strengths_weaknesses = self._extract_section(analysis, "STRENGTHS") + "\n\n" + self._extract_section(analysis, "WEAKNESSES")
                    cr.seo_notes = self._extract_section(analysis, "SEO_NOTES")
                    cr.pricing_strategy = self._extract_section(analysis, "PRICING_STRATEGY", "")
                    cr.opportunity = self._extract_section(analysis, "OPPORTUNITY", "")
                else:
                    cr.overview = f"Airsoft retailer with {len(cr.products)} products across {len(top_brands)} brands."
                    cr.strengths_weaknesses = "_Analysis pending._"

            except Exception as e:
                logger.warning(f"LLM analysis failed for {slug}: {e}")
                cr.overview = f"Airsoft retailer with {len(cr.products)} products."
                cr.strengths_weaknesses = "_Analysis failed._"

        # ── Cross-competitor brand analysis ──
        for brand_slug, brand in brands.items():
            if brand.total_competitor_products >= 5:
                try:
                    comp_details = ", ".join(
                        f"{c['slug']} ({c['count']} products)" for c in brand.competitors
                    )
                    prompt = (
                        f"Brief analysis of brand '{brand.name}' in the Spanish airsoft market. "
                        f"Found across competitors: {comp_details}. "
                        f"Total products: {brand.total_competitor_products}, avg price €{brand.avg_competitor_price}. "
                        f"What's this brand's positioning? Is it a gap for our store (Skirmshop.es)? 2-3 sentences."
                    )
                    brand.overview = await self._llm_summarize(prompt)
                    brand.opportunity_notes = await self._llm_summarize(
                        f"Should Skirmshop.es stock {brand.name}? They have {brand.total_competitor_products} products "
                        f"across {len(brand.competitors)} competitors at avg €{brand.avg_competitor_price}. "
                        f"Categories: {', '.join(brand.categories[:5])}. Give a 1-2 sentence recommendation."
                    )
                except Exception as e:
                    logger.warning(f"Brand analysis failed for {brand_slug}: {e}")

    @staticmethod
    def _extract_section(text: str, section: str, default: str = "_Analysis pending._") -> str:
        """Extract a named section from structured LLM response."""
        import re
        pattern = rf"{section}:\s*\n(.*?)(?=\n[A-Z_]+:|$)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return default

    async def _llm_analyze(self, prompt: str) -> Optional[str]:
        """Call LLM for deep analysis (higher token limit)."""
        if not self.llm_client:
            return None
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm_client.chat.completions.create(
                    model=os.getenv("LLM_MODEL", "local"),
                    messages=[
                        {"role": "system", "content": (
                            "You are a senior airsoft market analyst for Skirmshop.es, a Spanish online airsoft retailer. "
                            "Analyze competitor data and provide actionable intelligence. "
                            "Output ONLY the analysis in the exact format requested. "
                            "No thinking process, no reasoning steps, no meta-commentary. "
                            "Be specific with numbers, brand names, and price points."
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1500,
                    timeout=120,
                ),
            )
            text = response.choices[0].message.content.strip()
            # Strip thinking tokens
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            text = re.sub(r"^Thinking Process:.*?(?=\n[A-Z_]+:)", "", text, flags=re.DOTALL).strip()
            if re.match(r"^\d+\.\s+\*\*(Analyze|Formulate|Interpret)", text):
                return None
            return text if text else None
        except Exception as e:
            logger.warning(f"LLM analysis call failed: {e}")
            return None

    async def _llm_summarize(self, prompt: str) -> str:
        """Call LLM for a brief summary."""
        if not self.llm_client:
            return "_LLM not configured._"
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm_client.chat.completions.create(
                    model=os.getenv("LLM_MODEL", "local"),
                    messages=[
                        {"role": "system", "content": "You are a concise airsoft market analyst. Output ONLY the final analysis text. Never show your reasoning, analysis steps, or thought process. No numbered steps. No bullet-pointed analysis. Just the direct answer in 2-3 sentences."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=300,
                    timeout=60,
                ),
            )
            text = response.choices[0].message.content.strip()
            # Strip thinking tokens from models that output chain-of-thought
            import re as _re
            # <think>...</think> blocks
            text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
            # "Thinking Process:" blocks (Qwen-style)
            text = _re.sub(r"^Thinking Process:.*?(?=\n[A-Z]|\n\n[^\s])", "", text, flags=_re.DOTALL).strip()
            # Numbered chain-of-thought reasoning (e.g., "1. **Analyze the Request:**...")
            text = _re.sub(r"^\d+\.\s+\*\*Analyze.*$", "", text, flags=_re.DOTALL | _re.MULTILINE).strip()
            # Strip entire response if it starts with numbered reasoning steps
            if _re.match(r"^\d+\.\s+\*\*(Analyze|Formulate|Interpret|Consider|Evaluate)", text):
                text = ""
            # Strip markdown-formatted internal reasoning blocks
            text = _re.sub(r"\*\s+\*\*(Constraint|Input|Interpretation|Task|Role|Analyze|Formulate).*?(?=\n\n|\Z)", "", text, flags=_re.DOTALL).strip()
            # If after stripping it's mostly reasoning artifacts, clear it
            if text and len(text) > 100:
                non_reasoning = _re.sub(r"\d+\.\s+\*\*.*?\*\*.*?$", "", text, flags=_re.MULTILINE).strip()
                if len(non_reasoning) < len(text) * 0.3:
                    text = ""  # More than 70% was reasoning — discard
            if not text:
                text = "_Summary generation returned empty._"
            return text
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return "_Summary generation failed._"

    # ── Note writing ──────────────────────────────────────────

    def _write_note(self, path: Path, content: str) -> bool:
        """Write a note, preserving human-edited sections if the file exists."""
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            old_content = path.read_text(encoding="utf-8")
            old_blocks = _extract_human_blocks(old_content)
            if old_blocks:
                content = _inject_human_blocks(content, old_blocks)

        path.write_text(content, encoding="utf-8")
        return True

    def _write_competitor_note(
        self,
        config: CompetitorConfig,
        cr: CompetitorResult,
        brands: Dict[str, BrandAggregate],
        categories: Dict[str, CategoryAggregate],
    ) -> int:
        """Write a competitor note from template."""
        if not self.jinja:
            return 0

        # Build brand list for this competitor
        brand_counts = defaultdict(int)
        for p in cr.products:
            b = p.get("brand", "")
            if b and b.lower() not in ("unknown", "n/a", ""):
                brand_counts[b] += 1

        total = len(cr.products) or 1
        comp_brands = sorted(
            [
                {"slug": _slugify(name), "name": name, "count": count, "pct": round(count / total * 100)}
                for name, count in brand_counts.items()
            ],
            key=lambda x: -x["count"],
        )

        # Build category list
        cat_counts = defaultdict(list)
        for p in cr.products:
            cat = _categorize_slug(p.get("category", "other"))
            cat_counts[cat].append(p)

        comp_categories = []
        for cat, products in sorted(cat_counts.items(), key=lambda x: -len(x[1])):
            prices = [p.get("price", 0) for p in products if p.get("price")]
            avg = round(sum(prices) / len(prices), 2) if prices else 0
            comp_categories.append({
                "slug": cat,
                "count": len(products),
                "avg_price": avg,
                "vs_skirmshop": "—",  # would need our pricing data
            })

        # Prices
        all_prices = [p.get("price", 0) for p in cr.products if p.get("price")]

        template = self.jinja.get_template("competitor.md.j2")
        content = template.render(
            domain=config.url.replace("https://", "").replace("http://", "").rstrip("/"),
            url=config.url,
            name=config.name,
            last_crawled=datetime.now(timezone.utc).isoformat(),
            crawl_job_id=cr.crawl_job_id,
            total_products=len(cr.products),
            brands=comp_brands,
            categories=comp_categories,
            price_min=min(all_prices) if all_prices else 0,
            price_max=max(all_prices) if all_prices else 0,
            overview=cr.overview or f"Airsoft retailer with {len(cr.products)} products.",
            seo_notes=cr.seo_notes,
            unmatched_products=cr.unmatched,
            strengths_weaknesses=cr.strengths_weaknesses or "_Analysis pending._",
            pricing_strategy=getattr(cr, 'pricing_strategy', ''),
            opportunity=getattr(cr, 'opportunity', ''),
        )

        path = self.vault_path / "competitors" / f"{config.slug}.md"
        self._write_note(path, content)
        logger.info(f"Wrote competitor note: {path.name}")
        return 1

    def _write_brand_note(self, brand: BrandAggregate) -> int:
        """Write a brand note from template."""
        if not self.jinja:
            return 0

        template = self.jinja.get_template("brand.md.j2")
        content = template.render(
            name=brand.name,
            slug=brand.slug,
            origin=brand.origin,
            categories=brand.categories,
            competitors=brand.competitors,
            our_product_count=brand.our_product_count,
            total_competitor_products=brand.total_competitor_products,
            avg_competitor_price=brand.avg_competitor_price,
            category_breakdown=brand.category_breakdown,
            overview=brand.overview or f"Brand found across {len(brand.competitors)} competitors.",
            opportunity_notes=brand.opportunity_notes,
        )

        path = self.vault_path / "brands" / f"{brand.slug}.md"
        self._write_note(path, content)
        return 1

    def _write_category_note(self, cat: CategoryAggregate) -> int:
        """Write a category note from template."""
        if not self.jinja:
            return 0

        template = self.jinja.get_template("category.md.j2")
        content = template.render(
            name=cat.name,
            slug=cat.slug,
            our_products=cat.our_products,
            our_avg_price=cat.our_avg_price,
            competitor_avg_products=cat.competitor_avg_products,
            avg_price=cat.avg_price,
            brands=cat.brands,
            competitors=cat.competitors,
            brand_breakdown=cat.brand_breakdown or [
                {"slug": b, "tier": "—", "count": 0} for b in cat.brands[:10]
            ],
            market_overview=cat.market_overview or f"Category with {cat.competitor_avg_products} avg products across competitors.",
            translation_notes=cat.translation_notes,
        )

        path = self.vault_path / "categories" / f"{cat.slug}.md"
        self._write_note(path, content)
        return 1

    def _write_market_gap_note(self, all_results: Dict[str, CompetitorResult]) -> int:
        """Write market gap analysis note."""
        if not self.jinja:
            return 0

        all_unmatched = []
        sources = []
        for slug, cr in all_results.items():
            sources.append(slug)
            for p in cr.unmatched:
                p_copy = dict(p) if isinstance(p, dict) else {"name": str(p)}
                p_copy["source"] = slug
                brand = p_copy.get("brand", "Unknown")
                p_copy["brand_slug"] = _slugify(brand)
                p_copy["category_slug"] = _categorize_slug(p_copy.get("category", "other"))
                all_unmatched.append(p_copy)

        if not all_unmatched:
            return 0

        # Group by brand and category
        gaps_by_brand = defaultdict(list)
        gaps_by_category = defaultdict(list)
        for p in all_unmatched:
            gaps_by_brand[p.get("brand", "Unknown")].append(p)
            cat = p.get("category", "other")
            gaps_by_category[cat].append(p)

        top_missing_brands = sorted(gaps_by_brand.keys(), key=lambda b: -len(gaps_by_brand[b]))[:5]

        template = self.jinja.get_template("market-gap.md.j2")
        content = template.render(
            last_updated=datetime.now(timezone.utc).isoformat(),
            gaps=all_unmatched,
            sources=sources,
            top_missing_brands=top_missing_brands,
            gaps_by_brand=dict(gaps_by_brand),
            gaps_by_category=dict(gaps_by_category),
        )

        path = self.vault_path / "market" / "product-gaps.md"
        self._write_note(path, content)
        return 1

    def _write_glossary_gaps_note(self, gaps: List[dict], all_results: Dict[str, CompetitorResult]) -> int:
        """Write glossary gaps note."""
        if not self.jinja or not gaps:
            return 0

        sources = list(all_results.keys())
        high_priority = [g for g in gaps if len(g.get("sources", [])) >= 2]
        medium_priority = [g for g in gaps if len(g.get("sources", [])) < 2]

        template = self.jinja.get_template("glossary-gaps.md.j2")
        content = template.render(
            last_updated=datetime.now(timezone.utc).isoformat(),
            gaps=gaps,
            sources=sources,
            high_priority=high_priority,
            medium_priority=medium_priority[:20],
            recently_added=[],
        )

        path = self.vault_path / "translations" / "glossary-gaps.md"
        self._write_note(path, content)
        return 1

    # ── Meta ──────────────────────────────────────────────────

    def _write_meta(self, result: VaultBuildResult):
        """Write build metadata to _meta/last-build.json."""
        meta_dir = self.vault_path / "_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / "last-build.json"
        meta_path.write_text(
            json.dumps({
                "started_at": result.started_at,
                "completed_at": result.completed_at,
                "competitors_processed": result.competitors_processed,
                "competitors_failed": result.competitors_failed,
                "notes_written": result.notes_written,
                "errors": result.errors,
                "competitor_results": result.competitor_results,
            }, indent=2),
            encoding="utf-8",
        )

    # ── Status ────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get vault build status from meta file."""
        meta_path = self.vault_path / "_meta" / "last-build.json"
        if not meta_path.exists():
            return {"status": "never_built", "vault_path": str(self.vault_path)}

        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        # Count notes
        note_count = 0
        for ext in ("*.md",):
            for d in ("competitors", "brands", "categories", "market", "translations", "strategy",
                       "sources", "guides", "trends", "community"):
                d_path = self.vault_path / d
                if d_path.exists():
                    note_count += len(list(d_path.rglob(ext)))

        # Health check
        last_build = meta.get("completed_at", "")
        healthy = True
        if last_build:
            try:
                last_dt = datetime.fromisoformat(last_build.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - last_dt).days
                healthy = age_days < 8
            except Exception:
                healthy = False

        return {
            "status": "healthy" if healthy else "stale",
            "last_build": meta,
            "note_count": note_count,
            "vault_path": str(self.vault_path),
            "competitors_configured": len(self.competitors),
        }
