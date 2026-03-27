"""
Web Indexer for RAG Pipeline
Crawls websites and indexes content into Qdrant vector database
"""

import hashlib
import logging
import os
import re
import asyncio
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional, Callable
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import json

import httpx
import trafilatura
from bs4 import BeautifulSoup, Tag, NavigableString

try:
    from curl_cffi.requests import AsyncSession as CffiAsyncSession
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False
from qdrant_client.http.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
    ScrollRequest,
    SparseVectorParams, SparseIndexParams,
    Prefetch, FusionQuery, Fusion,
)
from sentence_transformers import SentenceTransformer

from qdrant_utils import make_qdrant_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Indicators of Cloudflare/bot protection in response body
_CF_INDICATORS = [
    "cloudflare", "cf-browser-verification", "cf-challenge",
    "just a moment", "checking your browser", "cf-chl-bypass",
    "ray id", "_cf_chl_opt",
]


@dataclass
class CrawlJob:
    job_id: str
    url: str
    max_depth: int
    max_pages: int
    status: str = "queued"          # queued | running | completed | failed
    pages_found: int = 0
    pages_indexed: int = 0          # consumer: pages whose chunks are embedded+upserted
    pages_scraped: int = 0          # producer: pages fetched+chunked and queued
    pages_visited: int = 0          # producer: pages attempted (including skipped/errored)
    chunks_indexed: int = 0
    chunks_queued: int = 0          # producer: chunks put into queue (not yet indexed)
    current_url: str = ""
    errors: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    eta_seconds: Optional[int] = None
    smart_mode: bool = True
    analysis_status: str = ""         # "" | "analyzing" | "done" | "skipped" | "failed"
    cancel_requested: bool = False    # cooperative cancellation flag
    visited_urls: List[str] = field(default_factory=list)       # saved on stop for resume
    pending_bfs_queue: List[tuple] = field(default_factory=list) # saved on stop for resume
    resumed_from: Optional[str] = None  # parent job_id if resumed
    _last_eta_check: float = 0.0
    _last_eta_visited: int = 0

    MAX_LOGS = 5000

    def log(self, msg: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {msg}")
        if len(self.logs) > self.MAX_LOGS:
            self.logs = self.logs[:100] + self.logs[-(self.MAX_LOGS - 100):]

    def update_eta(self):
        """Recalculate ETA every 2 minutes based on producer scraping rate."""
        import time
        now = time.monotonic()
        if self._last_eta_check == 0.0:
            self._last_eta_check = now
            self._last_eta_visited = self.pages_scraped
            return

        elapsed = now - self._last_eta_check
        if elapsed < 120:  # Recalculate every 2 min
            return

        pages_in_window = self.pages_scraped - self._last_eta_visited
        if pages_in_window <= 0:
            return

        rate = pages_in_window / elapsed  # pages per second
        remaining = self.max_pages - self.pages_scraped
        if remaining <= 0:
            self.eta_seconds = 0
        else:
            self.eta_seconds = int(remaining / rate)

        eta_min = self.eta_seconds // 60
        self.log(f"ETA UPDATE: {rate:.2f} pages/sec, ~{eta_min}m remaining ({remaining} pages left)")
        self._last_eta_check = now
        self._last_eta_visited = self.pages_scraped

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "url": self.url,
            "max_depth": self.max_depth,
            "max_pages": self.max_pages,
            "status": self.status,
            "pages_found": self.pages_found,
            "pages_indexed": self.pages_indexed,
            "pages_scraped": self.pages_scraped,
            "pages_visited": self.pages_visited,
            "chunks_indexed": self.chunks_indexed,
            "chunks_queued": self.chunks_queued,
            "current_url": self.current_url,
            "errors": self.errors[-10:],
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "eta_seconds": self.eta_seconds,
            "analysis_status": self.analysis_status,
            "resumed_from": self.resumed_from,
            "can_resume": self.status == "stopped",
        }


@dataclass
class ExtractionConfig:
    """AI-generated site-specific extraction rules."""
    domain: str
    content_selectors: List[str]
    exclude_selectors: List[str]
    title_selector: Optional[str] = None
    description: str = ""


class WebIndexer:
    COLLECTION_NAME = "web_pages"
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    MIN_CONTENT_LENGTH = 200

    # URL path patterns to exclude from crawling
    _URL_BLACKLIST_PATTERNS = [
        "/go/product/", "/go/category/",   # Lightspeed redirect URLs
        "/tags/", "/brands/",               # Thin listing pages
        "/admin", "/login", "/cart", "/checkout", "/account",
        "/privacy", "/terms", "/sitemap", "/wishlist", "/compare",
        "?page=", "?sort=", "?limit=",     # Pagination/sort variants
    ]

    @classmethod
    def _is_blacklisted_url(cls, url: str) -> bool:
        """Check if URL path matches any blacklist pattern."""
        parsed = urlparse(url)
        path_query = parsed.path + ("?" + parsed.query if parsed.query else "")
        return any(p in path_query for p in cls._URL_BLACKLIST_PATTERNS)

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        model: Optional[SentenceTransformer] = None,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        collection_name: Optional[str] = None,
    ):
        if collection_name:
            self.COLLECTION_NAME = collection_name
        self.client = make_qdrant_client(qdrant_url, qdrant_api_key)
        self._qdrant_url = qdrant_url.rstrip("/")
        self._qdrant_api_key = qdrant_api_key

        # Share model instance if provided (avoids double-loading ~440MB)
        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(embedding_model)

        self.dim = self.model.get_sentence_embedding_dimension()
        self._robots_cache: Dict[str, RobotFileParser] = {}
        self._site_configs: Dict[str, ExtractionConfig] = {}  # domain → cached config
        self._ensure_collection()

    # ── Collection management ──────────────────────────────────────

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.COLLECTION_NAME in collections:
            # Check if already migrated to named vectors (hybrid)
            info = self.client.get_collection(self.COLLECTION_NAME)
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict) and "dense" in vectors_config:
                return  # Already has named vectors
            # Old unnamed-vector collection — delete and recreate
            logger.warning(f"Recreating {self.COLLECTION_NAME} with named vectors for hybrid search")
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

    # ── Robots.txt ─────────────────────────────────────────────────

    async def _get_robots(self, base_url: str, client) -> RobotFileParser:
        parsed = urlparse(base_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        if origin in self._robots_cache:
            return self._robots_cache[origin]

        rp = RobotFileParser()
        robots_url = f"{origin}/robots.txt"
        try:
            resp = await client.get(robots_url, timeout=5.0)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            else:
                rp.parse([])  # No robots.txt — allow all
        except Exception:
            rp.parse([])

        self._robots_cache[origin] = rp
        return rp

    # ── Site probe (Cloudflare detection) ─────────────────────────

    # Suspicious redirect targets that indicate bot blocking
    _BOT_BLOCK_REDIRECT_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0", "::1"}

    async def _probe_site(self, url: str, job: 'CrawlJob') -> bool:
        """Probe site with httpx to detect Cloudflare/bot protection.

        Returns True if curl_cffi should be used instead of httpx.

        Detection strategies:
        1. Bot UA → 403/503 with Cloudflare indicators
        2. Bot UA → 302 redirect to localhost/bogus (common bot trap)
        3. Bot UA → 200 but Cloudflare JS challenge page
        4. Bot UA → tiny/empty body vs browser UA → real content
        """
        if not HAS_CURL_CFFI:
            job.log("PROBE: curl_cffi not installed — skipping Cloudflare detection")
            return False

        job.log(f"PROBE: Testing site accessibility...")
        try:
            # Phase 1: Test with bot UA, NO redirect following
            bot_headers = {
                "User-Agent": self._BOT_UA,
                "Accept": "text/html,application/xhtml+xml",
            }
            async with httpx.AsyncClient(
                follow_redirects=False, timeout=10.0, headers=bot_headers, verify=False,
            ) as client:
                resp = await client.get(url)

                # Check for redirect-based bot blocking (302 → localhost/127.0.0.1)
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location", "")
                    try:
                        redir_host = urlparse(location).hostname or ""
                    except Exception:
                        redir_host = ""

                    if redir_host in self._BOT_BLOCK_REDIRECT_HOSTS:
                        job.log(
                            f"PROBE: Bot-blocking redirect detected (HTTP {resp.status_code} → {location})"
                        )
                        job.log("PROBE: Switching to curl_cffi (browser TLS impersonation)")
                        return True

                    # Redirect to a completely different domain might also be blocking
                    original_host = urlparse(url).hostname or ""
                    if redir_host and redir_host != original_host and not redir_host.endswith(f".{original_host}"):
                        job.log(
                            f"PROBE: Suspicious cross-domain redirect (HTTP {resp.status_code} → {redir_host})"
                        )
                        job.log("PROBE: Switching to curl_cffi (browser TLS impersonation)")
                        return True

                # Bot UA got 403/503 — check for Cloudflare indicators
                if resp.status_code in (403, 503):
                    body_lower = resp.text.lower() if resp.text else ""
                    cf_matches = [ind for ind in _CF_INDICATORS if ind in body_lower]
                    if cf_matches:
                        job.log(
                            f"PROBE: Cloudflare/bot protection detected (HTTP {resp.status_code}, "
                            f"indicators: {', '.join(cf_matches[:3])})"
                        )
                        job.log("PROBE: Switching to curl_cffi (browser TLS impersonation)")
                        return True

                # Bot UA got 200 with content — site doesn't block bots
                if resp.status_code == 200 and len(resp.text or "") > 1000:
                    body_lower = resp.text.lower()
                    if "just a moment" in body_lower and "cloudflare" in body_lower:
                        job.log("PROBE: Cloudflare JS challenge detected (HTTP 200 challenge page)")
                        job.log("PROBE: Switching to curl_cffi (browser TLS impersonation)")
                        return True
                    job.log(f"PROBE: Site accessible with bot UA (HTTP 200, {len(resp.text)} bytes) — no protection detected")
                    return False

            # Phase 2: Bot UA was blocked or got tiny response — verify with browser UA
            browser_headers = {
                "User-Agent": self._BROWSER_UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
            }
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=10.0, headers=browser_headers, verify=False,
            ) as client:
                resp = await client.get(url)
                body_lower = resp.text.lower() if resp.text else ""
                cf_matches = [ind for ind in _CF_INDICATORS if ind in body_lower]

                if resp.status_code in (403, 503) and cf_matches:
                    job.log(
                        f"PROBE: Cloudflare/bot protection detected (HTTP {resp.status_code}, "
                        f"indicators: {', '.join(cf_matches[:3])})"
                    )
                    job.log("PROBE: Switching to curl_cffi (browser TLS impersonation)")
                    return True

                if resp.status_code == 200 and "just a moment" in body_lower and "cloudflare" in body_lower:
                    job.log("PROBE: Cloudflare JS challenge detected (HTTP 200 challenge page)")
                    job.log("PROBE: Switching to curl_cffi (browser TLS impersonation)")
                    return True

                if resp.status_code == 200:
                    job.log(f"PROBE: Site accessible with browser UA (HTTP 200, {len(resp.text)} bytes)")
                    return False

                job.log(f"PROBE: Site returned HTTP {resp.status_code} with browser UA")
                return False

        except Exception as e:
            job.log(f"PROBE: Probe failed ({str(e)[:80]}), will use browser UA with httpx")
            return False

    # ── HTTP client factory ────────────────────────────────────────

    _BROWSER_UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
    _BOT_UA = "DGXSparkRAGBot/1.0 (web indexer for local RAG)"

    @asynccontextmanager
    async def _make_http_client(self, use_curl_cffi: bool):
        """Yield an async HTTP client — curl_cffi (browser impersonation) or httpx.

        Both modes use browser UA to avoid bot-blocking.  curl_cffi adds TLS
        fingerprint impersonation on top, which is needed for Cloudflare etc.
        """
        if use_curl_cffi and HAS_CURL_CFFI:
            headers = {
                "User-Agent": self._BROWSER_UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
            }
            async with CffiAsyncSession(
                impersonate="chrome",
                headers=headers,
                verify=False,
                timeout=15,
                allow_redirects=True,
            ) as session:
                yield session
        else:
            headers = {
                "User-Agent": self._BROWSER_UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
            }
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=15.0,
                headers=headers,
                verify=False,
            ) as client:
                yield client

    # ── Content extraction ─────────────────────────────────────────

    # Common boilerplate patterns to strip from extracted text
    _BOILERPLATE_PATTERNS = [
        # Cookie consent banners — greedy match to catch entire block
        re.compile(r"Si aceptas.*?BLOQUEAR COOKIES", re.DOTALL),
        re.compile(r"If you accept.*?BLOCK COOKIES", re.DOTALL),
        # Cookie banners that start with "Cookies estrictamente"
        re.compile(r"Cookies?\s+estrictamente\s+necesarias.*?BLOQUEAR\s+COOKIES", re.DOTALL | re.IGNORECASE),
        re.compile(r"Cookies?\s+strictly\s+necessary.*?BLOCK\s+COOKIES", re.DOTALL | re.IGNORECASE),
        # Generic cookie/GDPR banners
        re.compile(r"(?:Utilizamos|Usamos|We use)\s+cookies.*?(?:Aceptar|Accept|Rechazar|Reject|cerrar|close)", re.DOTALL | re.IGNORECASE),
        re.compile(r"Este sitio (?:web )?utiliza cookies.*?\n", re.IGNORECASE),
        re.compile(r"This (?:site|website) uses cookies.*?\n", re.IGNORECASE),
        # "Confirmar tus preferencias" block that sometimes remains
        re.compile(r"Confirmar tus preferencias.*?BLOQUEAR\s+COOKIES", re.DOTALL | re.IGNORECASE),
        re.compile(r"Respetamos tu privacidad.*?BLOQUEAR\s+COOKIES", re.DOTALL | re.IGNORECASE),
        # E-commerce / Spanish e-commerce boilerplate
        re.compile(r"A[n\u00f1]adir al carrito.*?(?:\n|$)", re.IGNORECASE),
        re.compile(r"Add to cart.*?(?:\n|$)", re.IGNORECASE),
        re.compile(r"Productos?\s+relacionados?.*", re.DOTALL | re.IGNORECASE),
        re.compile(r"Related products?.*", re.DOTALL | re.IGNORECASE),
        re.compile(r"Lorem ipsum.*?(?:\n\n|\Z)", re.DOTALL | re.IGNORECASE),
    ]

    # CSS selectors for e-commerce boilerplate elements (always removed from HTML)
    _ECOMMERCE_EXCLUDE_SELECTORS = [
        # Navigation & structure
        "nav", ".nav", "#nav", ".menu", "#menu", ".breadcrumb", ".breadcrumbs",
        "header", "footer", ".footer", "#footer", "aside", ".sidebar",
        # Cookie/consent/GDPR
        "[class*='cookie']", "[id*='cookie']",
        "[class*='consent']", "[class*='gdpr']",
        # E-commerce chrome
        ".cart-widget", ".mini-cart", ".wishlist", ".compare",
        ".related-products", ".also-bought", ".upsell", ".cross-sell",
        "[class*='newsletter']", "[class*='popup']", "[class*='modal']",
    ]

    @staticmethod
    def _remove_ecommerce_boilerplate(soup: BeautifulSoup) -> None:
        """Remove hardcoded e-commerce boilerplate elements from soup (mutates in place)."""
        for selector in WebIndexer._ECOMMERCE_EXCLUDE_SELECTORS:
            try:
                for el in soup.select(selector):
                    el.decompose()
            except Exception:
                continue

    @staticmethod
    def _strip_boilerplate(text: str) -> str:
        """Remove common boilerplate patterns (cookie banners, etc.)."""
        for pattern in WebIndexer._BOILERPLATE_PATTERNS:
            text = pattern.sub("", text)
        # Collapse resulting whitespace
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    # ── Smart extraction (AI-driven) ─────────────────────────────

    @staticmethod
    def _trim_html_for_llm(html: str, max_chars: int = 3000) -> str:
        """Reduce HTML to a structural skeleton for LLM analysis.

        Keeps class/id attributes for structure, truncates text nodes,
        removes scripts/styles/SVG. Target: ~3K chars per page.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove noise elements entirely
        for tag in soup(["script", "style", "svg", "noscript", "iframe", "link", "meta"]):
            tag.decompose()

        # Truncate text nodes to 80 chars
        for element in soup.descendants:
            if isinstance(element, NavigableString) and not isinstance(element, Tag):
                text = element.strip()
                if len(text) > 80:
                    element.replace_with(text[:80] + "...")

        # Strip attributes except structural ones
        keep_attrs = {"class", "id", "role"}
        for tag in soup.find_all(True):
            attrs_to_remove = [a for a in tag.attrs if a not in keep_attrs and not a.startswith("data-")]
            for attr in attrs_to_remove:
                del tag[attr]

        result = soup.prettify()
        if len(result) > max_chars:
            result = result[:max_chars] + "\n<!-- truncated -->"
        return result

    async def _fetch_sample_pages(
        self,
        start_url: str,
        max_depth: int,
        client,
        robots: RobotFileParser,
    ) -> List[Dict[str, str]]:
        """Fetch ~10 diverse sample pages via BFS for site analysis."""
        visited = set()
        bfs_queue = deque([(start_url, 0)])
        samples = []
        sample_depth = min(max_depth, 2)  # limit sampling depth
        # Track path patterns for diversity
        path_patterns: List[str] = []

        while bfs_queue and len(samples) < 10:
            url, depth = bfs_queue.popleft()
            url = url.rstrip("/")

            if url in visited:
                continue

            if not robots.can_fetch("*", url):
                visited.add(url)
                continue

            # Score diversity: prefer URLs with different path structures
            parsed_url = urlparse(url)
            path_parts = [p for p in parsed_url.path.split("/") if p]
            path_sig = "/".join(path_parts[:2]) if path_parts else "/"

            # Skip if we already have 3+ samples with the same path pattern
            same_pattern_count = sum(1 for p in path_patterns if p == path_sig)
            if same_pattern_count >= 3 and len(samples) > 3:
                visited.add(url)
                continue

            await asyncio.sleep(0.5)  # faster rate for sampling

            try:
                resp = await client.get(url, timeout=10.0)
                if resp.status_code != 200:
                    visited.add(url)
                    continue
                content_type = resp.headers.get("content-type", "")
                if "text/html" not in content_type:
                    visited.add(url)
                    continue

                html = resp.text
                samples.append({"url": url, "html": html})
                path_patterns.append(path_sig)
                visited.add(url)

                # Extract links for BFS
                if depth < sample_depth:
                    links = self._extract_links(html, url)
                    for link in links:
                        link_clean = link.rstrip("/")
                        if link_clean not in visited:
                            bfs_queue.append((link_clean, depth + 1))
            except Exception:
                visited.add(url)
                continue

        return samples

    @staticmethod
    def _parse_llm_config(response_text: str, domain: str) -> Optional[ExtractionConfig]:
        """Parse LLM response into ExtractionConfig with validation."""
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
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    return None
            else:
                return None

        # Validate required fields
        content_sels = data.get("content_selectors", [])
        exclude_sels = data.get("exclude_selectors", [])

        if not isinstance(content_sels, list) or not content_sels:
            return None
        if not isinstance(exclude_sels, list) or not exclude_sels:
            return None

        # Validate selectors are strings and valid CSS
        valid_content = []
        for sel in content_sels:
            if not isinstance(sel, str) or not sel.strip():
                continue
            try:
                BeautifulSoup("<div></div>", "html.parser").select(sel.strip())
                valid_content.append(sel.strip())
            except Exception:
                continue

        valid_exclude = []
        for sel in exclude_sels:
            if not isinstance(sel, str) or not sel.strip():
                continue
            try:
                BeautifulSoup("<div></div>", "html.parser").select(sel.strip())
                valid_exclude.append(sel.strip())
            except Exception:
                continue

        if not valid_content:
            return None

        title_sel = data.get("title_selector")
        if title_sel:
            try:
                BeautifulSoup("<div></div>", "html.parser").select(title_sel)
            except Exception:
                title_sel = None

        return ExtractionConfig(
            domain=domain,
            content_selectors=valid_content,
            exclude_selectors=valid_exclude,
            title_selector=title_sel if isinstance(title_sel, str) else None,
            description=str(data.get("description", ""))[:200],
        )

    async def _analyze_site(
        self,
        samples: List[Dict[str, str]],
        llm_client,
        job: 'CrawlJob',
    ) -> Optional[ExtractionConfig]:
        """Call LLM to analyze site structure and return extraction config."""
        if not samples:
            return None

        domain = urlparse(samples[0]["url"]).netloc

        # Build prompt with trimmed HTML samples
        sample_sections = []
        for i, sample in enumerate(samples, 1):
            trimmed = self._trim_html_for_llm(sample["html"])
            sample_sections.append(f"--- Page {i}: {sample['url']} ---\n{trimmed}")

        samples_text = "\n\n".join(sample_sections)

        system_prompt = (
            "You are a web scraping assistant. Analyze HTML samples from a website "
            "and return JSON with CSS selectors for content extraction."
        )

        user_prompt = f"""I'm building a web scraper for this site. Here are {len(samples)} sample pages (trimmed HTML):

{samples_text}

Analyze the HTML structure and return a JSON object with:
{{
  "content_selectors": ["CSS selectors that capture the MAIN content (articles, product details, listings, text bodies)"],
  "exclude_selectors": ["CSS selectors for elements to REMOVE before extraction (navigation, cookie banners, sidebars, footers, popups, ads)"],
  "title_selector": "CSS selector for page title (or null if <title> tag is good enough)",
  "description": "Brief description of this website type and content"
}}

Rules:
- content_selectors should be SPECIFIC enough to target real content, not generic ("div" is too broad)
- exclude_selectors should catch all boilerplate without removing content
- Use class-based selectors when available (more reliable than tag-only)
- Include at least 2-3 content_selectors covering different page types you see
- Return ONLY the JSON object, no explanation"""

        try:
            model_id = os.environ.get("CRAWL_MODEL", "local")
            job.log(f"SMART: Calling LLM ({model_id}) with {len(samples)} page samples...")

            # Run sync LLM call in thread pool to avoid blocking the event loop.
            # Timeout set high (5 min) because thinking models (Qwen3-32B) can take
            # 2-3 minutes to reason through large HTML analysis prompts.
            response = await loop.run_in_executor(
                None,
                lambda: llm_client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=2048,
                    temperature=0.1,
                    timeout=300,
                    user="rag:smart-crawl",
                ),
            )

            response_text = response.choices[0].message.content
            job.log(f"SMART: LLM response received ({len(response_text)} chars)")

            config = self._parse_llm_config(response_text, domain)
            if config:
                job.log(
                    f"SMART: Parsed {len(config.content_selectors)} content selectors, "
                    f"{len(config.exclude_selectors)} exclude selectors"
                )
                if config.description:
                    job.log(f"SMART: Site description: {config.description}")
            else:
                job.log(f"SMART: Failed to parse LLM response")
                logger.warning(f"Failed to parse LLM config for {domain}: {response_text[:200]}")

            return config

        except Exception as e:
            job.log(f"SMART: LLM analysis failed: {str(e)[:120]}")
            logger.warning(f"LLM analysis failed for {domain}: {e}")
            return None

    @staticmethod
    def _extract_content_smart(html: str, url: str, config: ExtractionConfig) -> Dict[str, str]:
        """Extract content using AI-generated CSS selectors, fallback to standard."""
        title = ""

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Remove hardcoded e-commerce boilerplate before AI selectors
            WebIndexer._remove_ecommerce_boilerplate(soup)

            # Get title
            if config.title_selector:
                title_el = soup.select_one(config.title_selector)
                if title_el:
                    title = title_el.get_text(strip=True)
            if not title:
                title_tag = soup.find("title")
                if title_tag:
                    title = title_tag.get_text(strip=True)

            # Remove excluded elements
            for selector in config.exclude_selectors:
                try:
                    for el in soup.select(selector):
                        el.decompose()
                except Exception:
                    continue

            # Extract content from content selectors
            content_parts = []
            for selector in config.content_selectors:
                try:
                    for el in soup.select(selector):
                        text = el.get_text(separator="\n", strip=True)
                        if text and len(text) > 20:
                            content_parts.append(text)
                except Exception:
                    continue

            text = "\n\n".join(content_parts)

            # Apply boilerplate stripping as defense in depth
            if text:
                text = WebIndexer._strip_boilerplate(text)

            # If smart extraction yielded too little, fall back to standard
            if not text or len(text) < WebIndexer.MIN_CONTENT_LENGTH:
                return WebIndexer._extract_content(html, url)

            # Clean up whitespace
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            return {"title": title, "text": text, "url": url}

        except Exception:
            # Any error in smart extraction → fall back to standard
            return WebIndexer._extract_content(html, url)

    @staticmethod
    def _extract_content(html: str, url: str) -> Dict[str, str]:
        """Extract clean text from HTML using trafilatura, fallback to BS4."""
        title = ""

        # Try to get title from HTML
        try:
            soup = BeautifulSoup(html, "html.parser")
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)
        except Exception:
            pass

        # Primary: trafilatura for content extraction
        text = trafilatura.extract(html, include_links=False, include_tables=True)

        # Strip boilerplate from trafilatura output
        if text:
            text = WebIndexer._strip_boilerplate(text)

        # Fallback: BeautifulSoup if trafilatura returned nothing useful
        if not text or len(text) < WebIndexer.MIN_CONTENT_LENGTH:
            try:
                soup = BeautifulSoup(html, "html.parser")
                # Remove script/style/nav/cookie elements
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                # Remove common cookie/consent containers by class/id
                for sel in soup.select('[class*="cookie"], [class*="consent"], [id*="cookie"], [id*="consent"], [class*="gdpr"], [id*="gdpr"]'):
                    sel.decompose()
                # Remove e-commerce boilerplate elements
                WebIndexer._remove_ecommerce_boilerplate(soup)
                text = soup.get_text(separator="\n", strip=True)
                if text:
                    text = WebIndexer._strip_boilerplate(text)
            except Exception:
                text = ""

        # Clean up whitespace
        if text:
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()

        return {"title": title, "text": text, "url": url}

    # ── JSON-LD structured data extraction ─────────────────────────

    @staticmethod
    def _extract_jsonld(html: str) -> Optional[Dict]:
        """Extract Product structured data from JSON-LD script tags.

        Returns flattened dict with product_name, product_price, etc.
        Returns None if no Product schema found.
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            # Collect all items from all JSON-LD blocks
            all_items: List[dict] = []
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(script.string, strict=False)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(data, dict):
                    if "@graph" in data:
                        all_items.extend(data["@graph"])
                    else:
                        all_items.append(data)
                elif isinstance(data, list):
                    all_items.extend(data)

            result: Dict[str, str] = {}

            # Extract BreadcrumbList → category_path
            for item in all_items:
                if not isinstance(item, dict):
                    continue
                if item.get("@type") == "BreadcrumbList":
                    breadcrumbs = item.get("itemListElement", [])
                    if breadcrumbs:
                        path = " > ".join(
                            str(bc.get("item", {}).get("name", "") if isinstance(bc.get("item"), dict) else bc.get("name", ""))
                            for bc in sorted(breadcrumbs, key=lambda x: x.get("position", 0))
                            if (bc.get("item", {}).get("name") if isinstance(bc.get("item"), dict) else bc.get("name"))
                        )
                        if path:
                            result["category_path"] = path
                    break

            # Extract Product schema
            for item in all_items:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("@type", "")
                type_list = item_type if isinstance(item_type, list) else [item_type]
                if "Product" not in type_list:
                    continue

                offers = item.get("offers", {})
                if isinstance(offers, list):
                    offers = offers[0] if offers else {}
                if not isinstance(offers, dict):
                    offers = {}

                if item.get("name"):
                    result["product_name"] = str(item["name"])[:200]
                if item.get("sku"):
                    result["product_sku"] = str(item["sku"])
                ean = item.get("gtin13") or item.get("gtin14") or item.get("gtin")
                if ean:
                    result["product_ean"] = str(ean)
                brand = item.get("brand")
                if isinstance(brand, dict):
                    result["product_brand"] = str(brand.get("name", ""))[:100]
                elif brand:
                    result["product_brand"] = str(brand)[:100]
                if item.get("description"):
                    result["product_description"] = str(item["description"])[:500]
                img = item.get("image")
                if img:
                    result["product_image"] = str(img[0] if isinstance(img, list) else img)[:500]

                if offers.get("price"):
                    result["product_price"] = str(offers["price"])
                if offers.get("priceCurrency"):
                    result["product_currency"] = str(offers["priceCurrency"])
                if offers.get("availability"):
                    avail = str(offers["availability"])
                    if "InStock" in avail:
                        result["product_availability"] = "InStock"
                    elif "OutOfStock" in avail:
                        result["product_availability"] = "OutOfStock"
                    else:
                        result["product_availability"] = avail.split("/")[-1]
                break

            return result if result else None
        except Exception:
            pass
        return None

    # ── Link extraction ────────────────────────────────────────────

    @staticmethod
    def _extract_links(html: str, base_url: str) -> List[str]:
        """Extract same-domain links from HTML."""
        base_parsed = urlparse(base_url)
        base_domain = base_parsed.netloc
        links = []

        try:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()

                # Skip anchors, javascript, mailto
                if href.startswith(("#", "javascript:", "mailto:", "tel:")):
                    continue

                full_url = urljoin(base_url, href)
                parsed = urlparse(full_url)

                # Same domain only
                if parsed.netloc != base_domain:
                    continue

                # Clean URL: remove fragments, normalize
                clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if parsed.query:
                    clean += f"?{parsed.query}"

                # Skip common non-content URLs
                skip_exts = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg",
                             ".zip", ".tar", ".gz", ".mp4", ".mp3", ".webp"}
                if any(clean.lower().endswith(ext) for ext in skip_exts):
                    continue

                # Skip blacklisted URL patterns
                if WebIndexer._is_blacklisted_url(clean):
                    continue

                links.append(clean)
        except Exception:
            pass

        return list(set(links))

    # ── Chunking ───────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[Dict]:
        """Paragraph-aware text chunking."""
        if not text:
            return []

        # Split into paragraphs first
        paragraphs = re.split(r"\n\n+", text)
        chunks = []
        current = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)

            if current_len + para_len + 1 > max_chars and current:
                chunk_text = "\n\n".join(current)
                chunks.append({"text": chunk_text, "chunk_idx": len(chunks)})

                # Overlap: keep last paragraph if it fits
                overlap_parts = []
                overlap_len = 0
                for p in reversed(current):
                    if overlap_len + len(p) > overlap:
                        break
                    overlap_parts.insert(0, p)
                    overlap_len += len(p) + 2

                current = overlap_parts
                current_len = overlap_len

            current.append(para)
            current_len += para_len + 2

        if current:
            chunk_text = "\n\n".join(current)
            if chunk_text.strip():
                chunks.append({"text": chunk_text, "chunk_idx": len(chunks)})

        return chunks

    # ── ID generation ──────────────────────────────────────────────

    @staticmethod
    def _generate_id(url: str, chunk_idx: int) -> int:
        key = f"web:{url}:{chunk_idx}"
        h = hashlib.sha256(key.encode()).hexdigest()
        return int(h[:16], 16)

    # ── Crawl & Index (producer-consumer pipeline) ─────────────────

    async def crawl_and_index(
        self,
        start_url: str,
        max_depth: int = 2,
        max_pages: int = 50,
        progress_callback: Optional[Callable] = None,
        smart_mode: bool = True,
        llm_client=None,
        initial_visited: Optional[set] = None,
        initial_bfs_queue=None,
    ) -> CrawlJob:
        """BFS crawl with async producer-consumer pipeline.

        Producer: fetch pages → extract → chunk → put into asyncio.Queue
        Consumer: batch embed (50 chunks) → batch upsert (200 points) to Qdrant
        Neither blocks the other — connected via asyncio.Queue with backpressure.

        If smart_mode=True: samples pages first, calls LLM for site-specific
        CSS selectors, then uses them for targeted content extraction.
        """
        import uuid

        job = CrawlJob(
            job_id=uuid.uuid4().hex[:12],
            url=start_url,
            max_depth=max_depth,
            max_pages=max_pages,
            status="running",
            smart_mode=smart_mode,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        if progress_callback:
            progress_callback(job)

        # ── Site probe: detect Cloudflare/bot protection ──
        use_curl_cffi = await self._probe_site(start_url, job)
        if progress_callback:
            progress_callback(job)

        # ── Smart Mode: AI-driven site analysis ──
        extraction_config: Optional[ExtractionConfig] = None
        domain = urlparse(start_url).netloc

        if smart_mode and llm_client:
            # Check cache first
            if domain in self._site_configs:
                extraction_config = self._site_configs[domain]
                job.analysis_status = "skipped"
                job.log(f"SMART: Using cached config for {domain} ({len(extraction_config.content_selectors)} selectors)")
            else:
                job.analysis_status = "analyzing"
                job.log("SMART: Analyzing site structure (sampling up to 10 pages)...")
                if progress_callback:
                    progress_callback(job)

                try:
                    async with self._make_http_client(use_curl_cffi) as sample_client:
                        robots = await self._get_robots(start_url, sample_client)
                        samples = await self._fetch_sample_pages(start_url, max_depth, sample_client, robots)

                    job.log(f"SMART: Fetched {len(samples)} sample pages, calling LLM...")
                    if progress_callback:
                        progress_callback(job)

                    extraction_config = await self._analyze_site(samples, llm_client, job)

                    if extraction_config:
                        self._site_configs[domain] = extraction_config
                        job.analysis_status = "done"
                        job.log(f"SMART: Analysis complete — {len(extraction_config.content_selectors)} content selectors, {len(extraction_config.exclude_selectors)} exclude selectors")
                    else:
                        job.analysis_status = "failed"
                        job.log("SMART: Analysis failed (no extraction config). Falling back to standard extraction.")
                        job.errors.append("Smart mode analysis failed — falling back to standard extraction")
                except Exception as e:
                    job.analysis_status = "failed"
                    job.log(f"SMART: Analysis error: {str(e)[:120]}. Falling back to standard extraction.")
                    job.errors.append(f"Smart mode error: {str(e)[:200]} — falling back to standard extraction")
                    logger.warning(f"Smart analysis failed for {start_url}: {e}")

                if progress_callback:
                    progress_callback(job)
        elif smart_mode and not llm_client:
            job.analysis_status = "skipped"
            job.log("SMART: No LLM client available, using standard extraction")

        # Queue with backpressure: producer blocks if consumer falls behind
        # 500 chunks × ~1.5 KB each = ~750 KB max memory
        chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

        try:
            producer_task = asyncio.create_task(
                self._producer(start_url, max_depth, max_pages, chunk_queue, job, progress_callback, extraction_config, use_curl_cffi, initial_visited, initial_bfs_queue)
            )
            consumer_task = asyncio.create_task(
                self._consumer(chunk_queue, job, progress_callback)
            )
            await asyncio.gather(producer_task, consumer_task)

            if job.cancel_requested:
                job.status = "stopped"
                job.log(f"STOPPED: {job.pages_indexed} pages indexed, {job.chunks_indexed} chunks, {job.pages_visited} pages visited (resumable)")
            else:
                job.status = "completed"
                job.log(f"COMPLETED: {job.pages_indexed} pages indexed, {job.chunks_indexed} chunks, {job.pages_visited} pages visited")

        except Exception as e:
            job.status = "failed"
            job.errors.append(f"Crawl failed: {str(e)[:200]}")
            job.log(f"FAILED: {str(e)[:200]}")
            logger.error(f"Crawl failed for {start_url}: {e}")

        job.ended_at = datetime.now(timezone.utc).isoformat()

        if progress_callback:
            progress_callback(job)

        logger.info(
            f"Crawl {'completed' if job.status == 'completed' else 'failed'}: "
            f"{start_url} — {job.pages_indexed} pages, {job.chunks_indexed} chunks"
        )

        return job

    # ── Producer: fetch → extract → chunk → queue ────────────────

    async def _producer(
        self,
        start_url: str,
        max_depth: int,
        max_pages: int,
        chunk_queue: asyncio.Queue,
        job: CrawlJob,
        progress_callback: Optional[Callable] = None,
        extraction_config: Optional[ExtractionConfig] = None,
        use_curl_cffi: bool = False,
        initial_visited: Optional[set] = None,
        initial_bfs_queue=None,
    ):
        """BFS crawl: fetch pages, extract content, chunk text, put into queue."""
        visited = initial_visited if initial_visited is not None else set()
        bfs_queue = initial_bfs_queue if initial_bfs_queue is not None else deque([(start_url, 0)])
        if initial_visited:
            job.pages_visited = len(visited)
        # Content dedup: track text fingerprints to skip repeated boilerplate
        content_fingerprints: Dict[str, int] = {}  # hash → count

        try:
            async with self._make_http_client(use_curl_cffi) as client:

                robots = await self._get_robots(start_url, client)
                job.log(f"Starting crawl: {start_url} (depth={max_depth}, max_pages={max_pages})")

                while bfs_queue and len(visited) < max_pages:
                    # Check for cancellation
                    if job.cancel_requested:
                        job.log("PRODUCER: Stop requested, halting crawl...")
                        job.visited_urls = list(visited)
                        job.pending_bfs_queue = list(bfs_queue)
                        break

                    url, depth = bfs_queue.popleft()
                    url = url.rstrip("/")

                    if url in visited:
                        continue

                    if self._is_blacklisted_url(url):
                        visited.add(url)
                        continue

                    if not robots.can_fetch("*", url):
                        job.log(f"BLOCKED robots.txt: {url}")
                        continue

                    job.current_url = url
                    job.pages_found = len(visited) + len(bfs_queue) + 1

                    if progress_callback:
                        progress_callback(job)

                    # Rate limit (1s per page for politeness)
                    await asyncio.sleep(1.0)

                    # Fetch page
                    try:
                        resp = await client.get(url)
                        if resp.status_code != 200:
                            job.log(f"HTTP {resp.status_code}: {url}")
                            job.errors.append(f"HTTP {resp.status_code}: {url}")
                            visited.add(url)
                            job.pages_visited = len(visited)
                            job.update_eta()
                            continue

                        content_type = resp.headers.get("content-type", "")
                        if "text/html" not in content_type:
                            job.log(f"SKIP non-HTML ({content_type[:30]}): {url}")
                            visited.add(url)
                            job.pages_visited = len(visited)
                            job.update_eta()
                            continue

                        html = resp.text

                        # Resolve canonical URL after redirects
                        canonical_url = str(resp.url).rstrip("/")
                        if canonical_url != url:
                            if canonical_url in visited:
                                job.log(f"SKIP redirect duplicate: {url} -> {canonical_url}")
                                continue
                            job.log(f"CANONICAL: {url} -> {canonical_url}")
                            url = canonical_url
                    except Exception as e:
                        job.log(f"FETCH ERROR: {url} — {str(e)[:80]}")
                        job.errors.append(f"Fetch error: {url}: {str(e)[:100]}")
                        visited.add(url)
                        job.pages_visited = len(visited)
                        job.update_eta()
                        continue

                    visited.add(url)
                    job.pages_visited = len(visited)
                    job.update_eta()

                    # Extract and queue same-domain links BEFORE content check
                    # (even thin pages have useful links for BFS discovery)
                    if depth < max_depth:
                        links = self._extract_links(html, url)
                        new_links = [l.rstrip("/") for l in links if l.rstrip("/") not in visited]
                        for link in new_links:
                            bfs_queue.append((link, depth + 1))
                        if new_links:
                            job.log(f"FOUND {len(new_links)} new links (depth {depth+1}), queue: {len(bfs_queue)}")

                    # Extract content (smart or standard)
                    if extraction_config:
                        content = self._extract_content_smart(html, url, extraction_config)
                    else:
                        content = self._extract_content(html, url)
                    if not content["text"] or len(content["text"]) < self.MIN_CONTENT_LENGTH:
                        job.log(f"SKIP low content ({len(content['text'] or '')} chars): {url}")
                        continue

                    # Content dedup: skip pages whose text we've seen 3+ times
                    text_hash = hashlib.md5(content["text"].encode()).hexdigest()
                    content_fingerprints[text_hash] = content_fingerprints.get(text_hash, 0) + 1
                    if content_fingerprints[text_hash] > 3:
                        job.log(f"SKIP duplicate content (seen {content_fingerprints[text_hash]}x): {url}")
                        continue

                    # Extract JSON-LD structured data (product info)
                    jsonld_data = self._extract_jsonld(html)

                    # Chunk and queue (consumer will batch-embed)
                    domain = urlparse(url).netloc
                    chunks = self._chunk_text(content["text"])
                    now = datetime.now(timezone.utc).isoformat()

                    # Dedup individual chunks across pages
                    queued_chunks = 0
                    for chunk in chunks:
                        chunk_hash = hashlib.md5(chunk["text"].encode()).hexdigest()
                        chunk_count = content_fingerprints.get(f"c:{chunk_hash}", 0) + 1
                        content_fingerprints[f"c:{chunk_hash}"] = chunk_count
                        if chunk_count > 3:
                            continue  # skip repeated chunk
                        await chunk_queue.put({
                            "text": chunk["text"],
                            "chunk_idx": chunk["chunk_idx"],
                            "url": url,
                            "title": content["title"],
                            "domain": domain,
                            "fetch_date": now,
                            "jsonld_data": jsonld_data,
                        })
                        queued_chunks += 1

                    job.pages_scraped += 1
                    job.chunks_queued += queued_chunks
                    title_short = (content["title"][:60] + "...") if len(content["title"]) > 60 else content["title"]
                    dedup_note = f" ({len(chunks) - queued_chunks} deduped)" if queued_chunks < len(chunks) else ""
                    job.log(f"SCRAPED [{job.pages_scraped}/{max_pages}] {queued_chunks} chunks{dedup_note} — {title_short or url}")

                    if progress_callback:
                        progress_callback(job)

            if job.cancel_requested:
                job.log(f"PRODUCER STOPPED: {job.pages_scraped} pages scraped, {job.chunks_queued} chunks queued (state saved for resume)")
            else:
                job.log(f"PRODUCER DONE: {job.pages_scraped} pages scraped, {job.chunks_queued} chunks queued")

        finally:
            # Always send sentinel so consumer doesn't hang
            await chunk_queue.put(None)

    # ── Consumer: batch embed → batch upsert ─────────────────────

    async def _consumer(
        self,
        chunk_queue: asyncio.Queue,
        job: CrawlJob,
        progress_callback: Optional[Callable] = None,
    ):
        """Consume chunks from queue: batch embed, batch upsert to Qdrant."""
        EMBED_BATCH_SIZE = 50
        UPSERT_BATCH_SIZE = 200

        batch_chunks = []
        batch_points = []

        while True:
            item = await chunk_queue.get()

            if item is None:
                break

            batch_chunks.append(item)

            if len(batch_chunks) >= EMBED_BATCH_SIZE:
                points = await self._embed_and_build_points(batch_chunks)
                batch_points.extend(points)
                batch_chunks = []

                if len(batch_points) >= UPSERT_BATCH_SIZE:
                    await self._async_upsert(batch_points, job)
                    job.pages_indexed = job.pages_scraped
                    batch_points = []
                    if progress_callback:
                        progress_callback(job)

        # Drain remaining after sentinel
        if batch_chunks:
            points = await self._embed_and_build_points(batch_chunks)
            batch_points.extend(points)

        if batch_points:
            await self._async_upsert(batch_points, job)

        job.pages_indexed = job.pages_scraped
        job.log(f"CONSUMER DONE: {job.chunks_indexed} total chunks indexed to Qdrant")

        if progress_callback:
            progress_callback(job)

    # ── Batch embed helper ───────────────────────────────────────

    async def _embed_and_build_points(self, chunk_items: list) -> list:
        """Batch-encode chunks (dense + sparse) and return PointStruct list.

        Uses run_in_executor to keep event loop responsive during CPU-bound
        embedding (allows producer to continue fetching + SSE to stream).
        """
        from sparse_encoder import encode_sparse

        texts = [item["text"] for item in chunk_items]

        loop = asyncio.get_event_loop()
        # Dense embeddings (semantic)
        dense_embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=len(texts),
            )
        )
        # Sparse embeddings (BM25 keyword)
        sparse_embeddings = await loop.run_in_executor(
            None,
            lambda: encode_sparse(texts)
        )

        points = []
        for i, item in enumerate(chunk_items):
            payload = {
                "url": item["url"],
                "title": item["title"],
                "domain": item["domain"],
                "text": item["text"],
                "chunk_idx": item["chunk_idx"],
                "fetch_date": item["fetch_date"],
                "source_type": "web",
            }
            # Merge JSON-LD product data into payload (flattened)
            if item.get("jsonld_data"):
                payload.update(item["jsonld_data"])

            points.append(PointStruct(
                id=self._generate_id(item["url"], item["chunk_idx"]),
                vector={
                    "dense": dense_embeddings[i].tolist(),
                    "sparse": sparse_embeddings[i],
                },
                payload=payload,
            ))

        return points

    # ── Async Qdrant upsert helper ───────────────────────────────

    async def _async_upsert(self, points: list, job: CrawlJob):
        """Upsert points to Qdrant in thread pool (non-blocking)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points,
            )
        )
        job.chunks_indexed += len(points)
        job.log(f"UPSERTED batch of {len(points)} chunks to Qdrant (total: {job.chunks_indexed})")

    # ── Sources ────────────────────────────────────────────────────

    def get_sources(self) -> List[Dict]:
        """List all indexed web sources grouped by domain."""
        try:
            # Check if collection exists
            collections = [c.name for c in self.client.get_collections().collections]
            if self.COLLECTION_NAME not in collections:
                return []

            # Scroll all points and group by domain
            domains: Dict[str, Dict] = {}
            offset = None
            while True:
                results = self.client.scroll(
                    collection_name=self.COLLECTION_NAME,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points, next_offset = results

                for point in points:
                    payload = point.payload or {}
                    domain = payload.get("domain", "unknown")
                    if domain not in domains:
                        domains[domain] = {
                            "domain": domain,
                            "url_count": 0,
                            "chunk_count": 0,
                            "urls": set(),
                            "last_indexed": payload.get("fetch_date", ""),
                        }
                    domains[domain]["chunk_count"] += 1
                    url = payload.get("url", "")
                    if url:
                        domains[domain]["urls"].add(url)
                    fetch_date = payload.get("fetch_date", "")
                    if fetch_date > domains[domain]["last_indexed"]:
                        domains[domain]["last_indexed"] = fetch_date

                if next_offset is None:
                    break
                offset = next_offset

            # Finalize
            sources = []
            for d in domains.values():
                d["url_count"] = len(d["urls"])
                del d["urls"]
                sources.append(d)

            return sorted(sources, key=lambda x: x["domain"])

        except Exception as e:
            logger.error(f"Error getting sources: {e}")
            return []

    # ── Delete ─────────────────────────────────────────────────────

    def delete_source(self, domain: str) -> bool:
        """Delete all indexed content from a domain."""
        try:
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="domain",
                            match=MatchValue(value=domain),
                        )
                    ]
                ),
            )
            logger.info(f"Deleted all content for domain: {domain}")
            return True
        except Exception as e:
            logger.error(f"Error deleting domain {domain}: {e}")
            return False

    # ── Search ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
    ) -> List[Dict]:
        """Hybrid search (dense + sparse BM25) on web_pages collection."""
        from sparse_encoder import encode_sparse_query

        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.COLLECTION_NAME not in collections:
                return []

            # Dense embedding with BGE query prefix
            prefixed_query = f"{self.BGE_QUERY_PREFIX}{query}"
            dense_embedding = self.model.encode(
                prefixed_query,
                normalize_embeddings=True,
            ).tolist()

            # Sparse embedding (BM25)
            sparse_embedding = encode_sparse_query(query)

            search_filter = None
            if domain_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="domain",
                            match=MatchValue(value=domain_filter),
                        )
                    ]
                )

            # Hybrid query with RRF fusion
            results = self.client.query_points(
                collection_name=self.COLLECTION_NAME,
                prefetch=[
                    Prefetch(
                        query=dense_embedding,
                        using="dense",
                        filter=search_filter,
                        limit=top_k * 3,
                    ),
                    Prefetch(
                        query=sparse_embedding,
                        using="sparse",
                        filter=search_filter,
                        limit=top_k * 3,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )

            _PRODUCT_KEYS = (
                "product_name", "product_price", "product_currency", "product_sku",
                "product_ean", "product_brand", "product_image", "product_availability",
                "product_description", "category_path",
            )
            out = []
            for r in results.points:
                item = {
                    "url": r.payload.get("url", ""),
                    "title": r.payload.get("title", ""),
                    "domain": r.payload.get("domain", ""),
                    "text": r.payload.get("text", ""),
                    "chunk_idx": r.payload.get("chunk_idx", 0),
                    "fetch_date": r.payload.get("fetch_date", ""),
                    "score": round(r.score, 4),
                    "source_type": "web",
                }
                for key in _PRODUCT_KEYS:
                    val = r.payload.get(key)
                    if val:
                        item[key] = val
                out.append(item)
            return out

        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return []

    # ── Collection stats ───────────────────────────────────────────

    def get_collection_stats(self, collection_names: Optional[List[str]] = None) -> List[Dict]:
        """Get stats for all RAG collections via REST API."""
        import httpx as _httpx

        if collection_names is None:
            collection_names = [
                "web_pages", "product_catalog",
                "competitor_products",
            ]

        stats = []
        headers = {}
        if self._qdrant_api_key:
            headers["api-key"] = self._qdrant_api_key

        for name in collection_names:
            try:
                resp = _httpx.get(
                    f"{self._qdrant_url}/collections/{name}",
                    headers=headers,
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    data = resp.json().get("result", {})
                    stats.append({
                        "name": name,
                        "points_count": data.get("points_count", 0),
                        "vectors_count": data.get("indexed_vectors_count", 0),
                        "status": data.get("status", "unknown"),
                    })
                else:
                    stats.append({
                        "name": name,
                        "points_count": 0,
                        "vectors_count": 0,
                        "status": "not_found",
                    })
            except Exception as e:
                logger.warning(f"Failed to get stats for {name}: {e}")
                stats.append({
                    "name": name,
                    "points_count": 0,
                    "vectors_count": 0,
                    "status": "error",
                })
        return stats
