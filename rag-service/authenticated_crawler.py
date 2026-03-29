"""
Authenticated Crawler for Supplier B2B Portals
Handles login flows and cookie-based crawling for sites requiring authentication.
Uses curl_cffi for login (browser TLS fingerprint) and Firecrawl for page scraping.
"""

import asyncio
import json
import logging
import re
import subprocess
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlencode

try:
    from curl_cffi.requests import AsyncSession as CffiAsyncSession
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _get_1password_password(item_title: str, vault: str = "Skirmshop") -> Optional[str]:
    """Fetch password from 1Password CLI."""
    try:
        result = subprocess.run(
            ["op", "item", "get", item_title, "--vault", vault,
             "--fields", "password", "--reveal"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logger.error(f"1Password fetch failed for {item_title}: {e}")
    return None


class AuthenticatedCrawler:
    """Handles authenticated crawling of supplier B2B portals."""

    def __init__(self, firecrawl_client=None, redis_client=None):
        self.firecrawl = firecrawl_client
        self.redis = redis_client
        self.sessions: Dict[str, dict] = {}  # slug -> {cookies, expires}

    async def get_session(self, supplier_slug: str) -> Optional[Dict[str, str]]:
        """Get cached session cookies for a supplier."""
        # Check Redis first
        if self.redis:
            cached = await self.redis.get(f"supplier_session:{supplier_slug}")
            if cached:
                try:
                    return json.loads(cached)
                except Exception:
                    pass
        return self.sessions.get(supplier_slug, {}).get("cookies")

    async def cache_session(self, supplier_slug: str, cookies: Dict[str, str], ttl: int = 14400):
        """Cache session cookies (default 4h TTL)."""
        self.sessions[supplier_slug] = {"cookies": cookies}
        if self.redis:
            await self.redis.set(
                f"supplier_session:{supplier_slug}",
                json.dumps(cookies),
                ex=ttl,
            )

    # ── AirsoftZone Login ─────────────────────────────────────

    async def login_airsoftzone(self, username: str = None, password: str = None) -> Optional[Dict[str, str]]:
        """Login to AirsoftZone dealer portal."""
        if not HAS_CURL_CFFI:
            logger.error("curl_cffi not available for AirsoftZone login")
            return None

        if not password:
            password = _get_1password_password("airsoftzone.com")
        if not password:
            logger.error("No password provided and 1Password unavailable for AirsoftZone")
            return None

        if not username:
            username = "skirmshop_es"

        try:
            async with CffiAsyncSession(impersonate="chrome") as session:
                # Step 1: GET the dealer/login page
                login_url = "https://www.airsoftzone.com/en/Dealer-c1402/"
                resp = await session.get(login_url, allow_redirects=True)
                logger.info(f"AirsoftZone GET login: {resp.status_code} → {resp.url}")

                # Parse the page for login form
                soup = BeautifulSoup(resp.text, "html.parser")

                # Find login form - look for common patterns
                form = soup.find("form", {"id": re.compile(r"login|sign.?in|dealer", re.I)})
                if not form:
                    # Try finding any form with password field
                    for f in soup.find_all("form"):
                        if f.find("input", {"type": "password"}):
                            form = f
                            break

                if not form:
                    logger.warning("No login form found on AirsoftZone, trying direct POST")
                    # Try common login endpoints
                    form_action = str(resp.url)
                else:
                    form_action = form.get("action", str(resp.url))
                    if form_action and not form_action.startswith("http"):
                        form_action = urljoin(str(resp.url), form_action)

                # Collect hidden fields
                hidden_fields = {}
                if form:
                    for inp in form.find_all("input", {"type": "hidden"}):
                        name = inp.get("name")
                        value = inp.get("value", "")
                        if name:
                            hidden_fields[name] = value

                # Build login payload
                # Try common field names
                payload = {**hidden_fields}

                # Detect field names from form
                username_field = "username"
                password_field = "password"
                if form:
                    for inp in form.find_all("input"):
                        inp_type = (inp.get("type") or "").lower()
                        inp_name = inp.get("name", "")
                        if inp_type == "password":
                            password_field = inp_name
                        elif inp_type in ("text", "email") and inp_name:
                            username_field = inp_name

                payload[username_field] = username
                payload[password_field] = password

                # Step 2: POST login
                resp2 = await session.post(
                    form_action,
                    data=payload,
                    allow_redirects=True,
                )
                logger.info(f"AirsoftZone POST login: {resp2.status_code} → {resp2.url}")

                # Verify login success
                cookies = dict(session.cookies)
                page_text = resp2.text.lower()

                if any(indicator in page_text for indicator in [
                    "my account", "mi cuenta", "logout", "sign out",
                    "dealer", "wholesale", "b2b price",
                ]):
                    logger.info(f"AirsoftZone login successful, {len(cookies)} cookies captured")
                    await self.cache_session("airsoftzone", cookies)
                    return cookies
                else:
                    logger.warning("AirsoftZone login may have failed — no success indicators found")
                    # Return cookies anyway, some sites don't change UI immediately
                    if cookies:
                        await self.cache_session("airsoftzone", cookies)
                        return cookies
                    return None

        except Exception as e:
            logger.error(f"AirsoftZone login failed: {e}")
            return None

    # ── Anareus Login (PrestaShop) ────────────────────────────

    async def login_anareus(self, email: str = None, password: str = None) -> Optional[Dict[str, str]]:
        """Login to Anareus PrestaShop."""
        if not HAS_CURL_CFFI:
            logger.error("curl_cffi not available for Anareus login")
            return None

        if not password:
            password = _get_1password_password("anareus.cz")
        if not password:
            logger.error("No password provided and 1Password unavailable for Anareus")
            return None

        if not email:
            email = "info@skirmshop.es"

        try:
            async with CffiAsyncSession(impersonate="chrome") as session:
                # Step 1: GET auth page to capture session cookie + token
                auth_url = "https://www.anareus.cz/gb/index.php?controller=authentication"
                resp = await session.get(auth_url, allow_redirects=True)
                logger.info(f"Anareus GET auth: {resp.status_code}")

                # Parse for CSRF token (PrestaShop uses token field)
                soup = BeautifulSoup(resp.text, "html.parser")
                token = ""
                token_input = soup.find("input", {"name": "token"})
                if token_input:
                    token = token_input.get("value", "")

                # Step 2: POST login
                login_payload = {
                    "email": email,
                    "passwd": password,
                    "submitLogin": "1",
                    "back": "my-account",
                }
                if token:
                    login_payload["token"] = token

                resp2 = await session.post(
                    auth_url,
                    data=login_payload,
                    allow_redirects=True,
                )
                logger.info(f"Anareus POST login: {resp2.status_code} → {resp2.url}")

                cookies = dict(session.cookies)
                page_text = resp2.text.lower()

                if any(indicator in page_text for indicator in [
                    "my account", "sign out", "log out", "order history",
                    "my-account", "identity",
                ]):
                    logger.info(f"Anareus login successful, {len(cookies)} cookies captured")
                    await self.cache_session("anareus", cookies)
                    return cookies
                else:
                    logger.warning("Anareus login may have failed")
                    if cookies:
                        await self.cache_session("anareus", cookies)
                        return cookies
                    return None

        except Exception as e:
            logger.error(f"Anareus login failed: {e}")
            return None

    # ── Generic Login Dispatcher ──────────────────────────────

    async def login(self, supplier_slug: str, username: str = None, password: str = None) -> Optional[Dict[str, str]]:
        """Login to a supplier, returns cookies."""
        # Check cache first
        cached = await self.get_session(supplier_slug)
        if cached:
            logger.info(f"Using cached session for {supplier_slug}")
            return cached

        if supplier_slug == "airsoftzone":
            return await self.login_airsoftzone(username=username, password=password)
        elif supplier_slug == "anareus":
            return await self.login_anareus(email=username, password=password)
        else:
            logger.warning(f"No login flow for supplier: {supplier_slug}")
            return None

    # ── Authenticated Crawling ────────────────────────────────

    async def scrape_with_cookies(
        self,
        url: str,
        cookies: Dict[str, str],
    ) -> Optional[dict]:
        """Scrape a single page with authentication cookies using curl_cffi."""
        if not HAS_CURL_CFFI:
            logger.error("curl_cffi not available for authenticated scraping")
            return None

        try:
            async with CffiAsyncSession(impersonate="chrome") as session:
                # Set cookies
                for k, v in cookies.items():
                    session.cookies.set(k, v)

                resp = await session.get(url, allow_redirects=True, timeout=30)

                if resp.status_code != 200:
                    logger.warning(f"Auth scrape got {resp.status_code} for {url}")
                    return None

                html = resp.text
                title = ""
                # Extract title
                import re as _re
                title_match = _re.search(r'<title[^>]*>(.*?)</title>', html, _re.IGNORECASE | _re.DOTALL)
                if title_match:
                    title = title_match.group(1).strip()

                # Extract main content via trafilatura
                markdown = ""
                try:
                    import trafilatura
                    markdown = trafilatura.extract(html, include_links=True) or ""
                except Exception:
                    # Fallback: strip HTML tags
                    from bs4 import BeautifulSoup as _BS
                    soup = _BS(html, "html.parser")
                    markdown = soup.get_text(separator="\n", strip=True)

                return {
                    "markdown": markdown,
                    "html": html,
                    "metadata": {
                        "sourceURL": str(resp.url),
                        "title": title,
                    },
                }

        except Exception as e:
            logger.warning(f"curl_cffi auth scrape error for {url}: {e}")
            return None
    async def crawl_product_listings(
        self,
        supplier_slug: str,
        start_url: str,
        cookies: Dict[str, str],
        max_pages: int = 50,
    ) -> List[dict]:
        """Crawl paginated product listings with authentication."""
        all_pages = []
        visited = set()
        to_visit = [start_url]
        page_num = 0

        while to_visit and page_num < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
            visited.add(url)
            page_num += 1

            logger.info(f"[{supplier_slug}] Scraping page {page_num}/{max_pages}: {url}")

            result = await self.scrape_with_cookies(url, cookies)
            if not result:
                continue

            all_pages.append({
                "markdown": result.get("markdown", ""),
                "html": result.get("html", ""),
                "url": result.get("metadata", {}).get("sourceURL", url),
                "title": result.get("metadata", {}).get("title", ""),
            })

            # Detect pagination links
            html = result.get("html", "")
            if html:
                next_pages = self._find_pagination_links(html, url)
                for next_url in next_pages:
                    if next_url not in visited:
                        to_visit.append(next_url)

            # Small delay to be polite
            await asyncio.sleep(1)

        logger.info(f"[{supplier_slug}] Crawled {len(all_pages)} pages")
        return all_pages

    def _find_pagination_links(self, html: str, current_url: str) -> List[str]:
        """Extract pagination links from HTML."""
        soup = BeautifulSoup(html, "html.parser")
        links = set()
        base = urlparse(current_url)

        # Common pagination patterns
        for a in soup.find_all("a", href=True):
            href = a["href"]
            classes = " ".join(a.get("class", []))
            text = a.get_text(strip=True).lower()
            parent_classes = " ".join(a.parent.get("class", [])) if a.parent else ""

            is_pagination = any([
                "pagination" in classes,
                "pagination" in parent_classes,
                "page-link" in classes,
                "next" in classes or "next" in text,
                re.match(r"^\d+$", text),
                re.search(r"[?&]p(age)?=\d+", href),
            ])

            if is_pagination:
                full_url = urljoin(current_url, href)
                parsed = urlparse(full_url)
                if parsed.netloc == base.netloc:
                    links.add(full_url)

        # Also look for category/subcategory links on the first page
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(current_url, href)
            parsed = urlparse(full_url)

            # Match product category URLs
            if parsed.netloc == base.netloc and any(
                p in href.lower() for p in ["/c-", "/category/", "/categor"]
            ):
                links.add(full_url)

        return list(links)

    # ── Full Crawl Pipeline ───────────────────────────────────

    async def authenticated_crawl(
        self,
        supplier_slug: str,
        start_url: str,
        max_pages: int = 50,
        username: str = None,
        password: str = None,
    ) -> Tuple[List[dict], int]:
        """Full pipeline: login → crawl → extract products.

        Returns (pages, product_count).
        """
        # Login
        cookies = await self.login(supplier_slug, username=username, password=password)
        if not cookies:
            return [], 0

        # Crawl
        pages = await self.crawl_product_listings(
            supplier_slug, start_url, cookies, max_pages
        )

        return pages, len(pages)
