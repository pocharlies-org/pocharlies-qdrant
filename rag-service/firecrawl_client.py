"""
Firecrawl Client — integrates with local self-hosted Firecrawl for crawling
sites that are Cloudflare-protected or otherwise inaccessible to regular crawlers.

Provides a simple interface matching what web_indexer needs:
- scrape(url) → returns markdown content
- crawl(url, max_pages, max_depth) → returns list of page results
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

FIRECRAWL_URL = os.getenv("FIRECRAWL_URL", "http://localhost:3003")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "fc-skirmshop-local")


class FirecrawlClient:
    """Client for local self-hosted Firecrawl instance."""

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
    ):
        self.base_url = (base_url or FIRECRAWL_URL).rstrip("/")
        self.api_key = api_key or FIRECRAWL_API_KEY
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def is_available(self) -> bool:
        """Check if Firecrawl is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(self.base_url, headers=self._headers)
                return resp.status_code == 200
        except Exception:
            return False

    async def scrape(
        self,
        url: str,
        formats: List[str] = None,
        wait_for: int = 3000,
        timeout: int = 30000,
        only_main_content: bool = True,
    ) -> Optional[dict]:
        """Scrape a single URL and return the result.

        Returns dict with keys: markdown, html, metadata, etc.
        Returns None on failure.
        """
        payload = {
            "url": url,
            "formats": formats or ["markdown"],
            "waitFor": wait_for,
            "timeout": timeout,
            "onlyMainContent": only_main_content,
        }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/scrape",
                    json=payload,
                    headers=self._headers,
                )
                data = resp.json()

                if data.get("success"):
                    return data.get("data", {})
                else:
                    logger.warning(f"Firecrawl scrape failed for {url}: {data}")
                    return None

        except Exception as e:
            logger.warning(f"Firecrawl scrape error for {url}: {e}")
            return None

    async def crawl(
        self,
        url: str,
        max_pages: int = 100,
        max_depth: int = 3,
        include_paths: List[str] = None,
        exclude_paths: List[str] = None,
        poll_interval: int = 10,
        timeout_minutes: int = 120,  # 2 hours for large sites (10k+ pages)
    ) -> List[dict]:
        """Crawl a website and return all page results.

        Returns list of dicts, each with: markdown, url, metadata, etc.
        """
        payload = {
            "url": url,
            "limit": max_pages,
            "maxDiscoveryDepth": max_depth,
            "scrapeOptions": {
                "formats": ["markdown"],
                "onlyMainContent": True,
                "waitFor": 3000,
            },
        }
        if include_paths:
            payload["includePaths"] = include_paths
        if exclude_paths:
            payload["excludePaths"] = exclude_paths

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                # Start crawl
                resp = await client.post(
                    f"{self.base_url}/v1/crawl",
                    json=payload,
                    headers=self._headers,
                )
                data = resp.json()

                if not data.get("success") and not data.get("id"):
                    logger.error(f"Firecrawl crawl start failed: {data}")
                    return []

                crawl_id = data.get("id")
                if not crawl_id:
                    logger.error(f"No crawl ID returned: {data}")
                    return []

                logger.info(f"Firecrawl crawl started: {crawl_id} for {url}")

                # Poll for completion
                max_polls = (timeout_minutes * 60) // poll_interval
                all_results = []

                for i in range(max_polls):
                    await asyncio.sleep(poll_interval)

                    status_resp = await client.get(
                        f"{self.base_url}/v1/crawl/{crawl_id}",
                        headers=self._headers,
                    )
                    status_data = status_resp.json()
                    status = status_data.get("status", "unknown")

                    completed = status_data.get("completed", 0)
                    total = status_data.get("total", 0)

                    if i % 6 == 0:  # Log every minute
                        logger.info(f"Firecrawl crawl {crawl_id}: {status} ({completed}/{total} pages)")

                    if status == "completed":
                        # Collect results
                        results = status_data.get("data", [])
                        all_results.extend(results)

                        # Handle pagination if needed
                        next_url = status_data.get("next")
                        while next_url:
                            next_resp = await client.get(next_url, headers=self._headers)
                            next_data = next_resp.json()
                            results = next_data.get("data", [])
                            all_results.extend(results)
                            next_url = next_data.get("next")

                        logger.info(f"Firecrawl crawl complete: {len(all_results)} pages from {url}")
                        return all_results

                    elif status in ("failed", "cancelled"):
                        logger.error(f"Firecrawl crawl {status}: {status_data}")
                        return all_results

                logger.warning(f"Firecrawl crawl timed out after {timeout_minutes}min")
                return all_results

        except Exception as e:
            logger.error(f"Firecrawl crawl error for {url}: {e}")
            return []

    async def crawl_to_chunks(
        self,
        url: str,
        max_pages: int = 100,
        max_depth: int = 3,
    ) -> List[dict]:
        """Crawl and return chunks compatible with web_indexer format.

        Returns list of dicts with: text, url, title, domain, source_type
        """
        from urllib.parse import urlparse

        results = await self.crawl(url, max_pages=max_pages, max_depth=max_depth)
        chunks = []

        for page in results:
            markdown = page.get("markdown", "")
            metadata = page.get("metadata", {})
            page_url = metadata.get("sourceURL", metadata.get("url", url))
            title = metadata.get("title", "")
            domain = urlparse(page_url).netloc

            if markdown and len(markdown) > 100:
                chunks.append({
                    "text": markdown,
                    "url": page_url,
                    "title": title,
                    "domain": domain,
                    "source_type": "firecrawl",
                    "content_length": len(markdown),
                })

        logger.info(f"Firecrawl: {len(chunks)} chunks from {url}")
        return chunks
