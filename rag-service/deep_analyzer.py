"""Deep Analyzer — Detailed competitor analysis using Firecrawl + LLM."""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DeepAnalyzer:
    """Performs deep analysis of competitor sites: pricing, catalog, structure."""

    def __init__(self, firecrawl=None, llm_client=None, vault_path: str = "./knowledge-vault"):
        self.firecrawl = firecrawl
        self.llm_client = llm_client
        self.vault_path = vault_path

    async def analyze_competitor(self, url: str, name: str, slug: str) -> Dict:
        """Run deep analysis on a competitor site."""
        logger.info(f"Deep analyzing: {name} ({url})")

        if not self.firecrawl:
            return {"error": "Firecrawl not available", "name": name}

        # Crawl the site
        pages = await self.firecrawl.crawl(url, max_pages=50)

        return {
            "name": name,
            "url": url,
            "slug": slug,
            "pages_crawled": len(pages) if pages else 0,
            "status": "analyzed",
        }
