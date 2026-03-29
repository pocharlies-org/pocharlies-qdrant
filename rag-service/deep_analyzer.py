"""
Deep Analyzer — Strategic competitor analysis using Firecrawl + LLM.

Instead of crawling 500 pages blindly, this:
1. Maps the sitemap to understand site structure (instant)
2. Scrapes strategic pages: homepage, main categories, sample products, about/services
3. Sends all content to LLM for deep competitive intelligence
"""

import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from firecrawl_client import FirecrawlClient

logger = logging.getLogger(__name__)


class DeepAnalyzer:
    """Deep competitive analysis using sitemap mapping + strategic scraping + LLM."""

    def __init__(self, firecrawl: FirecrawlClient, llm_client, vault_path: str):
        self.fc = firecrawl
        self.llm = llm_client
        self.vault_path = Path(vault_path)
        self.model = os.getenv("LLM_MODEL", "local")

    async def analyze_competitor(self, url: str, name: str, slug: str) -> dict:
        """Full deep analysis of a single competitor.

        Returns dict with all analysis sections ready for note rendering.
        """
        result = {
            "name": name,
            "slug": slug,
            "url": url,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "sitemap": {},
            "pages_scraped": 0,
            "brands": [],
            "categories": [],
            "products": [],
            "services": [],
            "analysis": {},
            "errors": [],
        }

        try:
            # ── Step 1: Map the sitemap (instant) ──
            logger.info(f"[{name}] Step 1: Mapping sitemap...")
            sitemap = await self._map_sitemap(url)
            result["sitemap"] = sitemap
            logger.info(f"[{name}] Sitemap: {sitemap.get('total_urls', 0)} URLs found")

            # ── Step 2: Scrape strategic pages ──
            logger.info(f"[{name}] Step 2: Scraping strategic pages...")
            pages = await self._scrape_strategic_pages(url, name, sitemap)
            result["pages_scraped"] = len(pages)
            logger.info(f"[{name}] Scraped {len(pages)} strategic pages")

            # ── Step 3: LLM deep analysis ──
            logger.info(f"[{name}] Step 3: LLM deep analysis...")
            analysis = await self._llm_deep_analysis(name, url, sitemap, pages)
            result["analysis"] = analysis
            logger.info(f"[{name}] Analysis complete")

            result["completed_at"] = datetime.now(timezone.utc).isoformat()

        except Exception as e:
            logger.error(f"[{name}] Analysis failed: {e}")
            result["errors"].append(str(e)[:300])

        return result

    async def _map_sitemap(self, url: str) -> dict:
        """Map site structure using Firecrawl's map endpoint + sitemap.xml."""
        import httpx

        sitemap = {
            "total_urls": 0,
            "categories": {},
            "url_patterns": {},
            "has_blog": False,
            "has_services": False,
            "has_workshop": False,
            "brands_in_urls": [],
            "estimated_products": 0,
        }

        # Try Firecrawl map
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.fc.base_url}/v1/map",
                    json={"url": url, "limit": 5000},
                    headers=self.fc._headers,
                )
                data = resp.json()
                if data.get("success"):
                    urls = data.get("links", [])
                    sitemap["total_urls"] = len(urls)
                    sitemap = self._classify_urls(urls, url, sitemap)
        except Exception as e:
            logger.warning(f"Firecrawl map failed: {e}")

        # Try sitemap.xml directly (with Firecrawl for CF-protected sites)
        sitemap_urls = await self._fetch_sitemap_xml(url)
        if sitemap_urls and len(sitemap_urls) > sitemap["total_urls"]:
            sitemap["total_urls"] = len(sitemap_urls)
            sitemap = self._classify_urls(sitemap_urls, url, sitemap)
            logger.info(f"Sitemap.xml: {len(sitemap_urls)} URLs")

        # Fallback: Google site: search to discover URLs for CF-protected sites
        if sitemap["total_urls"] <= 5:
            logger.info(f"Poor sitemap ({sitemap['total_urls']} URLs), using Google site: search...")
            google_urls = await self._google_site_search(url)
            if google_urls and len(google_urls) > sitemap["total_urls"]:
                sitemap["total_urls"] = len(google_urls)
                sitemap = self._classify_urls(google_urls, url, sitemap)
                sitemap["source"] = "google"
                logger.info(f"Google site: search found {len(google_urls)} URLs")

        return sitemap

    async def _fetch_sitemap_xml(self, url: str) -> List[str]:
        """Fetch sitemap.xml — try direct HTTP first, then Firecrawl for CF sites."""
        import httpx

        sitemap_url = f"{url.rstrip('/')}/sitemap.xml"

        # Try direct HTTP
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                resp = await client.get(sitemap_url)
                if resp.status_code == 200:
                    content_type = resp.headers.get("content-type", "")
                    if "xml" in content_type or "<urlset" in resp.text[:500] or "<sitemapindex" in resp.text[:500]:
                        urls = re.findall(r"<loc>(.*?)</loc>", resp.text)
                        # Handle sitemap index (sitemaps pointing to other sitemaps)
                        sub_sitemaps = [u for u in urls if "sitemap" in u.lower() and u.endswith(".xml")]
                        if sub_sitemaps:
                            all_urls = [u for u in urls if u not in sub_sitemaps]
                            for sub_url in sub_sitemaps[:5]:  # Process up to 5 sub-sitemaps
                                try:
                                    sub_resp = await client.get(sub_url)
                                    if sub_resp.status_code == 200:
                                        all_urls.extend(re.findall(r"<loc>(.*?)</loc>", sub_resp.text))
                                except Exception:
                                    pass
                            return all_urls
                        return urls
        except Exception:
            pass

        # Fallback: use Firecrawl to scrape sitemap.xml (handles CF)
        try:
            result = await self.fc.scrape(sitemap_url, formats=["markdown"], wait_for=3000, timeout=15000)
            if result and result.get("markdown"):
                # Extract URLs from the rendered sitemap content
                text = result["markdown"]
                urls = re.findall(r"https?://[^\s\)\"'>]+", text)
                domain = url.replace("https://", "").replace("http://", "").split("/")[0]
                return [u for u in urls if domain in u]
        except Exception:
            pass

        return []

    async def _google_site_search(self, url: str) -> List[str]:
        """Use Firecrawl's search or Google to discover URLs for a domain."""
        import httpx

        domain = url.replace("https://", "").replace("http://", "").rstrip("/")

        # Try Firecrawl search endpoint
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.fc.base_url}/v1/search",
                    json={"query": f"site:{domain}", "limit": 50},
                    headers=self.fc._headers,
                )
                data = resp.json()
                if data.get("success") and data.get("data"):
                    urls = []
                    for result in data["data"]:
                        page_url = result.get("metadata", {}).get("sourceURL", result.get("url", ""))
                        if page_url and domain in page_url:
                            urls.append(page_url)
                    if urls:
                        return urls
        except Exception as e:
            logger.debug(f"Firecrawl search failed: {e}")

        # Fallback: scrape Google search results page
        try:
            google_url = f"https://www.google.com/search?q=site:{domain}&num=100"
            result = await self.fc.scrape(google_url, formats=["markdown"], wait_for=3000, timeout=15000)
            if result and result.get("markdown"):
                # Extract domain URLs from Google results
                urls = re.findall(rf"https?://(?:www\.)?{re.escape(domain)}[^\s\)\"'>]*", result["markdown"])
                return list(set(urls))
        except Exception as e:
            logger.debug(f"Google scrape failed: {e}")

        # Last resort: scrape the homepage with Firecrawl and extract internal links
        try:
            result = await self.fc.scrape(url, formats=["links"], wait_for=5000, timeout=20000)
            if result:
                links = result.get("links", [])
                if links:
                    return [l for l in links if domain in l][:100]
        except Exception:
            pass

        return []

    def _classify_urls(self, urls: List[str], base_url: str, sitemap: dict) -> dict:
        """Classify URLs by type to understand site structure."""
        domain = base_url.replace("https://", "").replace("http://", "").rstrip("/")
        categories = defaultdict(int)
        product_count = 0
        brand_urls = []

        product_patterns = [r"-p-\d", r"/producto/", r"/product/", r"\.html$", r"/p/\d"]
        category_patterns = [r"-lp-", r"/categoria/", r"/category/", r"/c/", r"familia"]
        blog_patterns = [r"/blog", r"/noticias", r"/news", r"/articulo"]
        service_patterns = [r"/taller", r"/servicio", r"/service", r"/reparacion", r"/workshop"]
        brand_patterns = [r"/marca", r"/brand", r"-marca-"]

        for url_str in urls:
            path = url_str.replace(f"https://{domain}", "").replace(f"https://www.{domain}", "")
            path_lower = path.lower()

            # Classify
            if any(re.search(p, path_lower) for p in product_patterns):
                product_count += 1
                # Extract category from product URL
                parts = path.strip("/").split("/")
                if len(parts) > 1:
                    categories[parts[0]] += 1
            elif any(re.search(p, path_lower) for p in category_patterns):
                parts = path.strip("/").split("/")
                cat_name = parts[-1] if parts else "other"
                categories[cat_name] += 1
            elif any(re.search(p, path_lower) for p in blog_patterns):
                sitemap["has_blog"] = True
            elif any(re.search(p, path_lower) for p in service_patterns):
                sitemap["has_services"] = True
                sitemap["has_workshop"] = True
            elif any(re.search(p, path_lower) for p in brand_patterns):
                brand_name = path.strip("/").split("/")[-1].replace("-", " ").title()
                brand_urls.append(brand_name)

        sitemap["categories"] = dict(sorted(categories.items(), key=lambda x: -x[1])[:30])
        sitemap["estimated_products"] = product_count
        sitemap["brands_in_urls"] = brand_urls[:50]
        return sitemap

    async def _scrape_strategic_pages(self, url: str, name: str, sitemap: dict) -> List[dict]:
        """Scrape key pages for analysis — not everything, just strategic ones."""
        pages_to_scrape = []

        # Always scrape: homepage
        pages_to_scrape.append({"url": url, "type": "homepage"})

        # If we have a good sitemap, pick strategic pages from it
        if sitemap.get("total_urls", 0) > 10:
            # Scrape top category pages (up to 5)
            for cat in list(sitemap.get("categories", {}).keys())[:5]:
                cat_url = f"{url.rstrip('/')}/{cat}"
                pages_to_scrape.append({"url": cat_url, "type": "category"})

            # Scrape service/workshop pages if detected
            if sitemap.get("has_services") or sitemap.get("has_workshop"):
                for keyword in ["taller", "servicios", "services", "workshop", "reparaciones"]:
                    pages_to_scrape.append({"url": f"{url.rstrip('/')}/{keyword}", "type": "services"})

            # Scrape blog if exists
            if sitemap.get("has_blog"):
                pages_to_scrape.append({"url": f"{url.rstrip('/')}/blog", "type": "blog"})
        else:
            # Poor sitemap — use Firecrawl mini-crawl to discover pages
            logger.info(f"[{name}] Poor sitemap, using Firecrawl mini-crawl to discover pages...")
            try:
                fc_pages = await self.fc.crawl(url, max_pages=30, max_depth=2, timeout_minutes=5)
                for page in fc_pages:
                    md = page.get("markdown", "")
                    meta = page.get("metadata", {})
                    page_url = meta.get("sourceURL", meta.get("url", ""))
                    title = meta.get("title", "")
                    if md and len(md) > 100 and page_url:
                        # Classify page type from URL/title
                        page_lower = (page_url + " " + title).lower()
                        if any(k in page_lower for k in ["taller", "servicio", "repair", "workshop"]):
                            ptype = "services"
                        elif any(k in page_lower for k in ["replica", "arma", "gun", "rifle", "pistol", "aeg", "gbb"]):
                            ptype = "category"
                        elif any(k in page_lower for k in ["about", "sobre", "quienes", "nosotros", "contact"]):
                            ptype = "about"
                        elif any(k in page_lower for k in ["blog", "noticia", "guia", "guide"]):
                            ptype = "blog"
                        else:
                            ptype = "page"
                        pages_to_scrape.append({
                            "url": page_url,
                            "type": ptype,
                            "pre_scraped": True,
                            "markdown": md[:1500],
                            "title": title,
                            "description": meta.get("description", ""),
                        })
                logger.info(f"[{name}] Firecrawl mini-crawl discovered {len(pages_to_scrape) - 1} pages")
            except Exception as e:
                logger.warning(f"[{name}] Firecrawl mini-crawl failed: {e}")

        # Common pages to try for all sites
        for page in ["about", "sobre-nosotros", "quienes-somos", "contacto", "contact"]:
            pages_to_scrape.append({"url": f"{url.rstrip('/')}/{page}", "type": "about"})

        # Scrape pages that weren't pre-scraped
        scraped = []
        for page in pages_to_scrape:
            # If already scraped by mini-crawl, add directly
            if page.get("pre_scraped"):
                scraped.append({
                    "url": page["url"],
                    "type": page["type"],
                    "title": page.get("title", ""),
                    "markdown": page["markdown"],
                    "description": page.get("description", ""),
                })
                continue

            try:
                result = await self.fc.scrape(
                    page["url"],
                    formats=["markdown"],
                    wait_for=3000,
                    timeout=15000,
                    only_main_content=True,
                )
                if result and result.get("markdown") and len(result["markdown"]) > 100:
                    scraped.append({
                        "url": page["url"],
                        "type": page["type"],
                        "title": result.get("metadata", {}).get("title", ""),
                        "markdown": result["markdown"][:1500],
                        "description": result.get("metadata", {}).get("description", ""),
                    })
            except Exception as e:
                logger.debug(f"Failed to scrape {page['url']}: {e}")

        return scraped

    async def _llm_deep_analysis(self, name: str, url: str, sitemap: dict, pages: List[dict]) -> dict:
        """Send all collected data to LLM for deep competitive intelligence."""
        if not self.llm:
            return {"error": "No LLM configured"}

        # Build the content digest
        page_summaries = []
        for p in pages:
            page_summaries.append(
                f"### [{p['type'].upper()}] {p['title']}\nURL: {p['url']}\n"
                f"Description: {p.get('description', 'N/A')}\n"
                f"Content:\n{p['markdown'][:2000]}\n"
            )

        content_block = "\n---\n".join(page_summaries)
        # Cap total content to avoid LLM timeout
        if len(content_block) > 12000:
            content_block = content_block[:12000] + "\n\n[...truncated for length...]"

        sitemap_summary = (
            f"Total URLs discovered: {sitemap.get('total_urls', 'unknown')}\n"
            f"Estimated products: {sitemap.get('estimated_products', 'unknown')}\n"
            f"Has blog: {sitemap.get('has_blog', False)}\n"
            f"Has workshop/services: {sitemap.get('has_services', False)}\n"
            f"Top categories in sitemap: {json.dumps(dict(list(sitemap.get('categories', {}).items())[:15]))}\n"
            f"Brands found in URLs: {', '.join(sitemap.get('brands_in_urls', [])[:20])}\n"
        )

        prompt = f"""You are a senior competitive intelligence analyst for Skirmshop.es, a Spanish online airsoft retailer.

Perform a DEEP analysis of competitor "{name}" ({url}).

## SITEMAP STRUCTURE
{sitemap_summary}

## SCRAPED PAGES ({len(pages)} pages)
{content_block[:8000]}

## ANALYSIS REQUIRED

Analyze and respond in this EXACT format (use all sections):

MARKET_POSITIONING:
[3-5 sentences: Who is their target customer? What's their unique angle? How do they position vs other stores? Are they budget, mid-range, or premium?]

BRAND_PORTFOLIO:
[List every brand you can identify from the content. For each: brand name, approximate product count if visible, and whether it's a key brand for them]

CATEGORY_BREAKDOWN:
[List their main product categories with estimated depth. Which categories do they dominate? Which are thin?]

PRICING_STRATEGY:
[Analyze their pricing approach. Price ranges by category. Do they compete on price or value? Any notable pricing patterns?]

SERVICES_AND_DIFFERENTIATORS:
[What services do they offer beyond retail? Workshop? Repairs? Custom builds? Events? What makes them unique?]

CONTENT_AND_SEO:
[What's their content strategy? Blog? Guides? What keywords are they targeting? How strong is their SEO?]

WEBSITE_TECHNOLOGY:
[What platform do they use? PrestaShop, WooCommerce, Shopify, custom? Any notable UX features?]

STRENGTHS:
- [strength 1 with specific evidence]
- [strength 2]
- [strength 3]
- [strength 4]
- [strength 5]

WEAKNESSES:
- [weakness 1 with specific evidence]
- [weakness 2]
- [weakness 3]

THREATS_TO_SKIRMSHOP:
[How could this competitor take our customers? What are they doing better?]

OPPORTUNITIES_FOR_SKIRMSHOP:
[What gaps can we exploit? What products/services should we add? What can we learn from them?]

KEY_METRICS:
- Estimated catalog size: [number]
- Price range: €[min] - €[max]
- Number of brands: [number]
- Main categories: [list]
- Has physical store: [yes/no/unknown]
- Has workshop: [yes/no]
- Ships to: [countries]"""

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": (
                            "You are a senior competitive intelligence analyst. "
                            "Provide detailed, specific, evidence-based analysis. "
                            "Reference specific products, prices, and brands you see in the data. "
                            "Output ONLY the analysis in the exact format requested. "
                            "No meta-commentary, no thinking process."
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=4000,
                    timeout=300,
                    extra_body={"request_timeout": 300},
                ),
            )
            text = response.choices[0].message.content.strip()
            # Strip thinking tokens
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            # Parse into sections
            analysis = {}
            sections = [
                "MARKET_POSITIONING", "BRAND_PORTFOLIO", "CATEGORY_BREAKDOWN",
                "PRICING_STRATEGY", "SERVICES_AND_DIFFERENTIATORS", "CONTENT_AND_SEO",
                "WEBSITE_TECHNOLOGY", "STRENGTHS", "WEAKNESSES",
                "THREATS_TO_SKIRMSHOP", "OPPORTUNITIES_FOR_SKIRMSHOP", "KEY_METRICS",
            ]
            for section in sections:
                pattern = rf"{section}:\s*\n(.*?)(?=\n[A-Z_]{{3,}}:|\Z)"
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    analysis[section] = match.group(1).strip()

            analysis["raw"] = text
            return analysis

        except Exception as e:
            logger.error(f"LLM deep analysis failed for {name}: {e}")
            return {"error": str(e)[:300]}

    def write_deep_note(self, result: dict) -> str:
        """Write a deep analysis note to the vault."""
        slug = result["slug"]
        name = result["name"]
        analysis = result.get("analysis", {})
        sitemap = result.get("sitemap", {})

        content = f"""---
type: competitor-deep
name: {name}
slug: {slug}
url: {result['url']}
analyzed_at: {result.get('completed_at', datetime.now(timezone.utc).isoformat())}
sitemap_urls: {sitemap.get('total_urls', 0)}
estimated_products: {sitemap.get('estimated_products', 0)}
pages_analyzed: {result.get('pages_scraped', 0)}
has_workshop: {sitemap.get('has_services', False)}
has_blog: {sitemap.get('has_blog', False)}
---

# {name} — Deep Analysis

## Market Positioning
{analysis.get('MARKET_POSITIONING', '_Pending._')}

## Brand Portfolio
{analysis.get('BRAND_PORTFOLIO', '_Pending._')}

## Category Breakdown
{analysis.get('CATEGORY_BREAKDOWN', '_Pending._')}

## Pricing Strategy
{analysis.get('PRICING_STRATEGY', '_Pending._')}

## Services & Differentiators
{analysis.get('SERVICES_AND_DIFFERENTIATORS', '_Pending._')}

## Content & SEO Strategy
{analysis.get('CONTENT_AND_SEO', '_Pending._')}

## Website & Technology
{analysis.get('WEBSITE_TECHNOLOGY', '_Pending._')}

## Strengths
{analysis.get('STRENGTHS', '_Pending._')}

## Weaknesses
{analysis.get('WEAKNESSES', '_Pending._')}

## Threats to Skirmshop
{analysis.get('THREATS_TO_SKIRMSHOP', '_Pending._')}

## Opportunities for Skirmshop
{analysis.get('OPPORTUNITIES_FOR_SKIRMSHOP', '_Pending._')}

## Key Metrics
{analysis.get('KEY_METRICS', '_Pending._')}

## Sitemap Structure
- Total URLs: {sitemap.get('total_urls', 'unknown')}
- Estimated products: {sitemap.get('estimated_products', 'unknown')}
- Top categories: see below

<!-- human-start -->
## Strategic Notes
_Add your own analysis here. This section is preserved across rebuilds._
<!-- human-end -->
"""

        # Append category data outside f-string to avoid dict issues
        top_cats = dict(list(sitemap.get("categories", {}).items())[:10])
        if top_cats:
            cat_lines = "\n".join(f"  - {k}: {v} URLs" for k, v in top_cats.items())
            content = content.replace("- Top categories: see below", f"- Top categories:\n{cat_lines}")

        path = self.vault_path / "competitors" / f"{slug}-deep.md"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Preserve human blocks
        if path.exists():
            old = path.read_text(encoding="utf-8")
            human_blocks = re.findall(r"(<!-- human-start -->.*?<!-- human-end -->)", old, re.DOTALL)
            new_blocks = re.findall(r"(<!-- human-start -->.*?<!-- human-end -->)", content, re.DOTALL)
            for i, old_block in enumerate(human_blocks):
                if i < len(new_blocks):
                    content = content.replace(new_blocks[i], old_block, 1)

        path.write_text(content, encoding="utf-8")
        logger.info(f"Wrote deep analysis: {path}")
        return str(path)
