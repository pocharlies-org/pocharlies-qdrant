"""
Supplier Email Intelligence Monitor for Pocharlies RAG
Monitors incoming emails from known supplier domains, extracts intelligence
(promotions, price changes, stock updates), and indexes into supplier knowledge.
"""

import json
import logging
import subprocess
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SupplierEmailMonitor:
    """Monitors supplier emails and extracts intelligence."""

    def __init__(
        self,
        supplier_registry=None,
        llm_client=None,
        llm_model: str = "smart",
        redis_client=None,
    ):
        self.supplier_registry = supplier_registry
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.redis = redis_client
        self._domain_map = {}

    def _refresh_domain_map(self):
        """Build email domain → supplier slug mapping from registry."""
        if self.supplier_registry:
            self._domain_map = self.supplier_registry.get_email_domains()
            # Add known supplier domains that might not be in contact_email
            extra_domains = {
                "silverback-airsoft.com": "silverback",
                "taiwangun.com": "taiwangun",
                "gfcorp.pl": "gunfire",
                "gunfire.com": "gunfire",
                "3p-store.com": "3p-store",
            }
            for domain, slug in extra_domains.items():
                if domain not in self._domain_map:
                    self._domain_map[domain] = slug

    async def check_emails(self, days: int = 7) -> List[dict]:
        """Check recent emails from supplier domains and extract intelligence."""
        self._refresh_domain_map()
        if not self._domain_map:
            logger.warning("No supplier email domains configured")
            return []

        # Fetch recent emails via mcporter
        emails = await self._fetch_recent_emails(days)
        if not emails:
            return []

        intelligence = []
        for email in emails:
            sender = email.get("from", "")
            message_id = email.get("id", "")

            # Check if already processed
            if self.redis and message_id:
                seen = await self.redis.get(f"supplier_email:seen:{message_id}")
                if seen:
                    continue

            # Match sender to supplier
            supplier_slug = self._match_sender(sender)
            if not supplier_slug:
                continue

            # Extract intelligence
            intel = await self._extract_intelligence(email, supplier_slug)
            if intel:
                intelligence.append(intel)

                # Mark as processed
                if self.redis and message_id:
                    await self.redis.set(
                        f"supplier_email:seen:{message_id}",
                        "1",
                        ex=86400 * 30,  # 30 day TTL
                    )

        logger.info(f"Processed {len(intelligence)} supplier emails")
        return intelligence

    async def get_digest(self, supplier_slug: Optional[str] = None, days: int = 7) -> dict:
        """Get recent supplier email intelligence digest."""
        intelligence = await self.check_emails(days=days)

        if supplier_slug:
            intelligence = [i for i in intelligence if i.get("supplier_slug") == supplier_slug]

        # Group by type
        by_type = {}
        for intel in intelligence:
            t = intel.get("type", "other")
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(intel)

        # Group by supplier
        by_supplier = {}
        for intel in intelligence:
            s = intel.get("supplier_slug", "unknown")
            if s not in by_supplier:
                by_supplier[s] = []
            by_supplier[s].append(intel)

        return {
            "period_days": days,
            "total_emails": len(intelligence),
            "by_type": {k: len(v) for k, v in by_type.items()},
            "by_supplier": {k: len(v) for k, v in by_supplier.items()},
            "intelligence": intelligence[:50],
        }

    def _match_sender(self, sender: str) -> Optional[str]:
        """Match email sender to a supplier slug."""
        sender_lower = sender.lower()
        for domain, slug in self._domain_map.items():
            if domain.lower() in sender_lower:
                return slug
        return None

    async def _fetch_recent_emails(self, days: int = 7) -> List[dict]:
        """Fetch recent emails from Gmail via mcporter."""
        try:
            since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y/%m/%d")
            result = subprocess.run(
                ["mcporter", "call", "gmail-skirmshop.gmail_search_messages",
                 f"query=after:{since}", "maxResults=50", "--output", "json"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                messages = data if isinstance(data, list) else data.get("messages", [])
                return messages
        except Exception as e:
            logger.error(f"Failed to fetch emails: {e}")
        return []

    async def _extract_intelligence(self, email: dict, supplier_slug: str) -> Optional[dict]:
        """Extract intelligence from an email using LLM."""
        subject = email.get("subject", "")
        body = email.get("snippet", email.get("body", ""))[:2000]
        sender = email.get("from", "")
        date = email.get("date", "")

        if not body and not subject:
            return None

        # Simple keyword-based classification (fast, no LLM needed)
        text_lower = (subject + " " + body).lower()

        intel_type = "other"
        if any(w in text_lower for w in ["promotion", "sale", "discount", "off", "promo", "oferta"]):
            intel_type = "promotion"
        elif any(w in text_lower for w in ["new price", "price change", "price update", "nuevo precio"]):
            intel_type = "price_change"
        elif any(w in text_lower for w in ["in stock", "arrived", "available", "new arrival", "stock", "llegada"]):
            intel_type = "stock_update"
        elif any(w in text_lower for w in ["new product", "nuevo producto", "launch", "lanzamiento"]):
            intel_type = "new_product"
        elif any(w in text_lower for w in ["invoice", "factura", "payment", "pago", "shipment", "envío"]):
            intel_type = "order_update"

        intel = {
            "supplier_slug": supplier_slug,
            "type": intel_type,
            "subject": subject,
            "summary": body[:300],
            "sender": sender,
            "date": date,
            "email_id": email.get("id", ""),
        }

        # For promotions and price changes, try LLM extraction for details
        if self.llm_client and intel_type in ("promotion", "price_change", "new_product"):
            details = await self._llm_extract_details(subject, body, intel_type)
            if details:
                intel["details"] = details

        return intel

    async def _llm_extract_details(self, subject: str, body: str, intel_type: str) -> Optional[dict]:
        """Use LLM to extract structured details from email."""
        if not self.llm_client:
            return None

        prompt = f"""Extract key information from this supplier email.
Type: {intel_type}
Subject: {subject}
Body: {body[:1500]}

Return a JSON object with:
- products: list of product names mentioned (max 10)
- discount_pct: discount percentage if mentioned
- valid_until: expiry date if mentioned
- key_brands: brands mentioned
- summary: one-sentence summary

Return ONLY the JSON object."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                import re
                content = re.sub(r"```\w*\n?", "", content).strip()
            return json.loads(content)
        except Exception as e:
            logger.debug(f"LLM email extraction failed: {e}")
            return None
