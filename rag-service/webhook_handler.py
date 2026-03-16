"""
Shopify Webhook Handler.
Verifies HMAC signatures and processes product/collection CRUD events in real-time.
"""

import base64
import hashlib
import hmac
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ShopifyWebhookHandler:
    """Handles incoming Shopify webhooks with HMAC verification."""

    def __init__(self, webhook_secret: str, shopify_graphql, product_indexer,
                 catalog_indexer, shopify_client, content_hash_store=None,
                 compatibility_analyzer=None):
        self.webhook_secret = webhook_secret
        self.graphql = shopify_graphql
        self.product_indexer = product_indexer
        self.catalog_indexer = catalog_indexer
        self.shopify_client = shopify_client
        self.content_hash_store = content_hash_store
        self.compatibility_analyzer = compatibility_analyzer

    def verify_hmac(self, raw_body: bytes, hmac_header: str) -> bool:
        """Verify Shopify HMAC-SHA256 signature."""
        computed = base64.b64encode(
            hmac.new(
                self.webhook_secret.encode("utf-8"),
                raw_body,
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")
        return hmac.compare_digest(computed, hmac_header)

    async def handle(self, topic: str, shop_domain: str, body: Dict[str, Any]) -> None:
        """Route webhook by topic and process."""
        logger.info(f"Webhook received: {topic} from {shop_domain}")

        shopify_gid = body.get("admin_graphql_api_id")

        if topic in ("products/create", "products/update"):
            await self._handle_product_upsert(shopify_gid, body)
        elif topic == "products/delete":
            await self._handle_product_delete(shopify_gid, body)
        elif topic in ("collections/create", "collections/update"):
            await self._handle_collection_upsert(shopify_gid, body)
        elif topic == "collections/delete":
            await self._handle_collection_delete(shopify_gid, body)
        elif topic == "inventory_levels/update":
            logger.info("Inventory level updated — will pick up in next incremental sync")
        else:
            logger.warning(f"Unhandled webhook topic: {topic}")

    async def _handle_product_upsert(self, shopify_gid: Optional[str], body: Dict) -> None:
        """Fetch full product via GraphQL, re-embed, upsert to Qdrant."""
        if not shopify_gid:
            logger.warning("Product webhook missing admin_graphql_api_id")
            return

        product_node = await self.graphql.fetch_product(shopify_gid)
        if not product_node:
            logger.warning(f"Could not fetch product {shopify_gid}")
            return

        from shopify_graphql import ShopifyGraphQL
        product = ShopifyGraphQL.flatten_graphql_product(product_node)

        # Re-index single product
        from product_indexer import ProductSyncJob
        import uuid
        job = ProductSyncJob(job_id=uuid.uuid4().hex[:8], sync_type="webhook")
        points, _ = await self.product_indexer._process_product_batch(
            [product], self.shopify_client, job, self.content_hash_store
        )
        if points:
            self.product_indexer.client.upsert(
                collection_name=self.product_indexer.COLLECTION_NAME,
                points=points,
            )
        logger.info(f"Webhook: synced product {shopify_gid}")

        # Compatibility analysis (async, non-blocking for the upsert itself)
        if self.compatibility_analyzer:
            try:
                compat_result = await self.compatibility_analyzer.analyze_product(product_data)
                self.product_indexer.update_compatibility(
                    shopify_id=str(product_data.get("id", "")),
                    payload=compat_result.to_payload(),
                )
                logger.info(
                    f"Compatibility updated for {product_data.get('title', '?')}: "
                    f"platforms={compat_result.compatible_platforms}, type={compat_result.upgrade_type}"
                )
            except Exception as e:
                logger.error(f"Compatibility analysis failed for webhook product: {e}")

    async def _handle_product_delete(self, shopify_gid: Optional[str], body: Dict) -> None:
        """Delete product from Qdrant."""
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        gid = shopify_gid or f"gid://shopify/Product/{body.get('id', '')}"
        try:
            self.product_indexer.client.delete(
                collection_name=self.product_indexer.COLLECTION_NAME,
                points_selector=Filter(must=[
                    FieldCondition(key="shopify_id", match=MatchValue(value=gid))
                ]),
            )
            if self.content_hash_store:
                await self.content_hash_store.delete_hash(f"product:{gid}")
            logger.info(f"Webhook: deleted product {gid}")
        except Exception as e:
            logger.error(f"Webhook: failed to delete product {gid}: {e}")

    async def _handle_collection_upsert(self, shopify_gid: Optional[str], body: Dict) -> None:
        """Fetch full collection via GraphQL, re-embed, upsert to Qdrant."""
        if not shopify_gid:
            logger.warning("Collection webhook missing admin_graphql_api_id")
            return

        coll_node = await self.graphql.fetch_collection(shopify_gid)
        if not coll_node:
            logger.warning(f"Could not fetch collection {shopify_gid}")
            return

        from shopify_graphql import ShopifyGraphQL
        collection = ShopifyGraphQL.flatten_graphql_collection(coll_node)

        await self.catalog_indexer.index_collections(
            [collection], self.shopify_client, self.content_hash_store
        )
        logger.info(f"Webhook: synced collection {shopify_gid}")

    async def _handle_collection_delete(self, shopify_gid: Optional[str], body: Dict) -> None:
        """Delete collection from Qdrant."""
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        gid = shopify_gid or f"gid://shopify/Collection/{body.get('id', '')}"
        try:
            self.catalog_indexer.client.delete(
                collection_name="product_collections",
                points_selector=Filter(must=[
                    FieldCondition(key="shopify_id", match=MatchValue(value=gid))
                ]),
            )
            if self.content_hash_store:
                await self.content_hash_store.delete_hash(f"collection:{gid}")
            logger.info(f"Webhook: deleted collection {gid}")
        except Exception as e:
            logger.error(f"Webhook: failed to delete collection {gid}: {e}")
