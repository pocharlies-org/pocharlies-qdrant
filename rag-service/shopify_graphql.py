"""
Shopify GraphQL + Bulk Operations Client.
Ported from DGX TypeScript implementation to Python.
Provides fast bulk export of products, collections, and pages via Shopify's async bulk API.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class ShopifyGraphQL:
    """GraphQL client for Shopify Admin API with bulk operations support."""

    API_VERSION = "2024-10"

    def __init__(self, shop_domain: str, access_token: str):
        self.shop_domain = shop_domain
        self.access_token = access_token
        self.endpoint = f"https://{shop_domain}/admin/api/{self.API_VERSION}/graphql.json"
        self.headers = {
            "X-Shopify-Access-Token": access_token,
            "Content-Type": "application/json",
        }

    async def query(self, gql: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a GraphQL query/mutation with rate limit handling."""
        body: Dict[str, Any] = {"query": gql}
        if variables:
            body["variables"] = variables

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self.endpoint, headers=self.headers, json=body)
            resp.raise_for_status()
            result = resp.json()

        if result.get("errors"):
            msgs = [e["message"] for e in result["errors"]]
            raise Exception(f"Shopify GraphQL errors: {', '.join(msgs)}")

        cost = result.get("extensions", {}).get("cost", {})
        throttle = cost.get("throttleStatus", {})
        available = throttle.get("currentlyAvailable", 1000)
        restore_rate = throttle.get("restoreRate", 50)
        if available < 100 and restore_rate > 0:
            wait_s = (100 - available) / restore_rate
            logger.debug(f"Shopify rate limit: waiting {wait_s:.1f}s")
            await asyncio.sleep(wait_s)

        return result.get("data", {})

    # ── Single item fetches (for webhooks) ──────────────────────

    async def fetch_product(self, shopify_gid: str) -> Optional[Dict[str, Any]]:
        """Fetch a single product by GID with full details."""
        data = await self.query("""
            query GetProduct($id: ID!) {
                product(id: $id) {
                    id title descriptionHtml handle vendor productType tags status
                    createdAt updatedAt
                    seo { title description }
                    images(first: 10) { edges { node { url } } }
                    variants(first: 100) {
                        edges {
                            node {
                                id title sku barcode
                                price compareAtPrice
                                inventoryQuantity availableForSale
                                selectedOptions { name value }
                            }
                        }
                    }
                    collections(first: 20) { edges { node { id } } }
                    metafields(first: 10) { edges { node { namespace key value type } } }
                }
            }
        """, {"id": shopify_gid})
        return data.get("product")

    async def fetch_collection(self, shopify_gid: str) -> Optional[Dict[str, Any]]:
        """Fetch a single collection by GID."""
        data = await self.query("""
            query GetCollection($id: ID!) {
                collection(id: $id) {
                    id title descriptionHtml handle updatedAt
                    seo { title description }
                    image { url }
                    productsCount { count }
                    ruleSet { rules { column relation condition } }
                }
            }
        """, {"id": shopify_gid})
        return data.get("collection")

    # ── Paginated incremental fetches ───────────────────────────

    async def fetch_products_updated_since(self, since: str, limit: int = 250) -> List[Dict[str, Any]]:
        """Fetch products updated after ISO timestamp, paginated."""
        products = []
        cursor = None
        has_next = True

        while has_next and len(products) < limit:
            batch_size = min(50, limit - len(products))
            data = await self.query("""
                query ProductsUpdatedSince($query: String!, $first: Int!, $after: String) {
                    products(query: $query, first: $first, after: $after, sortKey: UPDATED_AT) {
                        edges {
                            cursor
                            node {
                                id title descriptionHtml handle vendor productType tags status
                                createdAt updatedAt
                                seo { title description }
                                images(first: 10) { edges { node { url } } }
                                variants(first: 100) {
                                    edges {
                                        node {
                                            id title sku barcode price compareAtPrice
                                            inventoryQuantity availableForSale
                                            selectedOptions { name value }
                                        }
                                    }
                                }
                                collections(first: 20) { edges { node { id } } }
                                metafields(first: 10) { edges { node { namespace key value type } } }
                            }
                        }
                        pageInfo { hasNextPage }
                    }
                }
            """, {
                "query": f"updated_at:>'{since}'",
                "first": batch_size,
                "after": cursor,
            })

            edges = data.get("products", {}).get("edges", [])
            for edge in edges:
                products.append(edge["node"])
                cursor = edge["cursor"]
            has_next = data.get("products", {}).get("pageInfo", {}).get("hasNextPage", False)

        logger.info(f"Fetched {len(products)} products updated since {since}")
        return products

    async def fetch_collections_updated_since(self, since: str, limit: int = 250) -> List[Dict[str, Any]]:
        """Fetch collections updated after ISO timestamp, paginated."""
        collections = []
        cursor = None
        has_next = True

        while has_next and len(collections) < limit:
            batch_size = min(50, limit - len(collections))
            data = await self.query("""
                query CollectionsUpdatedSince($query: String!, $first: Int!, $after: String) {
                    collections(query: $query, first: $first, after: $after, sortKey: UPDATED_AT) {
                        edges {
                            cursor
                            node {
                                id title descriptionHtml handle updatedAt
                                seo { title description }
                                image { url }
                                productsCount { count }
                            }
                        }
                        pageInfo { hasNextPage }
                    }
                }
            """, {
                "query": f"updated_at:>'{since}'",
                "first": batch_size,
                "after": cursor,
            })

            edges = data.get("collections", {}).get("edges", [])
            for edge in edges:
                collections.append(edge["node"])
                cursor = edge["cursor"]
            has_next = data.get("collections", {}).get("pageInfo", {}).get("hasNextPage", False)

        logger.info(f"Fetched {len(collections)} collections updated since {since}")
        return collections

    # ── Bulk Operations ─────────────────────────────────────────

    async def start_products_bulk_query(self) -> str:
        """Start async bulk export of all products. Returns operation ID."""
        data = await self.query('''
            mutation BulkProducts {
                bulkOperationRunQuery(query: """
                    {
                        products {
                            edges {
                                node {
                                    id title descriptionHtml handle vendor productType tags status
                                    createdAt updatedAt
                                    seo { title description }
                                    images(first: 10) { edges { node { url } } }
                                    variants(first: 100) {
                                        edges {
                                            node {
                                                id title sku barcode
                                                price compareAtPrice
                                                inventoryQuantity availableForSale
                                                selectedOptions { name value }
                                            }
                                        }
                                    }
                                    collections(first: 20) { edges { node { id } } }
                                    metafields(first: 10) { edges { node { namespace key value type } } }
                                }
                            }
                        }
                    }
                """) {
                    bulkOperation { id }
                    userErrors { field message }
                }
            }
        ''')

        result = data.get("bulkOperationRunQuery", {})
        errors = result.get("userErrors", [])
        if errors:
            raise Exception(f"Bulk operation errors: {', '.join(e['message'] for e in errors)}")
        op = result.get("bulkOperation")
        if not op:
            raise Exception("Bulk operation was not created")

        logger.info(f"Started bulk products query: {op['id']}")
        return op["id"]

    async def start_collections_bulk_query(self) -> str:
        """Start async bulk export of all collections. Returns operation ID."""
        data = await self.query('''
            mutation BulkCollections {
                bulkOperationRunQuery(query: """
                    {
                        collections {
                            edges {
                                node {
                                    id title descriptionHtml handle updatedAt
                                    seo { title description }
                                    image { url }
                                    productsCount { count }
                                }
                            }
                        }
                    }
                """) {
                    bulkOperation { id }
                    userErrors { field message }
                }
            }
        ''')

        result = data.get("bulkOperationRunQuery", {})
        errors = result.get("userErrors", [])
        if errors:
            raise Exception(f"Bulk operation errors: {', '.join(e['message'] for e in errors)}")
        op = result.get("bulkOperation")
        if not op:
            raise Exception("Bulk operation was not created")

        logger.info(f"Started bulk collections query: {op['id']}")
        return op["id"]

    async def start_pages_bulk_query(self) -> str:
        """Start async bulk export of all pages. Returns operation ID."""
        data = await self.query('''
            mutation BulkPages {
                bulkOperationRunQuery(query: """
                    {
                        pages {
                            edges {
                                node {
                                    id title body handle createdAt updatedAt
                                }
                            }
                        }
                    }
                """) {
                    bulkOperation { id }
                    userErrors { field message }
                }
            }
        ''')

        result = data.get("bulkOperationRunQuery", {})
        errors = result.get("userErrors", [])
        if errors:
            raise Exception(f"Bulk operation errors: {', '.join(e['message'] for e in errors)}")
        op = result.get("bulkOperation")
        if not op:
            raise Exception("Bulk operation was not created")

        logger.info(f"Started bulk pages query: {op['id']}")
        return op["id"]

    async def poll_until_complete(self, operation_id: str, interval_s: float = 5.0) -> Dict[str, Any]:
        """Poll a bulk operation until it completes."""
        while True:
            data = await self.query("""
                query PollBulkOp($id: ID!) {
                    node(id: $id) {
                        ... on BulkOperation {
                            id status url objectCount errorCode
                        }
                    }
                }
            """, {"id": operation_id})

            op = data.get("node")
            if not op:
                raise Exception(f"Bulk operation {operation_id} not found")

            logger.debug(f"Bulk op {op['id']}: status={op['status']}, objects={op.get('objectCount', 0)}")

            if op["status"] == "COMPLETED":
                return {
                    "id": op["id"],
                    "status": op["status"],
                    "url": op.get("url"),
                    "objectCount": int(op.get("objectCount") or 0),
                    "errorCode": op.get("errorCode"),
                }

            if op["status"] in ("FAILED", "CANCELED"):
                raise Exception(f"Bulk operation {op['id']} {op['status']}: {op.get('errorCode')}")

            await asyncio.sleep(interval_s)

    async def download_results(self, url: str) -> List[Dict[str, Any]]:
        """Download JSONL results from a completed bulk operation."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        lines = resp.text.strip().split("\n")
        results = [json.loads(line) for line in lines if line.strip()]
        logger.info(f"Downloaded {len(results)} items from bulk operation")
        return results

    # ── Utility ────────────────────────────────────────────────

    @staticmethod
    def flatten_graphql_product(node: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a GraphQL product node into a REST-like dict compatible with ShopifyClient.extract_*."""
        images = [e["node"]["url"] for e in node.get("images", {}).get("edges", [])]
        variants = []
        for ve in node.get("variants", {}).get("edges", []):
            v = ve["node"]
            variants.append({
                "id": v.get("id"),
                "title": v.get("title"),
                "sku": v.get("sku", ""),
                "barcode": v.get("barcode", ""),
                "price": v.get("price", "0"),
                "compare_at_price": v.get("compareAtPrice"),
                "inventory_quantity": v.get("inventoryQuantity", 0),
                "available": v.get("availableForSale", True),
                "options": v.get("selectedOptions", []),
            })

        collection_ids = [e["node"]["id"] for e in node.get("collections", {}).get("edges", [])]
        metafields = {}
        for me in node.get("metafields", {}).get("edges", []):
            mf = me["node"]
            metafields[f"{mf['namespace']}.{mf['key']}"] = mf.get("value", "")

        tags = node.get("tags", "")
        if isinstance(tags, list):
            tags = ", ".join(tags)

        return {
            "id": node.get("id"),
            "title": node.get("title", ""),
            "body_html": node.get("descriptionHtml", ""),
            "handle": node.get("handle", ""),
            "vendor": node.get("vendor", ""),
            "product_type": node.get("productType", ""),
            "tags": tags,
            "status": node.get("status", "ACTIVE").lower(),
            "created_at": node.get("createdAt", ""),
            "updated_at": node.get("updatedAt", ""),
            "image": {"src": images[0]} if images else None,
            "images": [{"src": u} for u in images],
            "variants": variants,
            "collection_ids": collection_ids,
            "metafields": metafields,
            "seo_title": node.get("seo", {}).get("title", ""),
            "seo_description": node.get("seo", {}).get("description", ""),
        }

    @staticmethod
    def flatten_graphql_collection(node: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a GraphQL collection node."""
        image = node.get("image", {}) or {}
        return {
            "id": node.get("id"),
            "title": node.get("title", ""),
            "body_html": node.get("descriptionHtml", ""),
            "handle": node.get("handle", ""),
            "updated_at": node.get("updatedAt", ""),
            "image_url": image.get("url", ""),
            "products_count": node.get("productsCount", {}).get("count", 0),
            "seo_title": node.get("seo", {}).get("title", ""),
            "seo_description": node.get("seo", {}).get("description", ""),
            "rules": node.get("ruleSet", {}).get("rules", []) if node.get("ruleSet") else [],
        }

    @staticmethod
    def flatten_graphql_page(node: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a GraphQL page node."""
        return {
            "id": node.get("id"),
            "title": node.get("title", ""),
            "body_html": node.get("body", ""),
            "handle": node.get("handle", ""),
            "created_at": node.get("createdAt", ""),
            "updated_at": node.get("updatedAt", ""),
        }

    # ── Order queries ───────────────────────────────────────────

    async def fetch_orders(
        self, query: Optional[str] = None, first: int = 20, status: str = "any"
    ) -> List[Dict[str, Any]]:
        """Fetch orders with optional search query and status filter."""
        gql_query = f"query ListOrders($first: Int!, $query: String) {{"
        gql_query += """
            orders(first: $first, query: $query, sortKey: CREATED_AT, reverse: true) {
                edges {
                    node {
                        id name email createdAt
                        displayFinancialStatus displayFulfillmentStatus
                        totalPriceSet { shopMoney { amount currencyCode } }
                        subtotalPriceSet { shopMoney { amount currencyCode } }
                        totalShippingPriceSet { shopMoney { amount currencyCode } }
                        totalTaxSet { shopMoney { amount currencyCode } }
                        customer { firstName lastName email }
                        shippingAddress { address1 city province country zip }
                        lineItems(first: 20) {
                            edges {
                                node {
                                    title quantity
                                    originalUnitPriceSet { shopMoney { amount currencyCode } }
                                    sku
                                    variant { title }
                                }
                            }
                        }
                        fulfillments {
                            status
                            trackingInfo { number url company }
                            createdAt
                        }
                        note tags cancelledAt closedAt
                    }
                }
            }
        }"""

        parts = []
        if query:
            parts.append(query)
        if status and status != "any":
            parts.append(f"status:{status}")
        combined_query = " ".join(parts) if parts else None

        variables: Dict[str, Any] = {"first": first}
        if combined_query:
            variables["query"] = combined_query

        data = await self.query(gql_query, variables)
        orders = data.get("orders", {}).get("edges", [])
        return [self.flatten_graphql_order(e["node"]) for e in orders]

    async def fetch_order(self, id_or_name: str) -> Optional[Dict[str, Any]]:
        """Fetch a single order by GID or order name (#1234)."""
        if id_or_name.startswith("gid://"):
            gid = id_or_name
        elif id_or_name.startswith("#"):
            # Search by name
            orders = await self.fetch_orders(query=f"name:{id_or_name}", first=1)
            return orders[0] if orders else None
        else:
            # Try as order number
            orders = await self.fetch_orders(query=f"name:#{id_or_name}", first=1)
            return orders[0] if orders else None

        data = await self.query("""
            query GetOrder($id: ID!) {
                order(id: $id) {
                    id name email createdAt
                    displayFinancialStatus displayFulfillmentStatus
                    totalPriceSet { shopMoney { amount currencyCode } }
                    subtotalPriceSet { shopMoney { amount currencyCode } }
                    totalShippingPriceSet { shopMoney { amount currencyCode } }
                    totalTaxSet { shopMoney { amount currencyCode } }
                    customer { firstName lastName email phone }
                    shippingAddress { address1 address2 city province country zip phone }
                    billingAddress { address1 address2 city province country zip }
                    lineItems(first: 50) {
                        edges {
                            node {
                                title quantity sku
                                originalUnitPriceSet { shopMoney { amount currencyCode } }
                                variant { title }
                            }
                        }
                    }
                    fulfillments {
                        status
                        trackingInfo { number url company }
                        createdAt updatedAt
                    }
                    note tags cancelledAt closedAt
                    paymentGatewayNames
                    refunds(first: 10) {
                        id createdAt
                        totalRefundedSet { shopMoney { amount currencyCode } }
                        note
                    }
                }
            }
        """, {"id": gid})
        node = data.get("order")
        return self.flatten_graphql_order(node) if node else None

    @staticmethod
    def flatten_graphql_order(node: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a GraphQL order node into a clean dict."""
        def money(field):
            m = node.get(field, {}) or {}
            s = m.get("shopMoney", {}) or {}
            return {"amount": s.get("amount", "0.00"), "currency": s.get("currencyCode", "EUR")}

        customer = node.get("customer") or {}
        shipping = node.get("shippingAddress") or {}

        line_items = []
        for edge in (node.get("lineItems", {}).get("edges", [])):
            li = edge.get("node", {})
            price = li.get("originalUnitPriceSet", {}).get("shopMoney", {})
            variant = li.get("variant") or {}
            line_items.append({
                "title": li.get("title", ""),
                "variant": variant.get("title", ""),
                "quantity": li.get("quantity", 0),
                "sku": li.get("sku", ""),
                "unit_price": price.get("amount", "0.00"),
            })

        fulfillments = []
        for f in (node.get("fulfillments") or []):
            tracking = f.get("trackingInfo", []) or []
            fulfillments.append({
                "status": f.get("status", ""),
                "created_at": f.get("createdAt", ""),
                "tracking": [{"number": t.get("number"), "url": t.get("url"), "company": t.get("company")} for t in tracking],
            })

        refunds = []
        for r in (node.get("refunds") or []):
            ref_money = r.get("totalRefundedSet", {}).get("shopMoney", {})
            refunds.append({
                "id": r.get("id"),
                "created_at": r.get("createdAt"),
                "amount": ref_money.get("amount", "0.00"),
                "note": r.get("note", ""),
            })

        return {
            "id": node.get("id"),
            "name": node.get("name", ""),
            "email": node.get("email", ""),
            "created_at": node.get("createdAt", ""),
            "financial_status": node.get("displayFinancialStatus", ""),
            "fulfillment_status": node.get("displayFulfillmentStatus", ""),
            "total": money("totalPriceSet"),
            "subtotal": money("subtotalPriceSet"),
            "shipping_cost": money("totalShippingPriceSet"),
            "tax": money("totalTaxSet"),
            "customer": {
                "name": f"{customer.get('firstName', '')} {customer.get('lastName', '')}".strip(),
                "email": customer.get("email", ""),
                "phone": customer.get("phone", ""),
            },
            "shipping_address": {
                "address": shipping.get("address1", ""),
                "city": shipping.get("city", ""),
                "province": shipping.get("province", ""),
                "country": shipping.get("country", ""),
                "zip": shipping.get("zip", ""),
            },
            "line_items": line_items,
            "fulfillments": fulfillments,
            "refunds": refunds,
            "note": node.get("note", ""),
            "tags": node.get("tags", []),
            "cancelled_at": node.get("cancelledAt"),
            "closed_at": node.get("closedAt"),
            "payment_gateways": node.get("paymentGatewayNames", []),
        }
