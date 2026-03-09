"""Test Picqer MCP tool functions route correctly to PicqerClient."""

import json
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def mock_api():
    """Create a mock PicqerClient and patch it into picqer_server."""
    mock = AsyncMock()
    mock.get = AsyncMock(return_value={"mocked": True})
    mock.post = AsyncMock(return_value={"mocked": True})
    mock.put = AsyncMock(return_value={"mocked": True})
    mock.delete = AsyncMock(return_value={"status": "deleted"})
    mock.get_list = AsyncMock(return_value=[{"id": 1}])
    with patch("picqer_server.api", mock):
        yield mock


# ═══════════════════════════════════════════════════════════════════════════
#  PRODUCTS
# ═══════════════════════════════════════════════════════════════════════════


class TestProductTools:
    @pytest.mark.asyncio
    async def test_list_products(self, mock_api):
        from picqer_server import picqer_products

        result = await picqer_products("list")
        mock_api.get_list.assert_called_once_with("/products", {})

    @pytest.mark.asyncio
    async def test_get_product(self, mock_api):
        from picqer_server import picqer_products

        result = await picqer_products("get", id=42)
        mock_api.get.assert_called_once_with("/products/42")

    @pytest.mark.asyncio
    async def test_create_product(self, mock_api):
        from picqer_server import picqer_products

        params = json.dumps({
            "productcode": "SKU-001",
            "name": "Test Product",
            "price": 19.99,
            "idvatgroup": 1,
        })
        result = await picqer_products("create", params=params)
        mock_api.post.assert_called_once_with("/products", {
            "productcode": "SKU-001",
            "name": "Test Product",
            "price": 19.99,
            "idvatgroup": 1,
        })

    @pytest.mark.asyncio
    async def test_update_product(self, mock_api):
        from picqer_server import picqer_products

        result = await picqer_products("update", id=1, params='{"name": "Updated"}')
        mock_api.put.assert_called_once_with("/products/1", {"name": "Updated"})

    @pytest.mark.asyncio
    async def test_activate_product(self, mock_api):
        from picqer_server import picqer_products

        result = await picqer_products("activate", id=1)
        mock_api.post.assert_called_once_with("/products/1/activate")

    @pytest.mark.asyncio
    async def test_deactivate_product(self, mock_api):
        from picqer_server import picqer_products

        result = await picqer_products("deactivate", id=1)
        mock_api.post.assert_called_once_with("/products/1/inactivate")

    @pytest.mark.asyncio
    async def test_unknown_action(self, mock_api):
        from picqer_server import picqer_products

        result = await picqer_products("bogus")
        assert "Unknown action" in result


# ═══════════════════════════════════════════════════════════════════════════
#  ORDERS
# ═══════════════════════════════════════════════════════════════════════════


class TestOrderTools:
    @pytest.mark.asyncio
    async def test_list_orders(self, mock_api):
        from picqer_server import picqer_orders

        result = await picqer_orders("list", params='{"status": "open"}')
        mock_api.get_list.assert_called_once_with("/orders", {"status": "open"})

    @pytest.mark.asyncio
    async def test_get_order(self, mock_api):
        from picqer_server import picqer_orders

        result = await picqer_orders("get", id=10)
        mock_api.get.assert_called_once_with("/orders/10")

    @pytest.mark.asyncio
    async def test_create_order(self, mock_api):
        from picqer_server import picqer_orders

        params = json.dumps({
            "idcustomer": 5,
            "products": [{"idproduct": 1, "amount": 2}],
        })
        result = await picqer_orders("create", params=params)
        mock_api.post.assert_called_once_with("/orders", {
            "idcustomer": 5,
            "products": [{"idproduct": 1, "amount": 2}],
        })

    @pytest.mark.asyncio
    async def test_process_order(self, mock_api):
        from picqer_server import picqer_order_actions

        result = await picqer_order_actions("process", order_id=10)
        mock_api.post.assert_called_once_with("/orders/10/process")

    @pytest.mark.asyncio
    async def test_pause_order(self, mock_api):
        from picqer_server import picqer_order_actions

        result = await picqer_order_actions(
            "pause", order_id=10, params='{"reason": "waiting"}'
        )
        mock_api.post.assert_called_once_with(
            "/orders/10/pause", {"reason": "waiting"}
        )


# ═══════════════════════════════════════════════════════════════════════════
#  STOCK
# ═══════════════════════════════════════════════════════════════════════════


class TestStockTools:
    @pytest.mark.asyncio
    async def test_get_stock(self, mock_api):
        from picqer_server import picqer_product_stock

        result = await picqer_product_stock("get", product_id=5)
        mock_api.get.assert_called_once_with("/products/5/stock")

    @pytest.mark.asyncio
    async def test_change_stock(self, mock_api):
        from picqer_server import picqer_product_stock

        result = await picqer_product_stock(
            "change",
            product_id=5,
            warehouse_id=1,
            params='{"change": 10, "reason": "restock"}',
        )
        mock_api.post.assert_called_once_with(
            "/products/5/stock/1", {"change": 10, "reason": "restock"}
        )

    @pytest.mark.asyncio
    async def test_move_stock(self, mock_api):
        from picqer_server import picqer_product_stock

        result = await picqer_product_stock(
            "move",
            product_id=5,
            warehouse_id=1,
            params='{"from_idlocation": 10, "to_idlocation": 20, "amount": 3}',
        )
        mock_api.post.assert_called_once()
        call_args = mock_api.post.call_args
        assert "/move" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_stock_history(self, mock_api):
        from picqer_server import picqer_product_stock

        result = await picqer_product_stock(
            "history", params='{"sincedate": "2026-01-01"}'
        )
        mock_api.get_list.assert_called_once_with(
            "/stockhistory", {"sincedate": "2026-01-01"}
        )


# ═══════════════════════════════════════════════════════════════════════════
#  CUSTOMERS
# ═══════════════════════════════════════════════════════════════════════════


class TestCustomerTools:
    @pytest.mark.asyncio
    async def test_list_customers(self, mock_api):
        from picqer_server import picqer_customers

        result = await picqer_customers("list")
        mock_api.get_list.assert_called_once_with("/customers", {})

    @pytest.mark.asyncio
    async def test_create_customer(self, mock_api):
        from picqer_server import picqer_customers

        result = await picqer_customers("create", params='{"name": "Test"}')
        mock_api.post.assert_called_once_with("/customers", {"name": "Test"})


# ═══════════════════════════════════════════════════════════════════════════
#  PICKLISTS
# ═══════════════════════════════════════════════════════════════════════════


class TestPicklistTools:
    @pytest.mark.asyncio
    async def test_list_picklists(self, mock_api):
        from picqer_server import picqer_picklists

        result = await picqer_picklists("list")
        mock_api.get_list.assert_called_once_with("/picklists", {})

    @pytest.mark.asyncio
    async def test_close_picklist(self, mock_api):
        from picqer_server import picqer_picklists

        result = await picqer_picklists("close", id=1)
        mock_api.post.assert_called_once_with("/picklists/1/close")

    @pytest.mark.asyncio
    async def test_pick_product(self, mock_api):
        from picqer_server import picqer_picklists

        result = await picqer_picklists(
            "pick", id=1, params='{"idpicklist_product": 5, "amount": 2}'
        )
        mock_api.post.assert_called_once_with(
            "/picklists/1/pick", {"idpicklist_product": 5, "amount": 2}
        )

    @pytest.mark.asyncio
    async def test_assign_picklist(self, mock_api):
        from picqer_server import picqer_picklists

        result = await picqer_picklists("assign", id=1, params='{"iduser": 7}')
        mock_api.post.assert_called_once_with(
            "/picklists/1/assign", {"iduser": 7}
        )


# ═══════════════════════════════════════════════════════════════════════════
#  WAREHOUSES
# ═══════════════════════════════════════════════════════════════════════════


class TestWarehouseTools:
    @pytest.mark.asyncio
    async def test_list_warehouses(self, mock_api):
        from picqer_server import picqer_warehouses

        result = await picqer_warehouses("list")
        mock_api.get.assert_called_once_with("/warehouses")

    @pytest.mark.asyncio
    async def test_get_warehouse_stock(self, mock_api):
        from picqer_server import picqer_warehouses

        result = await picqer_warehouses("stock", id=1)
        mock_api.get_list.assert_called_once_with("/warehouses/1/stock", {})


# ═══════════════════════════════════════════════════════════════════════════
#  WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════


class TestWebhookTools:
    @pytest.mark.asyncio
    async def test_create_webhook(self, mock_api):
        from picqer_server import picqer_webhooks

        params = json.dumps({
            "name": "Order hook",
            "event": "orders.created",
            "address": "https://example.com/hook",
        })
        result = await picqer_webhooks("create", params=params)
        mock_api.post.assert_called_once_with("/hooks", {
            "name": "Order hook",
            "event": "orders.created",
            "address": "https://example.com/hook",
        })

    @pytest.mark.asyncio
    async def test_delete_webhook(self, mock_api):
        from picqer_server import picqer_webhooks

        result = await picqer_webhooks("delete", id=99)
        mock_api.delete.assert_called_once_with("/hooks/99")


# ═══════════════════════════════════════════════════════════════════════════
#  STATS
# ═══════════════════════════════════════════════════════════════════════════


class TestStatsTools:
    @pytest.mark.asyncio
    async def test_list_stats(self, mock_api):
        from picqer_server import picqer_stats

        result = await picqer_stats("list")
        mock_api.get.assert_called_once_with("/stats")

    @pytest.mark.asyncio
    async def test_get_stat(self, mock_api):
        from picqer_server import picqer_stats

        result = await picqer_stats("get", key="open-orders")
        mock_api.get.assert_called_once_with("/stats/open-orders")


# ═══════════════════════════════════════════════════════════════════════════
#  GENERIC API
# ═══════════════════════════════════════════════════════════════════════════


class TestGenericApiTool:
    @pytest.mark.asyncio
    async def test_generic_get(self, mock_api):
        from picqer_server import picqer_api

        result = await picqer_api("GET", "/custom/endpoint", params='{"foo": "bar"}')
        mock_api.get.assert_called_once_with("/custom/endpoint", {"foo": "bar"})

    @pytest.mark.asyncio
    async def test_generic_post(self, mock_api):
        from picqer_server import picqer_api

        result = await picqer_api("POST", "/custom/endpoint", params='{"data": 1}')
        mock_api.post.assert_called_once_with("/custom/endpoint", {"data": 1})

    @pytest.mark.asyncio
    async def test_generic_unsupported_method(self, mock_api):
        from picqer_server import picqer_api

        result = await picqer_api("PATCH", "/custom/endpoint")
        assert "Unsupported" in result
