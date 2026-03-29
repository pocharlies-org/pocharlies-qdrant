# Picqer MCP Server — TDD, Best Practices & RAG Integration

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the Picqer MCP server with proper test coverage, clean architecture, and integrate it into the existing docker-compose stack alongside the RAG MCP server so the agent-service can call both.

**Architecture:** Extract the PicqerAPI client into its own module (`picqer_client.py`) so it's testable independently. Add pytest with httpx mocking (respx) for unit tests — no real API calls in CI. Add the Picqer MCP as a second service in docker-compose and register it in the agent-service's `mcp_servers.json`. Move the hardcoded API key to `.env`.

**Tech Stack:** Python 3.11, FastMCP, httpx, pytest, pytest-asyncio, respx (httpx mock), Docker Compose

---

## Wave 1: Test Infrastructure + Client Extraction (tasks 1–6)

### Task 1: Add test dependencies

**Files:**
- Modify: `mcp-server/requirements.txt`
- Create: `mcp-server/requirements-dev.txt`
- Create: `mcp-server/pyproject.toml`

**Step 1: Create requirements-dev.txt**

```
-r requirements.txt
pytest>=8.0
pytest-asyncio>=0.24
respx>=0.22
```

**Step 2: Create pyproject.toml for pytest config**

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
```

**Step 3: Install dev deps in the venv**

Run: `cd mcp-server && source .venv/bin/activate && pip install -r requirements-dev.txt`
Expected: All packages install cleanly

**Step 4: Verify pytest runs (no tests yet)**

Run: `cd mcp-server && source .venv/bin/activate && python -m pytest --co -q`
Expected: "no tests ran" (exit 5 is OK — means pytest works but found nothing)

**Step 5: Commit**

```bash
git add mcp-server/requirements-dev.txt mcp-server/pyproject.toml
git commit -m "chore: add test infrastructure (pytest, respx, pytest-asyncio)"
```

---

### Task 2: Extract PicqerClient into its own module

**Files:**
- Create: `mcp-server/picqer_client.py`
- Modify: `mcp-server/picqer_server.py`

**Step 1: Create `picqer_client.py`**

Extract the `PicqerAPI` class and config into a standalone module. The class should accept config via constructor (not globals) so it's testable with any base URL / API key.

```python
"""
Picqer API v1 async client.

Handles authentication, pagination, and error wrapping.
Designed for dependency injection — accepts config via constructor.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx

logger = logging.getLogger("picqer-client")


@dataclass(frozen=True)
class PicqerConfig:
    """Immutable Picqer connection config."""
    subdomain: str
    api_key: str
    user_agent: str = "SkirmshopMCP/1.0 (mcp@skirmshop.es)"
    timeout: float = 30.0

    @property
    def base_url(self) -> str:
        return f"https://{self.subdomain}.picqer.com/api/v1"


class PicqerClient:
    """Async HTTP client for Picqer API v1 with Basic auth and pagination."""

    def __init__(self, config: PicqerConfig):
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                auth=(self._config.api_key, "x"),
                headers={
                    "User-Agent": self._config.user_agent,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=self._config.timeout,
            )
        return self._client

    async def close(self):
        """Explicitly close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request(self, method: str, path: str, params=None, data=None) -> Any:
        try:
            r = await self.http.request(method, path, params=params, json=data)
            r.raise_for_status()
            if r.status_code == 204 or not r.content:
                return {"status": "ok"}
            return r.json()
        except httpx.HTTPStatusError as e:
            body = e.response.text
            try:
                body = e.response.json()
            except Exception:
                pass
            return {"error": True, "status_code": e.response.status_code, "detail": body}
        except httpx.RequestError as e:
            return {"error": True, "detail": str(e)}

    async def get(self, path: str, params: dict = None) -> Any:
        return await self._request("GET", path, params=params)

    async def post(self, path: str, data: dict = None) -> Any:
        return await self._request("POST", path, data=data)

    async def put(self, path: str, data: dict = None) -> Any:
        return await self._request("PUT", path, data=data)

    async def delete(self, path: str, params: dict = None) -> Any:
        return await self._request("DELETE", path, params=params)

    async def get_list(self, path: str, params: dict = None, max_results: int = 500) -> Any:
        """Paginated GET — fetches up to max_results items (100 per page)."""
        results = []
        offset = 0
        p = dict(params or {})
        while len(results) < max_results:
            p["offset"] = offset
            batch = await self._request("GET", path, params=p)
            if isinstance(batch, dict) and batch.get("error"):
                return batch
            if not isinstance(batch, list) or not batch:
                break
            results.extend(batch)
            if len(batch) < 100:
                break
            offset += 100
        return results[:max_results]
```

**Step 2: Update `picqer_server.py` to import from `picqer_client`**

Replace the inline `PicqerAPI` class and config block (lines 23–101) with:

```python
import os
from picqer_client import PicqerClient, PicqerConfig

# ── Configuration ──────────────────────────────────────────────────────────
_config = PicqerConfig(
    subdomain=os.getenv("PICQER_SUBDOMAIN", "skirmshop"),
    api_key=os.getenv("PICQER_API_KEY", ""),
)
api = PicqerClient(_config)
```

Remove the hardcoded API key default — it moves to `.env` (Task 7).

Keep every tool function unchanged — they already call `api.get()`, `api.post()`, etc.

**Step 3: Verify the server still loads**

Run: `cd mcp-server && python -c "from picqer_server import mcp; print(f'{len(mcp._tool_manager.list_tools())} tools')"`
Expected: `43 tools`

**Step 4: Commit**

```bash
git add mcp-server/picqer_client.py mcp-server/picqer_server.py
git commit -m "refactor: extract PicqerClient into testable module with DI config"
```

---

### Task 3: Write unit tests for PicqerClient — GET operations

**Files:**
- Create: `mcp-server/tests/__init__.py`
- Create: `mcp-server/tests/conftest.py`
- Create: `mcp-server/tests/test_picqer_client.py`

**Step 1: Create test scaffolding**

`tests/__init__.py` — empty file

`tests/conftest.py`:

```python
import pytest
from picqer_client import PicqerClient, PicqerConfig

TEST_CONFIG = PicqerConfig(
    subdomain="testshop",
    api_key="test-api-key-123",
    timeout=5.0,
)


@pytest.fixture
def config():
    return TEST_CONFIG


@pytest.fixture
async def client(config):
    c = PicqerClient(config)
    yield c
    await c.close()
```

**Step 2: Write failing tests for GET, error handling, and pagination**

`tests/test_picqer_client.py`:

```python
import httpx
import respx
import pytest
from picqer_client import PicqerClient, PicqerConfig

BASE = "https://testshop.picqer.com/api/v1"


class TestPicqerConfig:
    def test_base_url(self, config):
        assert config.base_url == BASE

    def test_config_is_immutable(self, config):
        with pytest.raises(AttributeError):
            config.subdomain = "other"


class TestGet:
    @respx.mock
    async def test_get_single_resource(self, client):
        respx.get(f"{BASE}/products/1").mock(
            return_value=httpx.Response(200, json={"idproduct": 1, "name": "Widget"})
        )
        result = await client.get("/products/1")
        assert result == {"idproduct": 1, "name": "Widget"}

    @respx.mock
    async def test_get_list_resource(self, client):
        respx.get(f"{BASE}/products").mock(
            return_value=httpx.Response(200, json=[{"idproduct": 1}, {"idproduct": 2}])
        )
        result = await client.get("/products")
        assert isinstance(result, list)
        assert len(result) == 2

    @respx.mock
    async def test_get_with_params(self, client):
        route = respx.get(f"{BASE}/products").mock(
            return_value=httpx.Response(200, json=[])
        )
        await client.get("/products", {"search": "widget", "inactive": "true"})
        assert route.called
        assert "search=widget" in str(route.calls[0].request.url)

    @respx.mock
    async def test_get_404_returns_error_dict(self, client):
        respx.get(f"{BASE}/products/999").mock(
            return_value=httpx.Response(404, json={"error": "Not found"})
        )
        result = await client.get("/products/999")
        assert result["error"] is True
        assert result["status_code"] == 404

    @respx.mock
    async def test_get_500_returns_error_dict(self, client):
        respx.get(f"{BASE}/warehouses").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        result = await client.get("/warehouses")
        assert result["error"] is True
        assert result["status_code"] == 500

    @respx.mock
    async def test_get_204_returns_ok(self, client):
        respx.get(f"{BASE}/something").mock(
            return_value=httpx.Response(204)
        )
        result = await client.get("/something")
        assert result == {"status": "ok"}


class TestGetList:
    @respx.mock
    async def test_single_page(self, client):
        respx.get(f"{BASE}/products").mock(
            return_value=httpx.Response(200, json=[{"id": i} for i in range(50)])
        )
        result = await client.get_list("/products")
        assert len(result) == 50

    @respx.mock
    async def test_pagination_two_pages(self, client):
        page1 = [{"id": i} for i in range(100)]
        page2 = [{"id": i} for i in range(100, 130)]

        route = respx.get(f"{BASE}/orders").mock(
            side_effect=[
                httpx.Response(200, json=page1),
                httpx.Response(200, json=page2),
            ]
        )
        result = await client.get_list("/orders")
        assert len(result) == 130
        assert route.call_count == 2

    @respx.mock
    async def test_pagination_respects_max_results(self, client):
        page1 = [{"id": i} for i in range(100)]

        respx.get(f"{BASE}/products").mock(
            return_value=httpx.Response(200, json=page1)
        )
        result = await client.get_list("/products", max_results=50)
        assert len(result) == 50

    @respx.mock
    async def test_pagination_error_on_second_page(self, client):
        page1 = [{"id": i} for i in range(100)]

        respx.get(f"{BASE}/products").mock(
            side_effect=[
                httpx.Response(200, json=page1),
                httpx.Response(500, text="boom"),
            ]
        )
        result = await client.get_list("/products")
        # Should return error from second page
        assert isinstance(result, dict)
        assert result["error"] is True

    @respx.mock
    async def test_empty_list(self, client):
        respx.get(f"{BASE}/backorders").mock(
            return_value=httpx.Response(200, json=[])
        )
        result = await client.get_list("/backorders")
        assert result == []
```

**Step 3: Run tests to verify they fail**

Run: `cd mcp-server && source .venv/bin/activate && python -m pytest tests/test_picqer_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'picqer_client'` (if Task 2 not done yet) or all PASS if Task 2 is complete.

**Step 4: Make tests pass**

If Task 2 is done, all tests should pass. Fix any mismatches.

Run: `cd mcp-server && source .venv/bin/activate && python -m pytest tests/test_picqer_client.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add mcp-server/tests/
git commit -m "test: add PicqerClient unit tests — GET, pagination, error handling"
```

---

### Task 4: Write unit tests for PicqerClient — POST/PUT/DELETE

**Files:**
- Modify: `mcp-server/tests/test_picqer_client.py`

**Step 1: Add write-operation tests**

Append to `test_picqer_client.py`:

```python
class TestPost:
    @respx.mock
    async def test_post_creates_resource(self, client):
        respx.post(f"{BASE}/products").mock(
            return_value=httpx.Response(201, json={"idproduct": 42, "name": "New"})
        )
        result = await client.post("/products", {"name": "New", "productcode": "NW1", "price": 10, "idvatgroup": 1})
        assert result["idproduct"] == 42

    @respx.mock
    async def test_post_sends_json_body(self, client):
        route = respx.post(f"{BASE}/orders").mock(
            return_value=httpx.Response(200, json={"idorder": 1})
        )
        await client.post("/orders", {"idcustomer": 5, "products": [{"idproduct": 1, "amount": 2}]})
        body = route.calls[0].request.content
        import json
        parsed = json.loads(body)
        assert parsed["idcustomer"] == 5

    @respx.mock
    async def test_post_action_no_body(self, client):
        respx.post(f"{BASE}/orders/1/process").mock(
            return_value=httpx.Response(200, json={"status": "processing"})
        )
        result = await client.post("/orders/1/process")
        assert "status" in result

    @respx.mock
    async def test_post_422_validation_error(self, client):
        respx.post(f"{BASE}/products").mock(
            return_value=httpx.Response(422, json={"error_message": "productcode is required"})
        )
        result = await client.post("/products", {"name": "Bad"})
        assert result["error"] is True
        assert result["status_code"] == 422


class TestPut:
    @respx.mock
    async def test_put_updates_resource(self, client):
        respx.put(f"{BASE}/products/1").mock(
            return_value=httpx.Response(200, json={"idproduct": 1, "name": "Updated"})
        )
        result = await client.put("/products/1", {"name": "Updated"})
        assert result["name"] == "Updated"

    @respx.mock
    async def test_put_empty_response(self, client):
        respx.put(f"{BASE}/receipts/1").mock(
            return_value=httpx.Response(204)
        )
        result = await client.put("/receipts/1", {"status": "completed"})
        assert result == {"status": "ok"}


class TestDelete:
    @respx.mock
    async def test_delete_resource(self, client):
        respx.delete(f"{BASE}/orders/1").mock(
            return_value=httpx.Response(200, json={"status": "cancelled"})
        )
        result = await client.delete("/orders/1")
        # _request returns parsed JSON for 200, but our delete override returns {"status":"deleted"}
        # Actually check current implementation — delete calls _request which returns json
        # Let's verify what actually happens
        assert "status" in result

    @respx.mock
    async def test_delete_204(self, client):
        respx.delete(f"{BASE}/backorders/5").mock(
            return_value=httpx.Response(204)
        )
        result = await client.delete("/backorders/5")
        assert result == {"status": "ok"}


class TestAuth:
    @respx.mock
    async def test_basic_auth_header_sent(self, client):
        route = respx.get(f"{BASE}/users/me").mock(
            return_value=httpx.Response(200, json={"iduser": 1})
        )
        await client.get("/users/me")
        auth_header = route.calls[0].request.headers.get("authorization", "")
        # Basic auth: base64("test-api-key-123:x")
        assert auth_header.startswith("Basic ")

    @respx.mock
    async def test_user_agent_header(self, client):
        route = respx.get(f"{BASE}/users/me").mock(
            return_value=httpx.Response(200, json={})
        )
        await client.get("/users/me")
        ua = route.calls[0].request.headers.get("user-agent", "")
        assert "SkirmshopMCP" in ua


class TestConnectionErrors:
    @respx.mock
    async def test_network_error_returns_error_dict(self, client):
        respx.get(f"{BASE}/products").mock(side_effect=httpx.ConnectError("Connection refused"))
        result = await client.get("/products")
        assert result["error"] is True
        assert "Connection refused" in result["detail"]

    @respx.mock
    async def test_timeout_returns_error_dict(self, client):
        respx.get(f"{BASE}/products").mock(side_effect=httpx.ReadTimeout("Timed out"))
        result = await client.get("/products")
        assert result["error"] is True
        assert "Timed out" in result["detail"]
```

**Step 2: Run tests**

Run: `cd mcp-server && source .venv/bin/activate && python -m pytest tests/test_picqer_client.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add mcp-server/tests/test_picqer_client.py
git commit -m "test: add POST/PUT/DELETE, auth, and connection error tests"
```

---

### Task 5: Write unit tests for MCP tool functions

**Files:**
- Create: `mcp-server/tests/test_picqer_tools.py`

These tests verify the tool layer — that tools call the right API paths with correct parameters. We mock at the `PicqerClient` level (not HTTP) since the client is already tested.

**Step 1: Write tool tests**

```python
"""Test Picqer MCP tool functions route correctly to PicqerClient."""

import json
from unittest.mock import AsyncMock, patch

import pytest


# We need to patch the `api` global in picqer_server before importing tools
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


class TestProductTools:
    async def test_list_products(self, mock_api):
        from picqer_server import picqer_products
        result = await picqer_products("list")
        mock_api.get_list.assert_called_once_with("/products", {})

    async def test_get_product(self, mock_api):
        from picqer_server import picqer_products
        result = await picqer_products("get", id=42)
        mock_api.get.assert_called_once_with("/products/42")

    async def test_create_product(self, mock_api):
        from picqer_server import picqer_products
        params = json.dumps({"productcode": "ABC", "name": "Test", "price": 10, "idvatgroup": 1})
        await picqer_products("create", params=params)
        mock_api.post.assert_called_once()
        call_args = mock_api.post.call_args
        assert call_args[0][0] == "/products"
        assert call_args[0][1]["productcode"] == "ABC"

    async def test_update_product(self, mock_api):
        from picqer_server import picqer_products
        await picqer_products("update", id=1, params='{"name": "Updated"}')
        mock_api.put.assert_called_once_with("/products/1", {"name": "Updated"})

    async def test_activate_product(self, mock_api):
        from picqer_server import picqer_products
        await picqer_products("activate", id=1)
        mock_api.post.assert_called_once_with("/products/1/activate")

    async def test_deactivate_product(self, mock_api):
        from picqer_server import picqer_products
        await picqer_products("deactivate", id=1)
        mock_api.post.assert_called_once_with("/products/1/inactivate")

    async def test_unknown_action(self, mock_api):
        from picqer_server import picqer_products
        result = await picqer_products("bogus")
        assert "Unknown action" in result


class TestOrderTools:
    async def test_list_orders(self, mock_api):
        from picqer_server import picqer_orders
        await picqer_orders("list", params='{"status": "open"}')
        mock_api.get_list.assert_called_once_with("/orders", {"status": "open"})

    async def test_process_order(self, mock_api):
        from picqer_server import picqer_order_actions
        await picqer_order_actions("process", order_id=10)
        mock_api.post.assert_called_once_with("/orders/10/process")

    async def test_pause_order(self, mock_api):
        from picqer_server import picqer_order_actions
        await picqer_order_actions("pause", order_id=10, params='{"reason": "waiting"}')
        mock_api.post.assert_called_once_with("/orders/10/pause", {"reason": "waiting"})


class TestStockTools:
    async def test_get_stock(self, mock_api):
        from picqer_server import picqer_product_stock
        await picqer_product_stock("get", product_id=5)
        mock_api.get.assert_called_once_with("/products/5/stock")

    async def test_change_stock(self, mock_api):
        from picqer_server import picqer_product_stock
        await picqer_product_stock("change", product_id=5, warehouse_id=1, params='{"change": 10, "reason": "restock"}')
        mock_api.post.assert_called_once_with("/products/5/stock/1", {"change": 10, "reason": "restock"})

    async def test_move_stock(self, mock_api):
        from picqer_server import picqer_product_stock
        await picqer_product_stock("move", product_id=5, warehouse_id=1, params='{"from_idlocation": 1, "to_idlocation": 2, "amount": 5}')
        mock_api.post.assert_called_once()
        assert "/move" in mock_api.post.call_args[0][0]


class TestWarehouseTools:
    async def test_list_warehouses(self, mock_api):
        from picqer_server import picqer_warehouses
        await picqer_warehouses("list")
        mock_api.get.assert_called_once_with("/warehouses")

    async def test_get_warehouse_stock(self, mock_api):
        from picqer_server import picqer_warehouses
        await picqer_warehouses("stock", id=1)
        mock_api.get_list.assert_called_once()


class TestWebhookTools:
    async def test_create_webhook(self, mock_api):
        from picqer_server import picqer_webhooks
        p = json.dumps({"name": "test", "event": "orders.created", "address": "https://example.com/hook"})
        await picqer_webhooks("create", params=p)
        mock_api.post.assert_called_once()
        assert mock_api.post.call_args[0][0] == "/hooks"

    async def test_delete_webhook(self, mock_api):
        from picqer_server import picqer_webhooks
        await picqer_webhooks("delete", id=99)
        mock_api.delete.assert_called_once_with("/hooks/99")


class TestGenericApiTool:
    async def test_generic_get(self, mock_api):
        from picqer_server import picqer_api
        await picqer_api("GET", "/custom/endpoint", '{"foo": "bar"}')
        mock_api.get.assert_called_once_with("/custom/endpoint", {"foo": "bar"})

    async def test_generic_post(self, mock_api):
        from picqer_server import picqer_api
        await picqer_api("POST", "/custom/endpoint", '{"data": 1}')
        mock_api.post.assert_called_once_with("/custom/endpoint", {"data": 1})

    async def test_generic_unsupported_method(self, mock_api):
        from picqer_server import picqer_api
        result = await picqer_api("PATCH", "/foo")
        assert "Unsupported" in result
```

**Step 2: Run tests**

Run: `cd mcp-server && source .venv/bin/activate && python -m pytest tests/test_picqer_tools.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add mcp-server/tests/test_picqer_tools.py
git commit -m "test: add MCP tool routing tests — products, orders, stock, webhooks, generic API"
```

---

### Task 6: Run full test suite + verify coverage

**Files:** None new — verification only

**Step 1: Run all tests**

Run: `cd mcp-server && source .venv/bin/activate && python -m pytest tests/ -v --tb=short`
Expected: All PASS (should be ~30+ tests)

**Step 2: Check coverage (optional — install pytest-cov if not present)**

Run: `cd mcp-server && source .venv/bin/activate && pip install pytest-cov && python -m pytest tests/ --cov=picqer_client --cov=picqer_server --cov-report=term-missing`
Expected: picqer_client.py should be >90% covered

**Step 3: Commit coverage config if added**

```bash
git add -A && git commit -m "test: verify full test suite passes, add coverage"
```

---

## Wave 2: Docker Integration + RAG Compatibility (tasks 7–10)

### Task 7: Move API key to `.env` and clean up config

**Files:**
- Modify: `mcp-server/picqer_server.py` (remove hardcoded key default)
- Modify: `.env` (add PICQER_API_KEY and PICQER_SUBDOMAIN)
- Modify: `docker-compose.yml` (add env vars to mcp-server service)

**Step 1: Update `.env`**

Add to bottom of `.env`:

```
# Picqer WMS
PICQER_SUBDOMAIN=skirmshop
PICQER_API_KEY=Tcp3JY1GyYxnyR4Of1OrqqkE8y41vUv4zZddROAHa5UfUqlp
```

**Step 2: Remove hardcoded default from picqer_server.py**

In `picqer_server.py`, the config creation should become:

```python
_config = PicqerConfig(
    subdomain=os.getenv("PICQER_SUBDOMAIN", "skirmshop"),
    api_key=os.getenv("PICQER_API_KEY", ""),
)
```

The empty string default means the server won't accidentally work without proper config.

**Step 3: Verify it still works with env vars**

Run: `cd mcp-server && PICQER_SUBDOMAIN=skirmshop PICQER_API_KEY=Tcp3JY1GyYxnyR4Of1OrqqkE8y41vUv4zZddROAHa5UfUqlp python -c "import asyncio; from picqer_server import api; print(asyncio.run(api.get('/users/me'))['full_name'])"`
Expected: `Wesley Beekvelt`

**Step 4: Commit**

```bash
git add mcp-server/picqer_server.py .env
git commit -m "security: move Picqer API key to .env, remove hardcoded default"
```

---

### Task 8: Add Picqer MCP to docker-compose as second service

**Files:**
- Create: `mcp-server/Dockerfile.picqer`
- Modify: `docker-compose.yml`

**Step 1: Create lightweight Dockerfile (no ML deps needed)**

`mcp-server/Dockerfile.picqer`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir mcp httpx
COPY picqer_client.py picqer_server.py ./
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8001
EXPOSE 8001
CMD ["python", "picqer_server.py", "sse"]
```

Note: Only installs `mcp` and `httpx` — no sentence-transformers, no qdrant-client, no ML models. Image will be ~150MB vs ~3GB for the RAG MCP.

**Step 2: Add service to docker-compose.yml**

Add after the `mcp-server` service block:

```yaml
  # Picqer WMS MCP Server (SSE mode for agent-service)
  picqer-mcp:
    build:
      context: ./mcp-server
      dockerfile: Dockerfile.picqer
    container_name: pocharlies-picqer-mcp
    restart: unless-stopped
    environment:
      - PICQER_SUBDOMAIN=${PICQER_SUBDOMAIN:-skirmshop}
      - PICQER_API_KEY=${PICQER_API_KEY}
    networks:
      - skirmshop-network
    ports:
      - "127.0.0.1:8003:8001"
```

**Step 3: Build and test**

Run: `cd /home/ubuntu/skirmshop/pocharlies-qdrant && docker compose build picqer-mcp`
Expected: Build succeeds

Run: `docker compose up -d picqer-mcp && sleep 3 && curl -s http://localhost:8003/sse | head -5`
Expected: SSE stream starts (or connection established)

**Step 4: Commit**

```bash
git add mcp-server/Dockerfile.picqer docker-compose.yml
git commit -m "infra: add Picqer MCP as docker-compose service on port 8003"
```

---

### Task 9: Register Picqer MCP in agent-service

**Files:**
- Modify: `agent-service/mcp_client/mcp_servers.json`

**Step 1: Add picqer-wms server entry**

Update `mcp_servers.json` to:

```json
{
  "servers": {
    "pocharlies-rag": {
      "type": "sse",
      "url": "http://pocharlies-mcp:8000/sse",
      "description": "Pocharlies RAG - web crawling, products, competitors, translation"
    },
    "picqer-wms": {
      "type": "sse",
      "url": "http://pocharlies-picqer-mcp:8001/sse",
      "description": "Picqer WMS - warehouse management, stock, orders, picklists, shipments"
    }
  }
}
```

**Step 2: Verify agent-service discovers both servers' tools**

After `docker compose up -d`, check agent health:

Run: `curl -s http://localhost:8100/health | python3 -m json.tool`
Expected: Both MCP servers connected, tools from both listed

**Step 3: Commit**

```bash
git add agent-service/mcp_client/mcp_servers.json
git commit -m "feat: register Picqer WMS MCP in agent-service for tool discovery"
```

---

### Task 10: Update server.json metadata + final tag

**Files:**
- Modify: `mcp-server/server.json`

**Step 1: Update server.json**

Add a second package entry for the Picqer server, or create a separate `picqer_server.json`. At minimum update the `_meta` section to document both servers.

**Step 2: Run full test suite one last time**

Run: `cd mcp-server && source .venv/bin/activate && python -m pytest tests/ -v`
Expected: All PASS

**Step 3: Run both MCP servers and verify coexistence**

Run: `docker compose up -d mcp-server picqer-mcp && sleep 5 && curl -s http://localhost:8002/sse | head -1 && curl -s http://localhost:8003/sse | head -1`
Expected: Both respond with SSE headers

**Step 4: Final commit + tag**

```bash
git add -A
git commit -m "docs: update server.json metadata for dual MCP setup"
git tag -a picqer-mcp-v1.1.0 -m "Picqer MCP v1.1.0 - TDD, DI, Docker, agent-service integration"
git push origin master --tags
```

---

## Summary

| Wave | Tasks | What it delivers |
|------|-------|-----------------|
| 1 (tasks 1–6) | Test infra, client extraction, 30+ unit tests | Testable, maintainable `PicqerClient` with full coverage |
| 2 (tasks 7–10) | .env secrets, Docker service, agent-service registration | Production-ready dual-MCP stack, RAG + Picqer side by side |

**Key design decisions:**
- `PicqerClient` accepts config via constructor (DI) — fully testable without env vars
- `respx` mocks HTTP at transport level — no real API calls in tests
- Tool tests mock at `PicqerClient` level — test routing, not HTTP
- Separate Dockerfile (only `mcp` + `httpx`) — ~150MB image vs ~3GB for RAG
- Both MCPs coexist on the same Docker network — agent-service discovers both
