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
