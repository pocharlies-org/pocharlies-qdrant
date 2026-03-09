"""
Picqer WMS MCP Server
Complete warehouse management through Picqer API v1.

Covers: Products, Stock, Orders, Customers, Picklists, Purchase Orders,
Receipts, Returns, Backorders, Warehouses, Locations, Suppliers, Tags,
Webhooks, Tasks, Stats, Shipments, Fulfilment, and all reference data.
"""

import json
import logging
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from picqer_client import PicqerClient, PicqerConfig

# Logging to stderr (stdout reserved for MCP JSON-RPC)
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("picqer-mcp")

# ── Configuration ──────────────────────────────────────────────────────────
_config = PicqerConfig(
    subdomain=os.getenv("PICQER_SUBDOMAIN", "skirmshop"),
    api_key=os.getenv("PICQER_API_KEY", "Tcp3JY1GyYxnyR4Of1OrqqkE8y41vUv4zZddROAHa5UfUqlp"),
)
api = PicqerClient(_config)


# ── MCP Server ─────────────────────────────────────────────────────────────
mcp = FastMCP(
    "picqer-wms",
    instructions=(
        "Complete Picqer warehouse management system API. "
        "Manage products, stock, orders, customers, picklists, purchase orders, "
        "receipts, returns, backorders, warehouses, locations, suppliers, and more."
    ),
)


# ── Helpers ────────────────────────────────────────────────────────────────
def P(s: str) -> dict:
    """Parse JSON params string, return empty dict for empty/default input."""
    if not s or s == "{}":
        return {}
    return json.loads(s)


def R(data: Any) -> str:
    """Format API response as indented JSON string."""
    return json.dumps(data, indent=2, default=str, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
#  CONNECTION TEST
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_test_connection() -> str:
    """Test the Picqer API connection. Returns current user info if successful."""
    return R(await api.get("/users/me"))


# ═══════════════════════════════════════════════════════════════════════════
#  PRODUCTS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_products(action: str, id: int = 0, params: str = "{}") -> str:
    """Manage Picqer products.

    Actions:
      list       - List products. params: {search, inactive, tag, type, productcode, idsupplier, idfulfilment_customer}
      get        - Get product by id
      create     - Create product. params: {productcode, name, price, idvatgroup} (required)
      update     - Update product by id. params: any product field
      activate   - Reactivate inactive product by id
      deactivate - Deactivate product by id
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get_list("/products", p))
        case "get":
            return R(await api.get(f"/products/{id}"))
        case "create":
            return R(await api.post("/products", p))
        case "update":
            return R(await api.put(f"/products/{id}", p))
        case "activate":
            return R(await api.post(f"/products/{id}/activate"))
        case "deactivate":
            return R(await api.post(f"/products/{id}/inactivate"))
        case _:
            return f"Unknown action: {action}. Use: list, get, create, update, activate, deactivate"


@mcp.tool()
async def picqer_product_stock(
    action: str, product_id: int = 0, warehouse_id: int = 0, params: str = "{}"
) -> str:
    """Manage product stock levels and view stock history.

    Actions:
      get           - Get total stock for product across all warehouses
      get_warehouse - Get stock in specific warehouse (with location detail)
      change        - Change stock. params: {change: +/-N, reason: "..."} or {amount: N, reason: "..."}
                      Optional: idlocation
      move          - Move stock between locations. params: {from_idlocation, to_idlocation, amount}
      history       - Get stock change history. params: {sincedate, untildate, idproduct, idwarehouse, iduser}
    """
    p = P(params)
    match action:
        case "get":
            return R(await api.get(f"/products/{product_id}/stock"))
        case "get_warehouse":
            return R(await api.get(f"/products/{product_id}/stock/{warehouse_id}"))
        case "change":
            return R(await api.post(f"/products/{product_id}/stock/{warehouse_id}", p))
        case "move":
            return R(await api.post(f"/products/{product_id}/stock/{warehouse_id}/move", p))
        case "history":
            return R(await api.get_list("/stockhistory", p))
        case _:
            return f"Unknown action: {action}. Use: get, get_warehouse, change, move, history"


@mcp.tool()
async def picqer_product_images(
    action: str, product_id: int = 0, image_id: int = 0, params: str = "{}"
) -> str:
    """Manage product images.

    Actions:
      list   - List images for product
      add    - Add image. params: {image: "<base64 JPEG/PNG>"}
      delete - Delete image by image_id
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get(f"/products/{product_id}/images"))
        case "add":
            return R(await api.post(f"/products/{product_id}/images", p))
        case "delete":
            return R(await api.delete(f"/products/{product_id}/images/{image_id}"))
        case _:
            return f"Unknown action: {action}. Use: list, add, delete"


@mcp.tool()
async def picqer_product_parts(
    action: str, product_id: int = 0, part_id: int = 0, params: str = "{}"
) -> str:
    """Manage product compositions (bundles / bill of materials).

    Actions:
      list    - List parts of composition. params: {nested: true} for nested
      get     - Get single part by part_id
      add     - Add part. params: {idproduct: <part_product_id>, amount: N}
      update  - Update part. params: {amount: N}
      remove  - Remove part
      produce - Run production. params: {idwarehouse, amount, reason}
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get(f"/products/{product_id}/parts", p))
        case "get":
            return R(await api.get(f"/products/{product_id}/parts/{part_id}"))
        case "add":
            return R(await api.post(f"/products/{product_id}/parts", p))
        case "update":
            return R(await api.put(f"/products/{product_id}/parts/{part_id}", p))
        case "remove":
            return R(await api.delete(f"/products/{product_id}/parts/{part_id}"))
        case "produce":
            return R(await api.post(f"/products/{product_id}/produce", p))
        case _:
            return f"Unknown action: {action}. Use: list, get, add, update, remove, produce"


@mcp.tool()
async def picqer_product_locations(
    action: str, product_id: int = 0, location_id: int = 0, warehouse_id: int = 0, params: str = "{}"
) -> str:
    """Manage product storage locations and warehouse settings.

    Actions:
      list               - List locations for product
      assign             - Assign to location. params: {idlocation, note, is_preferred}
      update             - Update assignment. params: {note, is_preferred}
      remove             - Unlink from location
      warehouse_settings - Get warehouse-specific settings
      update_warehouse   - Update warehouse settings. params: {stock_level_order, stock_level_desired, stock_location}
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get(f"/products/{product_id}/locations"))
        case "assign":
            return R(await api.post(f"/products/{product_id}/locations", p))
        case "update":
            return R(await api.put(f"/products/{product_id}/locations/{location_id}", p))
        case "remove":
            return R(await api.delete(f"/products/{product_id}/locations/{location_id}"))
        case "warehouse_settings":
            return R(await api.get(f"/products/{product_id}/warehouses"))
        case "update_warehouse":
            return R(await api.put(f"/products/{product_id}/warehouses/{warehouse_id}", p))
        case _:
            return f"Unknown action: {action}. Use: list, assign, update, remove, warehouse_settings, update_warehouse"


@mcp.tool()
async def picqer_product_comments(action: str, product_id: int = 0, params: str = "{}") -> str:
    """Manage product comments and view price history.

    Actions:
      list_comments  - List comments on product
      add_comment    - Add comment. params: {body: "..."}
      price_history  - View price change history
      expected       - Get outstanding purchase orders for product
    """
    p = P(params)
    match action:
        case "list_comments":
            return R(await api.get(f"/products/{product_id}/comments"))
        case "add_comment":
            return R(await api.post(f"/products/{product_id}/comments", p))
        case "price_history":
            return R(await api.get(f"/products/{product_id}/pricehistory"))
        case "expected":
            return R(await api.get(f"/products/{product_id}/expected"))
        case _:
            return f"Unknown action: {action}. Use: list_comments, add_comment, price_history, expected"


# ═══════════════════════════════════════════════════════════════════════════
#  ORDERS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_orders(action: str, id: int = 0, params: str = "{}") -> str:
    """Manage Picqer orders.

    Actions:
      list   - List orders. params: {search, sinceid, beforeid, sincedate, status, tag, idcustomer}
      get    - Get order by id
      create - Create order (status: concept). params: {idcustomer, products: [{idproduct, amount}], ...}
      update - Update order. params: {deliveryname, reference, idshippingprovider_profile, ...}
      delete - Cancel order (concept/expected only). params: {force: true} optional
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get_list("/orders", p))
        case "get":
            return R(await api.get(f"/orders/{id}"))
        case "create":
            return R(await api.post("/orders", p))
        case "update":
            return R(await api.put(f"/orders/{id}", p))
        case "delete":
            return R(await api.delete(f"/orders/{id}"))
        case _:
            return f"Unknown action: {action}. Use: list, get, create, update, delete"


@mcp.tool()
async def picqer_order_actions(action: str, order_id: int = 0, params: str = "{}") -> str:
    """Execute order workflow actions.

    Actions:
      process            - Process order (creates picklist/backorders)
      pause              - Pause processing. params: {reason: "..."}
      resume             - Resume paused order
      reopen             - Reopen processing order
      change_to_concept  - Convert expected order to concept
      undo_cancellation  - Restore cancelled order
      allocate           - Reserve stock for concept order
      deallocate         - Release allocated stock
      prioritise         - Prioritize backorders
      process_backorders - Create picklist from backorders. params: {prefer_bulk_locations: true}
      anonymize          - Remove PII from old orders (7+ days)
      product_status     - Get status of all order lines
    """
    p = P(params)
    match action:
        case "process":
            return R(await api.post(f"/orders/{order_id}/process"))
        case "pause":
            return R(await api.post(f"/orders/{order_id}/pause", p))
        case "resume":
            return R(await api.post(f"/orders/{order_id}/resume"))
        case "reopen":
            return R(await api.post(f"/orders/{order_id}/reopen"))
        case "change_to_concept":
            return R(await api.post(f"/orders/{order_id}/change-to-concept"))
        case "undo_cancellation":
            return R(await api.post(f"/orders/{order_id}/undo-cancellation"))
        case "allocate":
            return R(await api.post(f"/orders/{order_id}/allocate"))
        case "deallocate":
            return R(await api.post(f"/orders/{order_id}/deallocate"))
        case "prioritise":
            return R(await api.post(f"/orders/{order_id}/prioritise"))
        case "process_backorders":
            return R(await api.post(f"/orders/{order_id}/process-backorders", p))
        case "anonymize":
            return R(await api.post(f"/orders/{order_id}/anonymize"))
        case "product_status":
            return R(await api.get(f"/orders/{order_id}/productstatus"))
        case _:
            return f"Unknown action: {action}. Use: process, pause, resume, reopen, change_to_concept, undo_cancellation, allocate, deallocate, prioritise, process_backorders, anonymize, product_status"


@mcp.tool()
async def picqer_order_products(
    action: str, order_id: int = 0, order_product_id: int = 0, params: str = "{}"
) -> str:
    """Manage products within an order.

    Actions:
      add    - Add product to concept order. params: {idproduct, amount, price, name}
      update - Update order product. params: {name, price, amount, remarks}
      remove - Remove product from order
    """
    p = P(params)
    match action:
        case "add":
            return R(await api.post(f"/orders/{order_id}/products", p))
        case "update":
            return R(await api.put(f"/orders/{order_id}/products/{order_product_id}", p))
        case "remove":
            return R(await api.delete(f"/orders/{order_id}/products/{order_product_id}"))
        case _:
            return f"Unknown action: {action}. Use: add, update, remove"


@mcp.tool()
async def picqer_order_fields(
    action: str, order_id: int = 0, field_id: int = 0, params: str = "{}"
) -> str:
    """Manage order field values and order comments/tags.

    Actions:
      set_field    - Set order field value. params: {value: "..."}
      remove_field - Remove order field value
      list_tags    - List tags on order
      add_tag      - Add tag. params: {idtag: N}
      remove_tag   - Remove tag (field_id = idtag)
      list_comments - List order comments
      add_comment   - Add comment. params: {body: "..."}
    """
    p = P(params)
    match action:
        case "set_field":
            return R(await api.put(f"/orders/{order_id}/orderfields/{field_id}", p))
        case "remove_field":
            return R(await api.delete(f"/orders/{order_id}/orderfields/{field_id}"))
        case "list_tags":
            return R(await api.get(f"/orders/{order_id}/tags"))
        case "add_tag":
            return R(await api.post(f"/orders/{order_id}/tags", p))
        case "remove_tag":
            return R(await api.delete(f"/orders/{order_id}/tags/{field_id}"))
        case "list_comments":
            return R(await api.get(f"/orders/{order_id}/comments"))
        case "add_comment":
            return R(await api.post(f"/orders/{order_id}/comments", p))
        case _:
            return f"Unknown action: {action}. Use: set_field, remove_field, list_tags, add_tag, remove_tag, list_comments, add_comment"


@mcp.tool()
async def picqer_webshop_orders(action: str, id: int = 0, params: str = "{}") -> str:
    """View webshop orders imported into Picqer.

    Actions:
      list - List webshop orders. params: {foreign_id, foreign_number, idorder}
      get  - Get single webshop order by id
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get_list("/webshoporders", p))
        case "get":
            return R(await api.get(f"/webshoporders/{id}"))
        case _:
            return f"Unknown action: {action}. Use: list, get"


# ═══════════════════════════════════════════════════════════════════════════
#  CUSTOMERS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_customers(action: str, id: int = 0, params: str = "{}") -> str:
    """Manage Picqer customers.

    Actions:
      list   - List customers. params: {search, customerid}
      get    - Get customer by id
      create - Create customer. params: {name, contactname, telephone, emailaddress, addresses: [...]}
      update - Update customer. params: any customer field
      delete - Delete customer (restricted if orders exist)
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get_list("/customers", p))
        case "get":
            return R(await api.get(f"/customers/{id}"))
        case "create":
            return R(await api.post("/customers", p))
        case "update":
            return R(await api.put(f"/customers/{id}", p))
        case "delete":
            return R(await api.delete(f"/customers/{id}"))
        case _:
            return f"Unknown action: {action}. Use: list, get, create, update, delete"


@mcp.tool()
async def picqer_customer_addresses(
    action: str, customer_id: int = 0, address_id: int = 0, params: str = "{}"
) -> str:
    """Manage customer addresses, tags, and comments.

    Actions:
      list_addresses - List addresses
      add_address    - Add address. params: {name, address, zipcode, city, country}
      update_address - Update address
      delete_address - Delete address
      list_tags      - List customer tags
      add_tag        - Add tag. params: {idtag: N}
      remove_tag     - Remove tag (address_id = idtag)
      list_comments  - List customer comments
      add_comment    - Add comment. params: {body: "..."}
    """
    p = P(params)
    match action:
        case "list_addresses":
            return R(await api.get(f"/customers/{customer_id}/addresses"))
        case "add_address":
            return R(await api.post(f"/customers/{customer_id}/addresses", p))
        case "update_address":
            return R(await api.put(f"/customers/{customer_id}/addresses/{address_id}", p))
        case "delete_address":
            return R(await api.delete(f"/customers/{customer_id}/addresses/{address_id}"))
        case "list_tags":
            return R(await api.get(f"/customers/{customer_id}/tags"))
        case "add_tag":
            return R(await api.post(f"/customers/{customer_id}/tags", p))
        case "remove_tag":
            return R(await api.delete(f"/customers/{customer_id}/tags/{address_id}"))
        case "list_comments":
            return R(await api.get(f"/customers/{customer_id}/comments"))
        case "add_comment":
            return R(await api.post(f"/customers/{customer_id}/comments", p))
        case _:
            return f"Unknown action: {action}. Use: list_addresses, add_address, update_address, delete_address, list_tags, add_tag, remove_tag, list_comments, add_comment"


# ═══════════════════════════════════════════════════════════════════════════
#  PICKLISTS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_picklists(action: str, id: int = 0, params: str = "{}") -> str:
    """Manage picklists (pick orders).

    Actions:
      list     - List picklists. params: {sinceid, beforeid, sincedate, untildate, closed_after,
                 closed_before, picklistid, assigned_to_iduser, idwarehouse, status, tag}
      get      - Get picklist by id
      update   - Update picklist. params: {urgent, invoiced, idshippingprovider_profile}
      close    - Close picklist
      pick     - Mark products as picked. params: {idpicklist_product, amount}
      pickall  - Mark all products as picked
      assign   - Assign user. params: {iduser: N}
      unassign - Unassign user
      snooze   - Snooze picklist. params: {snooze_until: "YYYY-MM-DD HH:MM:SS"}
      pause    - Pause picklist. params: {reason: "..."}
      resume   - Resume picklist
      cancel   - Cancel picklist (converts to backorders)
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get_list("/picklists", p))
        case "get":
            return R(await api.get(f"/picklists/{id}"))
        case "update":
            return R(await api.put(f"/picklists/{id}", p))
        case "close":
            return R(await api.post(f"/picklists/{id}/close"))
        case "pick":
            return R(await api.post(f"/picklists/{id}/pick", p))
        case "pickall":
            return R(await api.post(f"/picklists/{id}/pickall"))
        case "assign":
            return R(await api.post(f"/picklists/{id}/assign", p))
        case "unassign":
            return R(await api.post(f"/picklists/{id}/unassign"))
        case "snooze":
            return R(await api.post(f"/picklists/{id}/snooze", p))
        case "pause":
            return R(await api.post(f"/picklists/{id}/pause", p))
        case "resume":
            return R(await api.post(f"/picklists/{id}/resume"))
        case "cancel":
            return R(await api.post(f"/picklists/{id}/cancel"))
        case _:
            return f"Unknown action: {action}. Use: list, get, update, close, pick, pickall, assign, unassign, snooze, pause, resume, cancel"


@mcp.tool()
async def picqer_picklist_products(
    action: str, picklist_id: int = 0, product_id: int = 0, params: str = "{}"
) -> str:
    """Manage pick locations and packaging for picklist products.

    Actions:
      update_pick_locations - Update pick locations. params: {idlocation, amount}
      packaging_advice      - Get packaging recommendation
      set_packaging         - Set packaging. params: {idpackaging: N}
      pdf                   - Get picklist PDF URL
      packing_pdf           - Get packing list PDF URL
    """
    p = P(params)
    match action:
        case "update_pick_locations":
            return R(await api.put(f"/picklists/{picklist_id}/products/{product_id}/pick_locations", p))
        case "packaging_advice":
            return R(await api.get(f"/picklists/{picklist_id}/packaging-advice"))
        case "set_packaging":
            return R(await api.put(f"/picklists/{picklist_id}/packaging-advice", p))
        case "pdf":
            return R({"url": f"{BASE_URL}/picklists/{picklist_id}/picklistpdf"})
        case "packing_pdf":
            return R({"url": f"{BASE_URL}/picklists/{picklist_id}/packinglistpdf"})
        case _:
            return f"Unknown action: {action}. Use: update_pick_locations, packaging_advice, set_packaging, pdf, packing_pdf"


@mcp.tool()
async def picqer_picklist_shipments(
    action: str, picklist_id: int = 0, params: str = "{}"
) -> str:
    """Manage picklist shipments.

    Actions:
      list    - List shipments for picklist
      create  - Create shipment. params: {idshippingprofile, idpackaging, weight, extrafields}
                For multicolli: {parcels: [{idpackaging, weight}]}
      methods - List allowed shipping methods for picklist
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get(f"/picklists/{picklist_id}/shipments"))
        case "create":
            return R(await api.post(f"/picklists/{picklist_id}/shipments", p))
        case "methods":
            return R(await api.get(f"/picklists/{picklist_id}/shippingmethods"))
        case _:
            return f"Unknown action: {action}. Use: list, create, methods"


@mcp.tool()
async def picqer_picklist_comments(action: str, picklist_id: int = 0, params: str = "{}") -> str:
    """Manage picklist comments.

    Actions:
      list - List comments on picklist
      add  - Add comment. params: {body: "..."}
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get(f"/picklists/{picklist_id}/comments"))
        case "add":
            return R(await api.post(f"/picklists/{picklist_id}/comments", p))
        case _:
            return f"Unknown action: {action}. Use: list, add"


@mcp.tool()
async def picqer_picklist_batches(
    action: str, id: int = 0, picklist_id: int = 0, params: str = "{}"
) -> str:
    """Manage picklist batches.

    Actions:
      list            - List batches. params: {idwarehouse, assigned_to_iduser, type, status}
      get             - Get batch by id
      create          - Create batch. params: {idpicklist_batch_preset} or {idpicklists: [1,2,3]}
      add_picklist    - Add picklist to batch. params: {idpicklist: N}
      remove_picklist - Remove picklist from batch (picklist_id required)
      assign          - Assign/unassign user. params: {iduser: N} or {iduser: null}
      pdf             - Get batch PDF URL. params: {includePicklists, includePackinglists}
      list_comments   - List batch comments
      add_comment     - Add comment. params: {body: "..."}
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get_list("/picklists/batches", p))
        case "get":
            return R(await api.get(f"/picklists/batches/{id}"))
        case "create":
            return R(await api.post("/picklists/batches", p))
        case "add_picklist":
            return R(await api.post(f"/picklists/batches/{id}/picklists", p))
        case "remove_picklist":
            return R(await api.delete(f"/picklists/batches/{id}/picklists/{picklist_id}"))
        case "assign":
            return R(await api.post(f"/picklists/batches/{id}/assign", p))
        case "pdf":
            return R({"url": f"{BASE_URL}/picklists/batches/{id}/pdf"})
        case "list_comments":
            return R(await api.get(f"/picklists/batches/{id}/comments"))
        case "add_comment":
            return R(await api.post(f"/picklists/batches/{id}/comments", p))
        case _:
            return f"Unknown action: {action}. Use: list, get, create, add_picklist, remove_picklist, assign, pdf, list_comments, add_comment"


# ═══════════════════════════════════════════════════════════════════════════
#  PURCHASE ORDERS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_purchase_orders(action: str, id: int = 0, params: str = "{}") -> str:
    """Manage purchase orders.

    Actions:
      list            - List purchase orders. params: {status, search, updated_after, idsupplier, idwarehouse}
      get             - Get purchase order by id
      create          - Create. params: {idsupplier, idwarehouse, products: [{idproduct, amount}]}
      update          - Update. params: {remarks, supplier_orderid}
      mark_purchased  - Change status to purchased
      close           - Close partially received order
      cancel          - Cancel purchase order
      list_comments   - List comments
      add_comment     - Add comment. params: {body: "..."}
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get_list("/purchaseorders", p))
        case "get":
            return R(await api.get(f"/purchaseorders/{id}"))
        case "create":
            return R(await api.post("/purchaseorders", p))
        case "update":
            return R(await api.put(f"/purchaseorders/{id}", p))
        case "mark_purchased":
            return R(await api.post(f"/purchaseorders/{id}/mark-as-purchased"))
        case "close":
            return R(await api.post(f"/purchaseorders/{id}/close"))
        case "cancel":
            return R(await api.post(f"/purchaseorders/{id}/cancel"))
        case "list_comments":
            return R(await api.get(f"/purchaseorders/{id}/comments"))
        case "add_comment":
            return R(await api.post(f"/purchaseorders/{id}/comments", p))
        case _:
            return f"Unknown action: {action}. Use: list, get, create, update, mark_purchased, close, cancel, list_comments, add_comment"


@mcp.tool()
async def picqer_purchase_order_products(
    action: str, po_id: int = 0, product_id: int = 0, params: str = "{}"
) -> str:
    """Manage products within a purchase order.

    Actions:
      list   - List products in purchase order
      add    - Add product. params: {idproduct, amount, price, delivery_date}
      update - Update product. params: {amount, price, delivery_date}
      remove - Remove product
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get(f"/purchaseorders/{po_id}/products"))
        case "add":
            return R(await api.post(f"/purchaseorders/{po_id}/products", p))
        case "update":
            return R(await api.put(f"/purchaseorders/{po_id}/products/{product_id}", p))
        case "remove":
            return R(await api.delete(f"/purchaseorders/{po_id}/products/{product_id}"))
        case _:
            return f"Unknown action: {action}. Use: list, add, update, remove"


# ═══════════════════════════════════════════════════════════════════════════
#  RECEIPTS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_receipts(
    action: str, id: int = 0, product_id: int = 0, params: str = "{}"
) -> str:
    """Manage goods receipts (receiving inventory).

    Actions:
      list              - List receipts. params: {receiptid, status, idpurchaseorder, idsupplier, updated_after, completed_after}
      get               - Get receipt by id
      create            - Create receipt. params: {idpurchaseorder} or {idsupplier}, optional: {version}
      complete          - Mark as completed. params: {status: "completed"}
      delete            - Delete receipt (only if empty)
      expected_products - Get expected products with strategies
      expected_product  - Get strategies for specific product (product_id required)
      receive           - Register received product. params: {idproduct, amount, idlocation, idpurchaseorder_product or strategy}
      revert            - Revert received product (product_id = idreceipt_product)
      list_comments     - List comments
      add_comment       - Add comment. params: {body: "..."}
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get_list("/receipts", p))
        case "get":
            return R(await api.get(f"/receipts/{id}"))
        case "create":
            return R(await api.post("/receipts", p))
        case "complete":
            return R(await api.put(f"/receipts/{id}", p))
        case "delete":
            return R(await api.delete(f"/receipts/{id}"))
        case "expected_products":
            return R(await api.get(f"/receipts/{id}/expected-products"))
        case "expected_product":
            return R(await api.get(f"/receipts/{id}/expected-products/{product_id}"))
        case "receive":
            return R(await api.post(f"/receipts/{id}/products", p))
        case "revert":
            return R(await api.post(f"/receipts/{id}/products/{product_id}/revert"))
        case "list_comments":
            return R(await api.get(f"/receipts/{id}/comments"))
        case "add_comment":
            return R(await api.post(f"/receipts/{id}/comments", p))
        case _:
            return f"Unknown action: {action}. Use: list, get, create, complete, delete, expected_products, expected_product, receive, revert, list_comments, add_comment"


# ═══════════════════════════════════════════════════════════════════════════
#  RETURNS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_returns(action: str, id: int = 0, params: str = "{}") -> str:
    """Manage returns.

    Actions:
      list     - List returns. params: {search, status, emailaddress, idcustomer, created_after, updated_after}
      get      - Get return by id
      create   - Create return. params: {idreturn_status, idtemplate, name, returned_products: [...], replacement_products: [...]}
      update   - Update return. params: {name, emailaddress, ...}
      delete   - Delete return (no linked picklists/backorders)
      receive  - Register receipt. params: {idwarehouse, return_products: [{idreturn_product, status}]}
      add_log  - Add log/change status. params: {idreturn_status, message, notify_customer}
      logs     - List return logs
      statuses - List all return statuses
      reasons  - List all return reasons
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get_list("/returns", p))
        case "get":
            return R(await api.get(f"/returns/{id}"))
        case "create":
            return R(await api.post("/returns", p))
        case "update":
            return R(await api.put(f"/returns/{id}", p))
        case "delete":
            return R(await api.delete(f"/returns/{id}"))
        case "receive":
            return R(await api.post(f"/returns/{id}/receive", p))
        case "add_log":
            return R(await api.post(f"/returns/{id}/logs", p))
        case "logs":
            return R(await api.get(f"/returns/{id}/logs"))
        case "statuses":
            return R(await api.get("/return_statuses"))
        case "reasons":
            return R(await api.get("/return_reasons"))
        case _:
            return f"Unknown action: {action}. Use: list, get, create, update, delete, receive, add_log, logs, statuses, reasons"


@mcp.tool()
async def picqer_return_products(
    action: str, return_id: int = 0, product_id: int = 0, product_type: str = "returned", params: str = "{}"
) -> str:
    """Manage returned and replacement products within a return.

    product_type: "returned" or "replacement"

    Actions:
      list   - List products (returned or replacement)
      add    - Add product. For returned: {idreturn_reason, idproduct, price, amount}
               For replacement: {idproduct, price, amount}
      update - Update product. params: {price, amount, ...}
      remove - Remove product
    """
    p = P(params)
    path_segment = "returned_products" if product_type == "returned" else "replacement_products"
    match action:
        case "list":
            return R(await api.get(f"/returns/{return_id}/{path_segment}"))
        case "add":
            return R(await api.post(f"/returns/{return_id}/{path_segment}", p))
        case "update":
            return R(await api.put(f"/returns/{return_id}/{path_segment}/{product_id}", p))
        case "remove":
            return R(await api.delete(f"/returns/{return_id}/{path_segment}/{product_id}"))
        case _:
            return f"Unknown action: {action}. Use: list, add, update, remove"


@mcp.tool()
async def picqer_return_comments(action: str, return_id: int = 0, params: str = "{}") -> str:
    """Manage return comments.

    Actions:
      list - List comments on return
      add  - Add comment. params: {body: "..."}
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get(f"/returns/{return_id}/comments"))
        case "add":
            return R(await api.post(f"/returns/{return_id}/comments", p))
        case _:
            return f"Unknown action: {action}. Use: list, add"


# ═══════════════════════════════════════════════════════════════════════════
#  BACKORDERS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_backorders(action: str, id: int = 0, params: str = "{}") -> str:
    """Manage backorders.

    Actions:
      list    - List backorders. params: {idorder, idproduct, idcustomer, sinceid, sincedate}
      get     - Get backorder by id
      delete  - Delete backorder (also cancels order line)
      process - Process available backorders into picklists
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get_list("/backorders", p))
        case "get":
            return R(await api.get(f"/backorders/{id}"))
        case "delete":
            return R(await api.delete(f"/backorders/{id}"))
        case "process":
            return R(await api.post("/backorders/process"))
        case _:
            return f"Unknown action: {action}. Use: list, get, delete, process"


# ═══════════════════════════════════════════════════════════════════════════
#  WAREHOUSES & LOCATIONS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_warehouses(action: str, id: int = 0, params: str = "{}") -> str:
    """Manage warehouses.

    Actions:
      list  - List all warehouses
      get   - Get warehouse by id
      stock - Get all stock in warehouse (paginated)
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get("/warehouses"))
        case "get":
            return R(await api.get(f"/warehouses/{id}"))
        case "stock":
            return R(await api.get_list(f"/warehouses/{id}/stock", p))
        case _:
            return f"Unknown action: {action}. Use: list, get, stock"


@mcp.tool()
async def picqer_locations(
    action: str, id: int = 0, product_id: int = 0, params: str = "{}"
) -> str:
    """Manage warehouse locations.

    Actions:
      list     - List locations. params: {type, idwarehouse, name, idlocation_type, only_available}
      get      - Get location by id
      create   - Create location. params: {name, idwarehouse, type, auto_link_to_parent}
      update   - Update location. params: {name, remarks, ...}
      delete   - Delete location (only if no products linked)
      products - List products at location
      set_intent - Set destination intent for product in container. params: {idlocation}
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get_list("/locations", p))
        case "get":
            return R(await api.get(f"/locations/{id}"))
        case "create":
            return R(await api.post("/locations", p))
        case "update":
            return R(await api.put(f"/locations/{id}", p))
        case "delete":
            return R(await api.delete(f"/locations/{id}"))
        case "products":
            return R(await api.get(f"/locations/{id}/products"))
        case "set_intent":
            return R(await api.put(f"/locations/{id}/products/{product_id}/intent", p))
        case _:
            return f"Unknown action: {action}. Use: list, get, create, update, delete, products, set_intent"


@mcp.tool()
async def picqer_location_types(action: str, id: int = 0, params: str = "{}") -> str:
    """Manage location types.

    Actions:
      list        - List all location types
      get         - Get location type by id
      create      - Create. params: {name, color} (both required, color as hex)
      update      - Update. params: {name, color}
      delete      - Delete location type
      set_default - Set as default location type
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get("/locationtypes"))
        case "get":
            return R(await api.get(f"/locationtypes/{id}"))
        case "create":
            return R(await api.post("/locationtypes", p))
        case "update":
            return R(await api.put(f"/locationtypes/{id}", p))
        case "delete":
            return R(await api.delete(f"/locationtypes/{id}"))
        case "set_default":
            return R(await api.post(f"/locationtypes/{id}/default"))
        case _:
            return f"Unknown action: {action}. Use: list, get, create, update, delete, set_default"


# ═══════════════════════════════════════════════════════════════════════════
#  SUPPLIERS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_suppliers(action: str, id: int = 0, params: str = "{}") -> str:
    """Manage suppliers.

    Actions:
      list          - List active suppliers. params: {inactive: true} to include inactive
      get           - Get supplier by id
      create        - Create. params: {name, address, zipcode, city, country}
      update        - Update. params: any supplier field
      list_comments - List supplier comments
      add_comment   - Add comment. params: {body: "..."}
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get_list("/suppliers", p))
        case "get":
            return R(await api.get(f"/suppliers/{id}"))
        case "create":
            return R(await api.post("/suppliers", p))
        case "update":
            return R(await api.put(f"/suppliers/{id}", p))
        case "list_comments":
            return R(await api.get(f"/suppliers/{id}/comments"))
        case "add_comment":
            return R(await api.post(f"/suppliers/{id}/comments", p))
        case _:
            return f"Unknown action: {action}. Use: list, get, create, update, list_comments, add_comment"


# ═══════════════════════════════════════════════════════════════════════════
#  USERS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_users(action: str, id: int = 0, params: str = "{}") -> str:
    """Manage Picqer users.

    Actions:
      list - List all users. params: {active: true/false}
      get  - Get user by id (includes rights/permissions)
      me   - Get current authenticated user
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get("/users", p))
        case "get":
            return R(await api.get(f"/users/{id}"))
        case "me":
            return R(await api.get("/users/me"))
        case _:
            return f"Unknown action: {action}. Use: list, get, me"


# ═══════════════════════════════════════════════════════════════════════════
#  TAGS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_tags(action: str, id: int = 0, params: str = "{}") -> str:
    """Manage tags (used on products, orders, customers, picklists).

    Actions:
      list   - List all tags. params: {search}
      get    - Get tag by id
      create - Create tag. params: {title, color (hex), inherit (bool)} - all required
      update - Update tag. params: {name, color}
      delete - Delete tag
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get("/tags", p))
        case "get":
            return R(await api.get(f"/tags/{id}"))
        case "create":
            return R(await api.post("/tags", p))
        case "update":
            return R(await api.put(f"/tags/{id}", p))
        case "delete":
            return R(await api.delete(f"/tags/{id}"))
        case _:
            return f"Unknown action: {action}. Use: list, get, create, update, delete"


@mcp.tool()
async def picqer_resource_tags(
    action: str, resource_type: str = "products", resource_id: int = 0, tag_id: int = 0, params: str = "{}"
) -> str:
    """Add/remove tags on any resource. resource_type: products, orders, customers.

    Actions:
      list   - List tags on resource
      add    - Add tag. params: {idtag: N}
      remove - Remove tag (tag_id required)
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get(f"/{resource_type}/{resource_id}/tags"))
        case "add":
            return R(await api.post(f"/{resource_type}/{resource_id}/tags", p))
        case "remove":
            return R(await api.delete(f"/{resource_type}/{resource_id}/tags/{tag_id}"))
        case _:
            return f"Unknown action: {action}. Use: list, add, remove"


# ═══════════════════════════════════════════════════════════════════════════
#  COMMENTS (Global)
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_comments(action: str, id: int = 0, params: str = "{}") -> str:
    """Manage global comments across all resources.

    Actions:
      list   - List all comments. params: {idauthor, author_type, idmentioned, mentioned_type}
      get    - Get comment by id
      delete - Delete comment
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get_list("/comments", p))
        case "get":
            return R(await api.get(f"/comments/{id}"))
        case "delete":
            return R(await api.delete(f"/comments/{id}"))
        case _:
            return f"Unknown action: {action}. Use: list, get, delete"


# ═══════════════════════════════════════════════════════════════════════════
#  WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_webhooks(action: str, id: int = 0, params: str = "{}") -> str:
    """Manage webhooks.

    Actions:
      list       - List all webhooks
      get        - Get webhook by id
      create     - Create webhook. params: {name, event, address, secret (optional)}
                   Events: orders.created, orders.closed, orders.status_changed,
                   picklists.created, picklists.closed, picklists.shipments.created,
                   products.created, products.changed, products.stock_changed,
                   purchase_orders.created, purchase_orders.changed,
                   receipts.created, receipts.completed, receipts.product_received,
                   returns.created, returns.status_changed,
                   comments.created, tasks.created, tasks.completed,
                   webshop_orders.imported, and more.
      delete     - Deactivate webhook
      reactivate - Reactivate webhook
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get("/hooks"))
        case "get":
            return R(await api.get(f"/hooks/{id}"))
        case "create":
            return R(await api.post("/hooks", p))
        case "delete":
            return R(await api.delete(f"/hooks/{id}"))
        case "reactivate":
            return R(await api.post(f"/hooks/{id}/reactivate"))
        case _:
            return f"Unknown action: {action}. Use: list, get, create, delete, reactivate"


# ═══════════════════════════════════════════════════════════════════════════
#  SHIPMENTS (Global)
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_shipments(action: str, id: int = 0, params: str = "{}") -> str:
    """View shipments.

    Actions:
      list - List all shipments. params: {search, sincedate, untildate}
      get  - Get shipment by id
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get_list("/shipments", p))
        case "get":
            return R(await api.get(f"/shipments/{id}"))
        case _:
            return f"Unknown action: {action}. Use: list, get"


# ═══════════════════════════════════════════════════════════════════════════
#  TASKS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_tasks(action: str, id: int = 0, params: str = "{}") -> str:
    """Manage tasks.

    Actions:
      list       - List tasks. params: {completed: true/false}
      get        - Get task by id
      create     - Create task. params: {title (required), description, deadline_date, deadline_time, assigned_to_iduser}
      update     - Update task. params: {title, description, deadline_date, deadline_time, assigned_to_iduser}
      delete     - Delete task
      complete   - Mark as completed
      uncomplete - Revert completion
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get("/tasks", p))
        case "get":
            return R(await api.get(f"/tasks/{id}"))
        case "create":
            return R(await api.post("/tasks", p))
        case "update":
            return R(await api.put(f"/tasks/{id}", p))
        case "delete":
            return R(await api.delete(f"/tasks/{id}"))
        case "complete":
            return R(await api.post(f"/tasks/{id}/complete"))
        case "uncomplete":
            return R(await api.post(f"/tasks/{id}/uncomplete"))
        case _:
            return f"Unknown action: {action}. Use: list, get, create, update, delete, complete, uncomplete"


# ═══════════════════════════════════════════════════════════════════════════
#  STATS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_stats(action: str = "list", key: str = "") -> str:
    """Get warehouse statistics.

    Actions:
      list - List all available stat keys
      get  - Get specific stat. key: open-picklists, open-orders, backorders,
             new-orders-today, new-orders-this-week, closed-picklists-today,
             closed-picklists-this-week, new-customers-this-week, total-orders,
             total-products, active-products, inactive-products
    """
    match action:
        case "list":
            return R(await api.get("/stats"))
        case "get":
            return R(await api.get(f"/stats/{key}"))
        case _:
            return f"Unknown action: {action}. Use: list, get"


# ═══════════════════════════════════════════════════════════════════════════
#  REFERENCE DATA
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_reference_data(resource: str, id: int = 0) -> str:
    """Get reference/configuration data (read-only resources, some with CRUD).

    resource options:
      product_fields     - List all product custom fields
      product_field      - Get single product field (id required)
      order_fields       - List all order custom fields
      order_field        - Get single order field (id required)
      customer_fields    - List all customer custom fields
      customer_field     - Get single customer field (id required)
      pricelists         - List all pricelists
      pricelist          - Get single pricelist (id required)
      vat_groups         - List all VAT groups
      shipping_providers - List all shipping providers (with profiles)
      templates          - List all templates
      template           - Get single template (id required)
      packing_stations   - List all packing stations
      packing_station    - Get single packing station (id required)
    """
    match resource:
        case "product_fields":
            return R(await api.get("/productfields"))
        case "product_field":
            return R(await api.get(f"/productfields/{id}"))
        case "order_fields":
            return R(await api.get("/orderfields"))
        case "order_field":
            return R(await api.get(f"/orderfields/{id}"))
        case "customer_fields":
            return R(await api.get("/customerfields"))
        case "customer_field":
            return R(await api.get(f"/customerfields/{id}"))
        case "pricelists":
            return R(await api.get("/pricelists"))
        case "pricelist":
            return R(await api.get(f"/pricelists/{id}"))
        case "vat_groups":
            return R(await api.get("/vatgroups"))
        case "shipping_providers":
            return R(await api.get("/shippingproviders"))
        case "templates":
            return R(await api.get("/templates"))
        case "template":
            return R(await api.get(f"/templates/{id}"))
        case "packing_stations":
            return R(await api.get("/packingstations"))
        case "packing_station":
            return R(await api.get(f"/packingstations/{id}"))
        case _:
            return f"Unknown resource: {resource}. Use: product_fields, order_fields, customer_fields, pricelists, vat_groups, shipping_providers, templates, packing_stations"


# ═══════════════════════════════════════════════════════════════════════════
#  PACKAGINGS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_packagings(action: str, id: int = 0, params: str = "{}") -> str:
    """Manage packagings.

    Actions:
      list   - List all packagings. params: {inactive: true}
      get    - Get packaging by id
      create - Create. params: {name (required), barcode, length, width, height, use_in_auto_advice}
      update - Update. params: {name, barcode, length, width, height, use_in_auto_advice, active}
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get("/packagings", p))
        case "get":
            return R(await api.get(f"/packagings/{id}"))
        case "create":
            return R(await api.post("/packagings", p))
        case "update":
            return R(await api.put(f"/packagings/{id}", p))
        case _:
            return f"Unknown action: {action}. Use: list, get, create, update"


# ═══════════════════════════════════════════════════════════════════════════
#  FULFILMENT
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_fulfilment(action: str, id: int = 0, params: str = "{}") -> str:
    """Manage fulfilment customers.

    Actions:
      list   - List fulfilment customers
      get    - Get fulfilment customer by id
      create - Create. params: {name (required), can_login, username, password, emailaddress,
               language, allowed_shipping_profiles, preferred_idwarehouse}
      update - Update. params: same as create
      report - Get fulfilment report. params: {start-date, end-date} (YYYY-MM-DD)
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get("/fulfilment/customers"))
        case "get":
            return R(await api.get(f"/fulfilment/customers/{id}"))
        case "create":
            return R(await api.post("/fulfilment/customers", p))
        case "update":
            return R(await api.put(f"/fulfilment/customers/{id}", p))
        case "report":
            return R(await api.get(f"/fulfilment/customers/{id}/report", p))
        case _:
            return f"Unknown action: {action}. Use: list, get, create, update, report"


# ═══════════════════════════════════════════════════════════════════════════
#  PICKING CONTAINERS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_picking_containers(
    action: str, id: int = 0, picklist_id: int = 0, params: str = "{}"
) -> str:
    """Manage picking containers.

    Actions:
      list   - List containers. params: {idpicklist}
      get    - Get container by id
      create - Create. params: {name}
      update - Update. params: {name}
      delete - Delete container
      link   - Link picklist to container. params: {idpicklist: N}
    """
    p = P(params)
    match action:
        case "list":
            return R(await api.get("/picking-containers", p))
        case "get":
            return R(await api.get(f"/picking-containers/{id}"))
        case "create":
            return R(await api.post("/picking-containers", p))
        case "update":
            return R(await api.put(f"/picking-containers/{id}", p))
        case "delete":
            return R(await api.delete(f"/picking-containers/{id}"))
        case "link":
            return R(await api.post(f"/picking-containers/{id}/link", p))
        case _:
            return f"Unknown action: {action}. Use: list, get, create, update, delete, link"


# ═══════════════════════════════════════════════════════════════════════════
#  GENERIC API (escape hatch for any endpoint)
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def picqer_api(method: str, path: str, params: str = "{}") -> str:
    """Call any Picqer API endpoint directly.

    method: GET, POST, PUT, DELETE
    path:   API path (e.g., "/products/123/stock")
    params: JSON string - query params for GET/DELETE, body for POST/PUT
    """
    p = P(params)
    match method.upper():
        case "GET":
            return R(await api.get(path, p))
        case "POST":
            return R(await api.post(path, p))
        case "PUT":
            return R(await api.put(path, p))
        case "DELETE":
            return R(await api.delete(path))
        case _:
            return f"Unsupported method: {method}. Use: GET, POST, PUT, DELETE"


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    transport = "stdio"
    if len(sys.argv) > 1 and sys.argv[1] == "sse":
        transport = "sse"

    logger.info(f"Starting Picqer WMS MCP server ({transport}) → {_config.base_url}")
    logger.info(f"Subdomain: {_config.subdomain}")

    if transport == "sse":
        mcp.settings.host = os.getenv("MCP_HOST", "0.0.0.0")
        mcp.settings.port = int(os.getenv("MCP_PORT", "8001"))

    mcp.run(transport=transport)
