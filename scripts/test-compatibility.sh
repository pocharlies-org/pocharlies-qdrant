#!/bin/bash
# test-compatibility.sh — Validate compatibility system end-to-end
set -euo pipefail

RAG_URL="${RAG_SERVICE_URL:-http://localhost:5000}"
PASS=0
FAIL=0

check() {
    local desc="$1"
    local condition="$2"
    if eval "$condition"; then
        echo "✅ $desc"
        ((PASS++))
    else
        echo "❌ $desc"
        ((FAIL++))
    fi
}

echo "=== Compatibility System Tests ==="
echo ""

# Test 1: Stats endpoint works
STATS=$(curl -s "${RAG_URL}/admin/compatibility/stats")
check "Stats endpoint returns data" '[ "$(echo "$STATS" | python3 -c "import sys,json; print(json.load(sys.stdin).get(\"total_products\", 0))")" -gt 0 ]'

# Test 2: Search with compatible_with filter returns results
RESULTS=$(curl -s -X POST "${RAG_URL}/products/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "inner barrel", "top_k": 10, "compatible_with": "aap-01", "exclude_base_platforms": true}')
echo "AAP-01 inner barrel search:"
echo "$RESULTS" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for r in data.get('results', []):
    print(f'  - {r[\"title\"]} | platforms={r.get(\"compatible_platforms\",[])} | base={r.get(\"is_base_platform\",False)}')
" 2>/dev/null || echo "  (parse error)"

# Test 3: exclude_base_platforms removes guns
HAS_GUN=$(echo "$RESULTS" | python3 -c "
import sys, json
data = json.load(sys.stdin)
guns = [r for r in data.get('results', []) if r.get('is_base_platform', False)]
print(len(guns))
" 2>/dev/null || echo "-1")
check "Exclude base platforms removes guns" '[ "$HAS_GUN" = "0" ]'

# Test 4: Catalog search with filters works (chatbot endpoint)
CATALOG=$(curl -s -X POST "${RAG_URL}/catalog/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "inner barrel upgrade", "top_k": 10, "compatible_with": "aap-01", "exclude_base_platforms": true, "types": ["product", "collection"]}')
PRODUCT_COUNT=$(echo "$CATALOG" | python3 -c "
import sys, json
data = json.load(sys.stdin)
products = [r for r in data.get('results', []) if r.get('source_type') == 'product']
print(len(products))
" 2>/dev/null || echo "0")
check "Catalog search returns filtered products" '[ "$PRODUCT_COUNT" -gt 0 ]'

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
