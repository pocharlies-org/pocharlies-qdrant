#!/bin/bash
# Knowledge Vault — Daily reindex (picks up human edits)
# Called by cron: 0 4 * * * (Daily 4 AM)
set -euo pipefail

RAG_URL="http://localhost:5000"
LOG_PREFIX="[vault-reindex]"

echo "$LOG_PREFIX $(date -Iseconds) Starting vault reindex..."

HTTP_CODE=$(curl -s -o /tmp/vault-reindex-response.json -w "%{http_code}" \
    -X POST "$RAG_URL/knowledge/reindex" \
    --max-time 600)

if [ "$HTTP_CODE" != "200" ]; then
    echo "$LOG_PREFIX ERROR: Reindex failed with HTTP $HTTP_CODE"
    cat /tmp/vault-reindex-response.json 2>/dev/null
    exit 1
fi

echo "$LOG_PREFIX Reindex completed:"
cat /tmp/vault-reindex-response.json | python3 -m json.tool 2>/dev/null || cat /tmp/vault-reindex-response.json

echo "$LOG_PREFIX $(date -Iseconds) Done."
