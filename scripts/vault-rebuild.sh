#!/bin/bash
# Knowledge Vault — Weekly rebuild + git commit
# Called by cron: 0 3 * * 0 (Sunday 3 AM)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VAULT_DIR="$SCRIPT_DIR/../knowledge-vault"
RAG_URL="http://localhost:5000"
LOG_PREFIX="[vault-rebuild]"

echo "$LOG_PREFIX $(date -Iseconds) Starting full vault rebuild..."

# Trigger rebuild via API
HTTP_CODE=$(curl -s -o /tmp/vault-rebuild-response.json -w "%{http_code}" \
    -X POST "$RAG_URL/knowledge/rebuild" \
    --max-time 3600)

if [ "$HTTP_CODE" = "409" ]; then
    echo "$LOG_PREFIX Rebuild already in progress, skipping."
    exit 0
elif [ "$HTTP_CODE" != "200" ]; then
    echo "$LOG_PREFIX ERROR: Rebuild failed with HTTP $HTTP_CODE"
    cat /tmp/vault-rebuild-response.json 2>/dev/null
    exit 1
fi

echo "$LOG_PREFIX Rebuild completed:"
cat /tmp/vault-rebuild-response.json | python3 -m json.tool 2>/dev/null || cat /tmp/vault-rebuild-response.json

# Git commit vault changes (run on host, not in container)
cd "$VAULT_DIR"
if [ -d ".git" ] || git rev-parse --git-dir > /dev/null 2>&1; then
    CHANGES=$(git status --porcelain 2>/dev/null | wc -l)
    if [ "$CHANGES" -gt 0 ]; then
        git add -A
        git commit -m "vault: rebuild $(date +%Y-%m-%d) — auto-crawl update"
        echo "$LOG_PREFIX Git commit created ($CHANGES files changed)"
    else
        echo "$LOG_PREFIX No vault changes to commit"
    fi
else
    echo "$LOG_PREFIX Warning: vault directory is not a git repo"
fi

echo "$LOG_PREFIX $(date -Iseconds) Done."
