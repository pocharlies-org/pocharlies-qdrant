#!/bin/bash
# Knowledge Vault: Hourly research agent
# Fills intelligence gaps in competitor analyses via web search
# Runs every hour, reads existing notes, identifies missing data, researches

set -e

LOG_PREFIX="[vault-research $(date -u +%H:%M)]"

echo "$LOG_PREFIX Starting research agent..."

RESULT=$(curl -s --max-time 600 -X POST http://localhost:5000/knowledge/research)
STATUS=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('notes_updated',0))" 2>/dev/null || echo "0")

echo "$LOG_PREFIX Research complete: $STATUS notes updated"
echo "$LOG_PREFIX Full result: $RESULT"

# Fix permissions for SMB/Obsidian access
sudo chown -R ubuntu:ubuntu /home/ubuntu/skirmshop/pocharlies-qdrant/knowledge-vault 2>/dev/null || true
