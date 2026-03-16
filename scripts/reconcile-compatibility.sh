#!/bin/bash
# reconcile-compatibility.sh — Daily reconciliation of missing compatibility data
# Run at 4 AM via cron (after Knowledge Synthesizer at 3 AM)

set -euo pipefail

LOG_FILE="/var/log/skirmshop/compatibility-reconciliation.log"
RAG_URL="${RAG_SERVICE_URL:-http://localhost:5000}"

echo "$(date -Iseconds) Starting compatibility reconciliation..." >> "$LOG_FILE"

# Trigger batch analysis (only processes products missing data)
RESPONSE=$(curl -s -X POST "${RAG_URL}/admin/compatibility/analyze-all" \
  -H "Content-Type: application/json")

echo "$(date -Iseconds) Batch analysis triggered: $RESPONSE" >> "$LOG_FILE"

# Wait and check stats (check every 60s for up to 30 min)
PREV_COVERAGE=""
SAME_COUNT=0
for i in $(seq 1 30); do
    sleep 60
    STATS=$(curl -s "${RAG_URL}/admin/compatibility/stats" 2>/dev/null || echo '{}')
    COVERAGE=$(echo "$STATS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('coverage_pct', 0))" 2>/dev/null || echo "0")
    echo "$(date -Iseconds) Coverage: ${COVERAGE}%" >> "$LOG_FILE"

    # If coverage hasn't changed in 3 checks, analysis is done
    if [ "$COVERAGE" = "$PREV_COVERAGE" ]; then
        SAME_COUNT=$((SAME_COUNT + 1))
        [ "$SAME_COUNT" -ge 3 ] && break
    else
        SAME_COUNT=0
    fi
    PREV_COVERAGE="$COVERAGE"
done

echo "$(date -Iseconds) Reconciliation complete. Final stats: $(curl -s ${RAG_URL}/admin/compatibility/stats)" >> "$LOG_FILE"
