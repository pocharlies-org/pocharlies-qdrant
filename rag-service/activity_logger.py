"""
Activity Logger — Lightweight JSONL append log for knowledge brain events.
Process-safe via fcntl.flock. Auto-rotates after 1MB (keeps 7 days).
"""

import fcntl
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_log_path: Path = None


def init_activity_logger(vault_path: str):
    global _log_path
    _log_path = Path(vault_path) / "_meta" / "activity.jsonl"
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Activity logger initialized: {_log_path}")


def log_activity(event: str, detail: str = "", meta: dict = None):
    if not _log_path:
        return
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "detail": detail,
        "meta": meta or {},
    }
    try:
        with open(_log_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(entry) + "\n")
            if f.tell() > 1_000_000:
                _rotate()
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logger.warning(f"Activity log write failed: {e}")


def read_timeline(hours: int = 24, limit: int = 100) -> dict:
    if not _log_path or not _log_path.exists():
        return {"events": [], "total": 0}
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    events = []
    try:
        for line in _log_path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("ts", "") >= cutoff:
                events.append(entry)
    except Exception as e:
        logger.warning(f"Activity log read failed: {e}")
    events.sort(key=lambda e: e.get("ts", ""), reverse=True)
    return {"events": events[:limit], "total": len(events)}


def _rotate():
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        lines = _log_path.read_text().splitlines()
        kept = [l for l in lines if l.strip() and json.loads(l).get("ts", "") >= cutoff]
        _log_path.write_text("\n".join(kept) + "\n" if kept else "")
        logger.info(f"Activity log rotated: kept {len(kept)} entries")
    except Exception as e:
        logger.warning(f"Activity log rotation failed: {e}")
