"""Structured logging.

Console output is human-readable. The log file (``<data_dir>/logs/predictor.log``) holds
one JSON object per line so failures can be diagnosed afterwards. Use
``log_event(logger, "event_name", key=value, ...)`` to attach structured fields.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = "predictor"
_STANDARD_ATTRS = set(vars(logging.LogRecord("x", 0, "", 0, "", None, None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                try:
                    json.dumps(value)
                    payload[key] = value
                except TypeError:
                    payload[key] = repr(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{record.levelname:<7} {record.name}: {record.getMessage()}"
        extras = {k: v for k, v in record.__dict__.items()
                  if k not in _STANDARD_ATTRS and not k.startswith("_")}
        if extras:
            base += "  " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


_configured = False


def configure_logging(level: str = "INFO", log_dir: Path | None = None, *, json_file: bool = True,
                      max_bytes: int = 5_000_000, backups: int = 3, console_level: str = "WARNING") -> None:
    """Configure the ``predictor`` logger once (safe to call repeatedly)."""
    global _configured
    root = logging.getLogger(ROOT)
    root.setLevel(level.upper())
    if _configured:
        return
    console = logging.StreamHandler()
    console.setLevel(console_level.upper())
    console.setFormatter(ConsoleFormatter())
    root.addHandler(console)
    if log_dir is not None and json_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_dir / "predictor.log", maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
        )
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT}.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"event": event, **fields})
