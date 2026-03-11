"""Structured logging configuration for the Sanitizer application."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))
_LOG_FILE = _LOG_DIR / "sanitizer.log"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 3

_configured = False

# Patterns that look like absolute paths (Unix or Windows)
_ABS_PATH_RE = re.compile(
    r"(?:"
    r'[A-Za-z]:\\[^\s,;"\']+'  # Windows: C:\Users\...
    r'|/(?:home|Users|tmp|var|etc|opt|mnt|data)/[^\s,;"\']+'  # Unix: /home/user/...
    r")"
)
# Common PII patterns
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


class _SanitizingFilter(logging.Filter):
    """Scrub absolute file paths and PII patterns from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._scrub(record.msg)
        if record.args:
            record.args = (
                tuple(self._scrub(a) if isinstance(a, str) else a for a in record.args)
                if isinstance(record.args, tuple)
                else record.args
            )
        return True

    @staticmethod
    def _scrub(text: str) -> str:
        text = _ABS_PATH_RE.sub(lambda m: Path(m.group()).name, text)
        text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
        return text


def configure_logging() -> None:
    """Set up root logging with console + rotating file handler.

    Reads LOG_LEVEL from the environment (default: INFO).
    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _configured
    if _configured:
        return
    _configured = True

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    sanitizing_filter = _SanitizingFilter()

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler (stderr)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    console.addFilter(sanitizing_filter)
    root.addHandler(console)

    # Rotating file handler
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        file_handler.addFilter(sanitizing_filter)
        root.addHandler(file_handler)
        # Restrict log file permissions where supported
        with contextlib.suppress(OSError):
            os.chmod(_LOG_FILE, 0o600)
    except OSError:
        # If we can't write logs (e.g. read-only filesystem), continue with console only
        root.warning(
            "Could not create log file at %s — using console logging only", _LOG_FILE
        )

    # Quiet noisy third-party loggers
    for noisy in (
        "sdv",
        "rdt",
        "copulas",
        "ctgan",
        "graphviz",
        "urllib3",
        "matplotlib",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
