"""Structured JSON logging (structlog) with a stdlib fallback."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "alphaforge"):
    """Return a structlog logger; falls back to stdlib JSON-ish logging if structlog is absent."""
    try:
        import structlog

        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        )
        return structlog.get_logger(name)
    except ImportError:  # pragma: no cover - fallback path
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter('{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}')
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
