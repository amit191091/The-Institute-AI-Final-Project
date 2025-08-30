import logging
import os
from pathlib import Path
from typing import Optional
from functools import wraps
from RAG.app.config import settings

_LOGGER: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    logger = logging.getLogger("hybrid_rag")
    logger.setLevel(os.getenv("RAG_LOG_LEVEL", "INFO"))
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(os.getenv("RAG_LOG_LEVEL", "INFO"))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_dir = settings.paths.LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    fh.setLevel(os.getenv("RAG_FILE_LOG_LEVEL", "DEBUG"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    _LOGGER = logger
    return logger


# Lightweight function-entry tracer. Enable with env RAG_TRACE_FUNCS=1/true/yes/on
def trace_func(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        try:
            if str(os.getenv("RAG_TRACE_FUNCS", "0")).lower() in ("1", "true", "yes", "on"):
                get_logger().info(f"im here, this is {func.__name__}")
        except Exception:
            pass
        return func(*args, **kwargs)
    return _wrapper


def trace_here(name: str) -> None:
    """Manual trace injection for spots where a decorator isn't practical."""
    try:
        if str(os.getenv("RAG_TRACE_FUNCS", "0")).lower() in ("1", "true", "yes", "on"):
            get_logger().info(f"im here, this is {name}")
    except Exception:
        pass
