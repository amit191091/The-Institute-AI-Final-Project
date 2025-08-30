#!/usr/bin/env python3
"""Configuration helpers for chunking modules.

This module provides configuration access for chunking with environment variable overrides
and centralized settings management. Extracted from advanced chunking modules for consistency.

Functions:
    _get_chunking_setting: Get boolean chunking setting with environment override
    _get_chunking_int_setting: Get integer chunking setting with environment override
    _get_chunking_float_setting: Get float chunking setting with environment override
    _get_chunking_string_setting: Get string chunking setting with environment override
    _get_chunking_list_setting: Get list chunking setting with environment override
"""

import os
from typing import List, Union
from RAG.app.config import settings


def _is_truthy(val: str | None) -> bool:
    """Return True if the value is truthy.
    
    Truthy values: 1, true, yes, on, y (case-insensitive).
    """
    return bool(val) and (str(val).lower() in ("1", "true", "yes", "on", "y"))


def _get_chunking_setting(name: str, default: bool | None) -> bool | None:
    """Get chunking setting from centralized config with environment variable override.
    
    Using default=None means "auto" (try if available).
    """
    # Check environment variable first (for backward compatibility)
    raw = os.getenv(f"RAG_{name}")
    if raw is not None:
        return _is_truthy(raw)
    
    # Use centralized configuration
    config_map = {
        "USE_SEMANTIC_CHUNKING": settings.chunking.USE_SEMANTIC_CHUNKING,
        "USE_HEADING_DETECTION": settings.chunking.USE_HEADING_DETECTION,
        "USE_DYNAMIC_TOKENS": settings.chunking.USE_DYNAMIC_TOKENS,
        "PREFER_SEMANTIC_BREAKS": settings.chunking.PREFER_SEMANTIC_BREAKS,
        "CHUNKING_DEBUG": settings.chunking.CHUNKING_DEBUG,
        "LOG_CHUNK_STATS": settings.chunking.LOG_CHUNK_STATS,
        "EXPORT_CHUNK_METADATA": settings.chunking.EXPORT_CHUNK_METADATA,
        "TEXT_SPLIT_MULTI": settings.chunking.TEXT_SPLIT_MULTI,
        "SEMANTIC_CHUNKING": settings.chunking.SEMANTIC_CHUNKING,
    }
    
    return config_map.get(name, default)


def _get_chunking_int_setting(name: str, default: int = 0) -> int:
    """Get chunking integer setting from centralized config with environment variable override."""
    # Check environment variable first (for backward compatibility)
    raw = os.getenv(f"RAG_{name}")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            return default
    
    # Use centralized configuration
    config_map = {
        "SEMANTIC_BATCH_SIZE": settings.chunking.SEMANTIC_BATCH_SIZE,
        "HEADING_MIN_LENGTH": settings.chunking.HEADING_MIN_LENGTH,
        "HEADING_MAX_LENGTH": settings.chunking.HEADING_MAX_LENGTH,
        "TEXT_TARGET_TOK": settings.chunking.TEXT_TARGET_TOK,
        "TEXT_MAX_TOK": settings.chunking.TEXT_MAX_TOK,
        "FIGURE_TABLE_MAX_TOK": settings.chunking.FIGURE_TABLE_MAX_TOK,
        "CONTEXT_LOW_N": settings.chunking.CONTEXT_LOW_N,
        "MIN_CHUNK_LENGTH": settings.chunking.MIN_CHUNK_LENGTH,
        "MAX_CHUNK_LENGTH": settings.chunking.MAX_CHUNK_LENGTH,
        "MIN_CHUNK_TOKENS": settings.chunking.MIN_CHUNK_TOKENS,
        "TEXT_TARGET_TOKENS": settings.chunking.TEXT_TARGET_TOKENS,
        "TEXT_MAX_TOKENS": settings.chunking.TEXT_MAX_TOKENS,
        "TEXT_OVERLAP_SENTENCES": settings.chunking.TEXT_OVERLAP_SENTENCES,
    }
    
    return config_map.get(name, default)


def _get_chunking_float_setting(name: str, default: float = 0.0) -> float:
    """Get chunking float setting from centralized config with environment variable override."""
    # Check environment variable first (for backward compatibility)
    raw = os.getenv(f"RAG_{name}")
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            return default
    
    # Use centralized configuration
    config_map = {
        "SEMANTIC_SIMILARITY_THRESHOLD": settings.chunking.SEMANTIC_SIMILARITY_THRESHOLD,
        "DISTILL_RATIO": settings.chunking.DISTILL_RATIO,
    }
    
    return config_map.get(name, default)


def _get_chunking_string_setting(name: str, default: str = "") -> str:
    """Get chunking string setting from centralized config with environment variable override."""
    # Check environment variable first (for backward compatibility)
    raw = os.getenv(f"RAG_{name}")
    if raw is not None:
        return raw
    
    # Use centralized configuration
    config_map = {
        "SEMANTIC_MODEL_NAME": settings.chunking.SEMANTIC_MODEL_NAME,
    }
    
    return config_map.get(name, default)


def _get_chunking_list_setting(name: str, default: List[str] = None) -> List[str]:
    """Get chunking list setting from centralized config with environment variable override."""
    if default is None:
        default = []
    
    # Check environment variable first (for backward compatibility)
    raw = os.getenv(f"RAG_{name}")
    if raw is not None:
        # Parse comma-separated values
        return [item.strip() for item in raw.split(",") if item.strip()]
    
    # Use centralized configuration
    config_map = {
        "HEADING_PATTERNS": settings.chunking.HEADING_PATTERNS,
    }
    
    return config_map.get(name, default)


def _get_chunking_tuple_setting(name: str, default: tuple = None) -> tuple:
    """Get chunking tuple setting from centralized config with environment variable override."""
    if default is None:
        default = (0, 0)
    
    # Check environment variable first (for backward compatibility)
    raw = os.getenv(f"RAG_{name}")
    if raw is not None:
        try:
            # Parse comma-separated values as integers
            values = [int(item.strip()) for item in raw.split(",") if item.strip()]
            if len(values) >= 2:
                return tuple(values[:2])
        except ValueError:
            pass
        return default
    
    # Use centralized configuration
    config_map = {
        "CHUNK_TOK_AVG_RANGE": settings.chunking.CHUNK_TOK_AVG_RANGE,
    }
    
    return config_map.get(name, default)
