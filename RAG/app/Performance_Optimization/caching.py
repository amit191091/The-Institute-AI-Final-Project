"""
Caching system for the RAG pipeline.
Provides caching for query analysis, document embeddings, and LLM responses.
"""

import hashlib
import json
import pickle
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from functools import wraps
import threading
from collections import OrderedDict

from RAG.app.config import settings
from RAG.app.logger import get_logger


class CacheEntry:
    """Represents a single cache entry with metadata."""
    
    def __init__(self, key: str, value: Any, ttl: int = 3600):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl  # Time to live in seconds
        self.access_count = 0
        self.last_accessed = time.time()
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.created_at > self.ttl
    
    def access(self):
        """Mark the entry as accessed."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        entry = cls(data["key"], data["value"], data["ttl"])
        entry.created_at = data["created_at"]
        entry.access_count = data["access_count"]
        entry.last_accessed = data["last_accessed"]
        return entry


class MemoryCache:
    """In-memory cache with LRU eviction and TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.logger = get_logger()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        # Create a string representation of the arguments
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if entry.is_expired():
                    del self.cache[key]
                    return None
                
                # Mark as accessed and move to end (LRU)
                entry.access()
                self.cache.move_to_end(key)
                return entry.value
            
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache."""
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                del self.cache[key]
            
            # Check if we need to evict
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add new entry
            entry = CacheEntry(key, value, ttl or self.default_ttl)
            self.cache[key] = entry
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self.cache:
            # Remove expired entries first
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            for key in expired_keys:
                del self.cache[key]
            
            # If still at max size, remove LRU
            if len(self.cache) >= self.max_size:
                lru_key = next(iter(self.cache))
                del self.cache[lru_key]
                self.logger.debug(f"Evicted LRU cache entry: {lru_key}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_entries = len(self.cache)
            expired_entries = sum(1 for entry in self.cache.values() if entry.is_expired())
            total_accesses = sum(entry.access_count for entry in self.cache.values())
            
            return {
                "total_entries": total_entries,
                "expired_entries": expired_entries,
                "total_accesses": total_accesses,
                "max_size": self.max_size,
                "hit_rate": total_accesses / (total_accesses + total_entries) if total_accesses + total_entries > 0 else 0
            }


class FileCache:
    """File-based cache for persistent storage."""
    
    def __init__(self, cache_dir: Optional[Path] = None, default_ttl: int = 86400):
        self.cache_dir = cache_dir or settings.paths.INDEX_DIR / "cache"
        self.default_ttl = default_ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from file cache."""
        try:
            cache_path = self._get_cache_path(key)
            if not cache_path.exists():
                return None
            
            # Check if file is too old
            if time.time() - cache_path.stat().st_mtime > self.default_ttl:
                cache_path.unlink()
                return None
            
            with open(cache_path, 'rb') as f:
                entry = CacheEntry.from_dict(pickle.load(f))
                
                if entry.is_expired():
                    cache_path.unlink()
                    return None
                
                # Update access time
                entry.access()
                with open(cache_path, 'wb') as f:
                    pickle.dump(entry.to_dict(), f)
                
                return entry.value
                
        except Exception as e:
            self.logger.warning(f"Error reading cache file for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in file cache."""
        try:
            entry = CacheEntry(key, value, ttl or self.default_ttl)
            cache_path = self._get_cache_path(key)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(entry.to_dict(), f)
                
        except Exception as e:
            self.logger.warning(f"Error writing cache file for key {key}: {e}")
    
    def clear(self) -> None:
        """Clear all cache files."""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
        except Exception as e:
            self.logger.warning(f"Error clearing cache files: {e}")


class CacheManager:
    """Manages multiple cache layers for different types of data."""
    
    def __init__(self):
        self.memory_cache = MemoryCache(max_size=1000, default_ttl=3600)
        self.file_cache = FileCache(default_ttl=86400)
        self.logger = get_logger()
    
    def get(self, key: str, use_file_cache: bool = True) -> Optional[Any]:
        """Get value from cache, checking memory first, then file."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try file cache if enabled
        if use_file_cache:
            value = self.file_cache.get(key)
            if value is not None:
                # Store in memory cache for faster access
                self.memory_cache.set(key, value)
                return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            use_file_cache: bool = True) -> None:
        """Set value in cache layers."""
        # Set in memory cache
        self.memory_cache.set(key, value, ttl)
        
        # Set in file cache if enabled
        if use_file_cache:
            self.file_cache.set(key, value, ttl)
    
    def clear(self) -> None:
        """Clear all cache layers."""
        self.memory_cache.clear()
        self.file_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all cache layers."""
        return {
            "memory_cache": self.memory_cache.get_stats(),
            "file_cache": {
                "cache_dir": str(self.file_cache.cache_dir),
                "default_ttl": self.file_cache.default_ttl
            }
        }


# Global cache manager instance
_cache_manager = CacheManager()


def cached(ttl: Optional[int] = None, use_file_cache: bool = True, key_prefix: str = ""):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{_cache_manager.memory_cache._generate_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = _cache_manager.get(cache_key, use_file_cache)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            _cache_manager.set(cache_key, result, ttl, use_file_cache)
            
            return result
        return wrapper
    return decorator


# Cache-specific decorators
def cache_query_analysis(ttl: int = 3600):
    """Cache query analysis results."""
    return cached(ttl=ttl, use_file_cache=True, key_prefix="query_analysis")

def cache_embeddings(ttl: int = 86400):
    """Cache document embeddings."""
    return cached(ttl=ttl, use_file_cache=True, key_prefix="embeddings")

def cache_llm_response(ttl: int = 3600):
    """Cache LLM responses."""
    return cached(ttl=ttl, use_file_cache=False, key_prefix="llm_response")


# Convenience functions
def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    return _cache_manager

def clear_all_caches() -> None:
    """Clear all caches."""
    _cache_manager.clear()

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return _cache_manager.get_stats()
