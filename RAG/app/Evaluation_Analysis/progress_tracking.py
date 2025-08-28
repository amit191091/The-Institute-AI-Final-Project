"""
Progress tracking system for the RAG pipeline.
Provides progress bars and performance monitoring for long operations.
"""

import time
import threading
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from contextlib import contextmanager
import psutil
import os

from RAG.app.logger import get_logger


@dataclass
class ProgressMetrics:
    """Metrics for tracking progress and performance."""
    total_items: int = 0
    processed_items: int = 0
    start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def estimated_remaining_time(self) -> float:
        """Estimate remaining time in seconds."""
        if self.processed_items == 0:
            return 0.0
        
        avg_time_per_item = self.elapsed_time / self.processed_items
        remaining_items = self.total_items - self.processed_items
        return avg_time_per_item * remaining_items
    
    @property
    def items_per_second(self) -> float:
        """Calculate processing rate."""
        if self.elapsed_time == 0:
            return 0.0
        return self.processed_items / self.elapsed_time


class ProgressTracker:
    """Tracks progress of operations with detailed metrics."""
    
    def __init__(self, total_items: int = 0, description: str = "Processing"):
        self.metrics = ProgressMetrics(total_items=total_items)
        self.description = description
        self.logger = get_logger()
        self._lock = threading.Lock()
    
    def update(self, processed_items: int = None, increment: int = 1, 
               error: str = None, warning: str = None) -> None:
        """Update progress metrics."""
        with self._lock:
            if processed_items is not None:
                self.metrics.processed_items = processed_items
            else:
                self.metrics.processed_items += increment
            
            if error:
                self.metrics.errors.append(error)
                self.logger.error(f"Progress error: {error}")
            
            if warning:
                self.metrics.warnings.append(warning)
                self.logger.warning(f"Progress warning: {warning}")
            
            self.metrics.last_update_time = time.time()
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information."""
        with self._lock:
            return {
                "description": self.description,
                "progress_percentage": self.metrics.progress_percentage,
                "processed_items": self.metrics.processed_items,
                "total_items": self.metrics.total_items,
                "elapsed_time": self.metrics.elapsed_time,
                "estimated_remaining_time": self.metrics.estimated_remaining_time,
                "items_per_second": self.metrics.items_per_second,
                "errors": len(self.metrics.errors),
                "warnings": len(self.metrics.warnings)
            }
    
    def log_progress(self) -> None:
        """Log current progress."""
        info = self.get_progress_info()
        self.logger.info(
            f"{self.description}: {info['progress_percentage']:.1f}% "
            f"({info['processed_items']}/{info['total_items']}) "
            f"| {info['items_per_second']:.2f} items/s "
            f"| ETA: {info['estimated_remaining_time']:.1f}s"
        )


class ProgressBar:
    """Simple progress bar for console output."""
    
    def __init__(self, total: int, description: str = "Progress", width: int = 50):
        self.total = total
        self.description = description
        self.width = width
        self.current = 0
        self.start_time = time.time()
        self.logger = get_logger()
    
    def update(self, increment: int = 1) -> None:
        """Update progress bar."""
        self.current += increment
        self._display()
    
    def _display(self) -> None:
        """Display the progress bar."""
        if self.total == 0:
            return
        
        percentage = (self.current / self.total) * 100
        filled_width = int(self.width * self.current // self.total)
        bar = '█' * filled_width + '-' * (self.width - filled_width)
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            rate = self.current / elapsed
            eta = (self.total - self.current) / rate if rate > 0 else 0
        else:
            rate = 0
            eta = 0
        
        print(f'\r{self.description}: |{bar}| {percentage:.1f}% '
              f'({self.current}/{self.total}) {rate:.2f} items/s ETA: {eta:.1f}s', 
              end='', flush=True)
    
    def finish(self) -> None:
        """Finish the progress bar."""
        self.current = self.total
        self._display()
        print()  # New line after progress bar


class PerformanceMonitor:
    """Monitors system performance during operations."""
    
    def __init__(self):
        self.logger = get_logger()
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used
        self.start_cpu = psutil.cpu_percent(interval=0.1)
        self.logger.info("Performance monitoring started")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if self.start_time is None:
            return {}
        
        current_time = time.time()
        current_memory = psutil.virtual_memory()
        current_cpu = psutil.cpu_percent(interval=0.1)
        
        elapsed_time = current_time - self.start_time
        memory_used = current_memory.used - self.start_memory if self.start_memory else 0
        memory_percent = current_memory.percent
        
        return {
            "elapsed_time": elapsed_time,
            "memory_used_mb": memory_used / (1024 * 1024),
            "memory_percent": memory_percent,
            "cpu_percent": current_cpu,
            "available_memory_mb": current_memory.available / (1024 * 1024)
        }
    
    def log_performance(self) -> None:
        """Log current performance metrics."""
        metrics = self.get_performance_metrics()
        if metrics:
            self.logger.info(
                f"Performance: {metrics['elapsed_time']:.2f}s elapsed, "
                f"{metrics['memory_used_mb']:.1f}MB used, "
                f"{metrics['memory_percent']:.1f}% memory, "
                f"{metrics['cpu_percent']:.1f}% CPU"
            )
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return final metrics."""
        final_metrics = self.get_performance_metrics()
        self.logger.info("Performance monitoring stopped")
        return final_metrics


class OperationTracker:
    """Tracks operations with progress and performance monitoring."""
    
    def __init__(self, operation_name: str, total_items: int = 0):
        self.operation_name = operation_name
        self.progress_tracker = ProgressTracker(total_items, operation_name)
        self.performance_monitor = PerformanceMonitor()
        self.logger = get_logger()
    
    def start(self) -> None:
        """Start tracking the operation."""
        self.performance_monitor.start_monitoring()
        self.logger.info(f"Starting operation: {self.operation_name}")
    
    def update(self, processed_items: int = None, increment: int = 1,
               error: str = None, warning: str = None) -> None:
        """Update operation progress."""
        self.progress_tracker.update(processed_items, increment, error, warning)
        
        # Log progress periodically
        if self.progress_tracker.metrics.processed_items % 10 == 0:
            self.progress_tracker.log_progress()
            self.performance_monitor.log_performance()
    
    def finish(self) -> Dict[str, Any]:
        """Finish the operation and return summary."""
        final_metrics = self.performance_monitor.stop_monitoring()
        progress_info = self.progress_tracker.get_progress_info()
        
        summary = {
            "operation_name": self.operation_name,
            "progress": progress_info,
            "performance": final_metrics,
            "success": len(self.progress_tracker.metrics.errors) == 0
        }
        
        self.logger.info(f"Operation completed: {self.operation_name}")
        self.logger.info(f"Final summary: {summary}")
        
        return summary


@contextmanager
def track_operation(operation_name: str, total_items: int = 0):
    """Context manager for tracking operations."""
    tracker = OperationTracker(operation_name, total_items)
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.finish()


@contextmanager
def progress_bar(total: int, description: str = "Progress"):
    """Context manager for progress bar."""
    bar = ProgressBar(total, description)
    try:
        yield bar
    finally:
        bar.finish()


def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    def wrapper(*args, **kwargs):
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            final_metrics = monitor.stop_monitoring()
            monitor.logger.info(f"Function {func.__name__} completed: {final_metrics}")
    
    return wrapper


# Convenience functions
def create_progress_tracker(total_items: int, description: str = "Processing") -> ProgressTracker:
    """Create a progress tracker."""
    return ProgressTracker(total_items, description)

def create_operation_tracker(operation_name: str, total_items: int = 0) -> OperationTracker:
    """Create an operation tracker."""
    return OperationTracker(operation_name, total_items)

def get_system_info() -> Dict[str, Any]:
    """Get current system information."""
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    return {
        "cpu_percent": cpu_percent,
        "memory_total_gb": memory.total / (1024**3),
        "memory_available_gb": memory.available / (1024**3),
        "memory_percent": memory.percent,
        "disk_usage_percent": psutil.disk_usage('/').percent
    }
