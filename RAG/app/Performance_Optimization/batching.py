"""
Batching system for the RAG pipeline.
Provides batch processing for documents and streaming for large files.
"""

import asyncio
from typing import List, Dict, Any, Iterator, Generator, Optional, Callable
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from langchain.schema import Document
from RAG.app.config import settings
from RAG.app.logger import get_logger


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 10
    max_workers: int = 4
    timeout: int = 300  # 5 minutes
    chunk_size: int = 1024 * 1024  # 1MB for streaming


class BatchProcessor:
    """Processes items in batches with progress tracking."""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.logger = get_logger()
    
    def process_batches(self, items: List[Any], 
                       process_func: Callable[[List[Any]], List[Any]],
                       description: str = "Processing") -> List[Any]:
        """Process items in batches with progress tracking."""
        if not items:
            return []
        
        total_items = len(items)
        total_batches = (total_items + self.config.batch_size - 1) // self.config.batch_size
        results = []
        
        self.logger.info(f"Starting {description}: {total_items} items in {total_batches} batches")
        start_time = time.time()
        
        for i in range(0, total_items, self.config.batch_size):
            batch_num = i // self.config.batch_size + 1
            batch = items[i:i + self.config.batch_size]
            
            self.logger.info(f"{description} batch {batch_num}/{total_batches} ({len(batch)} items)")
            batch_start = time.time()
            
            try:
                batch_results = process_func(batch)
                results.extend(batch_results)
                
                batch_time = time.time() - batch_start
                self.logger.info(f"Batch {batch_num} completed in {batch_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_num}: {e}")
                # Continue with next batch
        
        total_time = time.time() - start_time
        self.logger.info(f"{description} completed: {len(results)} results in {total_time:.2f}s")
        
        return results
    
    def process_batches_parallel(self, items: List[Any],
                                process_func: Callable[[List[Any]], List[Any]],
                                description: str = "Processing") -> List[Any]:
        """Process items in batches using parallel execution."""
        if not items:
            return []
        
        total_items = len(items)
        total_batches = (total_items + self.config.batch_size - 1) // self.config.batch_size
        results = []
        
        self.logger.info(f"Starting parallel {description}: {total_items} items in {total_batches} batches")
        start_time = time.time()
        
        # Create batches
        batches = []
        for i in range(0, total_items, self.config.batch_size):
            batch = items[i:i + self.config.batch_size]
            batches.append(batch)
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_batch = {
                executor.submit(process_func, batch): i 
                for i, batch in enumerate(batches, 1)
            }
            
            for future in as_completed(future_to_batch, timeout=self.config.timeout):
                batch_num = future_to_batch[future]
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    self.logger.info(f"Batch {batch_num}/{total_batches} completed")
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_num}: {e}")
        
        total_time = time.time() - start_time
        self.logger.info(f"Parallel {description} completed: {len(results)} results in {total_time:.2f}s")
        
        return results


class DocumentStreamer:
    """Streams large documents in chunks."""
    
    def __init__(self, chunk_size: int = 1024 * 1024):  # 1MB default
        self.chunk_size = chunk_size
        self.logger = get_logger()
    
    def stream_file(self, file_path: Path) -> Generator[bytes, None, None]:
        """Stream a file in chunks."""
        try:
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except Exception as e:
            self.logger.error(f"Error streaming file {file_path}: {e}")
            raise
    
    def stream_text(self, text: str, chunk_size: int = None) -> Generator[str, None, None]:
        """Stream text in chunks."""
        chunk_size = chunk_size or self.chunk_size
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
    
    def stream_documents(self, documents: List[Document], 
                        chunk_size: int = None) -> Generator[List[Document], None, None]:
        """Stream documents in batches."""
        chunk_size = chunk_size or self.config.batch_size if hasattr(self, 'config') else 10
        
        for i in range(0, len(documents), chunk_size):
            yield documents[i:i + chunk_size]


class DocumentBatcher:
    """Batches documents for processing."""
    
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self.logger = get_logger()
    
    def create_batches(self, documents: List[Document]) -> List[List[Document]]:
        """Create batches of documents."""
        batches = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    def process_document_batches(self, documents: List[Document],
                                process_func: Callable[[List[Document]], List[Document]],
                                description: str = "Processing documents") -> List[Document]:
        """Process documents in batches."""
        processor = BatchProcessor(BatchConfig(batch_size=self.batch_size))
        return processor.process_batches(documents, process_func, description)
    
    def process_document_batches_parallel(self, documents: List[Document],
                                         process_func: Callable[[List[Document]], List[Document]],
                                         description: str = "Processing documents") -> List[Document]:
        """Process documents in batches using parallel execution."""
        processor = BatchProcessor(BatchConfig(batch_size=self.batch_size))
        return processor.process_batches_parallel(documents, process_func, description)


class AsyncBatchProcessor:
    """Asynchronous batch processor for I/O intensive operations."""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.logger = get_logger()
    
    async def process_batches_async(self, items: List[Any],
                                   process_func: Callable[[List[Any]], Any],
                                   description: str = "Processing") -> List[Any]:
        """Process items in batches asynchronously."""
        if not items:
            return []
        
        total_items = len(items)
        total_batches = (total_items + self.config.batch_size - 1) // self.config.batch_size
        results = []
        
        self.logger.info(f"Starting async {description}: {total_items} items in {total_batches} batches")
        start_time = time.time()
        
        # Create batches
        batches = []
        for i in range(0, total_items, self.config.batch_size):
            batch = items[i:i + self.config.batch_size]
            batches.append(batch)
        
        # Process batches concurrently
        tasks = []
        for i, batch in enumerate(batches, 1):
            task = asyncio.create_task(self._process_batch_async(batch, process_func, i, total_batches))
            tasks.append(task)
        
        # Wait for all tasks to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for result in batch_results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch processing error: {result}")
            else:
                results.extend(result)
        
        total_time = time.time() - start_time
        self.logger.info(f"Async {description} completed: {len(results)} results in {total_time:.2f}s")
        
        return results
    
    async def _process_batch_async(self, batch: List[Any],
                                  process_func: Callable[[List[Any]], Any],
                                  batch_num: int, total_batches: int) -> List[Any]:
        """Process a single batch asynchronously."""
        self.logger.info(f"Processing async batch {batch_num}/{total_batches} ({len(batch)} items)")
        batch_start = time.time()
        
        try:
            # Run the process function in a thread pool for CPU-bound operations
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, process_func, batch)
            
            batch_time = time.time() - batch_start
            self.logger.info(f"Async batch {batch_num} completed in {batch_time:.2f}s")
            
            return result if isinstance(result, list) else [result]
            
        except Exception as e:
            self.logger.error(f"Error processing async batch {batch_num}: {e}")
            return []


# Convenience functions
def create_batch_processor(batch_size: int = 10, max_workers: int = 4) -> BatchProcessor:
    """Create a batch processor with custom configuration."""
    config = BatchConfig(batch_size=batch_size, max_workers=max_workers)
    return BatchProcessor(config)

def create_document_batcher(batch_size: int = 10) -> DocumentBatcher:
    """Create a document batcher."""
    return DocumentBatcher(batch_size)

def create_async_batch_processor(batch_size: int = 10, max_workers: int = 4) -> AsyncBatchProcessor:
    """Create an async batch processor."""
    config = BatchConfig(batch_size=batch_size, max_workers=max_workers)
    return AsyncBatchProcessor(config)
