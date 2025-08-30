#!/usr/bin/env python3
"""
Pipeline Ingestion Module
=========================

Functions for document ingestion, path discovery, and pipeline utilities.
This module handles the ingestion phase of the RAG pipeline.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain.schema import Document

from RAG.app.config import settings
from RAG.app.logger import get_logger
from RAG.app.loaders import load_many
from RAG.app.chunking import structure_chunks
from RAG.app.Data_Management.metadata import attach_metadata
from RAG.app.Evaluation_Analysis.validate import validate_min_pages, validate_document_pages


def clean_run_outputs() -> None:
    """
    Delete prior run artifacts so new extraction overwrites files.
    
    This function cleans up directories and files from previous pipeline runs
    to ensure a clean state for new processing.
    """
    flag = os.getenv("RAG_CLEAN_RUN", "1").lower() not in ("0", "false", "no")
    if not flag:
        return
        
    log = get_logger()
    log.info("Cleaning previous run outputs...")
    
    # Directories to clean
    for d in (settings.paths.DATA_DIR / "images", settings.paths.DATA_DIR / "elements"):
        try:
            if d.exists():
                shutil.rmtree(d)
                log.debug(f"Cleaned directory: {d}")
        except Exception as e:
            log.warning(f"Failed to clean directory {d}: {e}")
            
    # Logs: queries.jsonl and logs/elements dumps
    try:
        q = settings.paths.LOGS_DIR / "queries.jsonl"
        if q.exists():
            q.unlink(missing_ok=True)
            log.debug(f"Cleaned file: {q}")
    except Exception as e:
        log.warning(f"Failed to clean queries file: {e}")
        
    try:
        elements_dir = settings.paths.LOGS_DIR / "elements"
        if elements_dir.exists():
            shutil.rmtree(elements_dir)
            log.debug(f"Cleaned directory: {elements_dir}")
    except Exception as e:
        log.warning(f"Failed to clean elements directory: {e}")
    
    log.info("Cleanup completed")


def discover_input_paths() -> List[Path]:
    """
    Discover input files for the pipeline.
    
    Returns:
        List[Path]: List of paths to input files (PDF, DOCX, TXT, etc.)
    """
    log = get_logger()
    
    # Look for files in the project root and data directory
    search_paths = [
        settings.paths.PROJECT_ROOT,  # Project root
        settings.paths.DATA_DIR,      # Data directory
    ]
    
    # Supported file extensions
    supported_extensions = {'.pdf', '.xlsx', '.xls'} # {'.pdf', '.txt', '.md', '.csv', '.docx', '.doc'} Optional
    
    # Files to exclude from processing
    excluded_files = {
        'הנחיות פרויקט גמר.pdf',  # Hebrew project guidelines - not part of analysis
        'Database figures and tables.pdf',  # MCP Tools config file - causes incorrect answers
        'README.md',
        'requirements.txt',
        '.gitignore'
    }
    
    # Skip temporary Excel files
    def should_skip_file(file_path: Path) -> bool:
        """Check if file should be skipped based on name patterns."""
        file_name = file_path.name
        
        # Skip excluded files
        if file_name in excluded_files:
            return True
            
        # Skip temporary Excel files (start with ~$)
        if file_name.startswith('~$'):
            return True
            
        return False
    
    discovered_paths = []
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
            
        log.debug(f"Searching for input files in: {search_path}")
        
        # Search recursively for supported files
        for file_path in search_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                # Skip files in certain directories
                if any(skip_dir in str(file_path) for skip_dir in ['__pycache__', '.git', 'node_modules', '.pytest_cache']):
                    continue
                
                # Skip excluded files and temporary files
                if should_skip_file(file_path):
                    log.debug(f"Skipping file: {file_path}")
                    continue
                    
                discovered_paths.append(file_path)
                log.debug(f"Found input file: {file_path}")
    
    # Sort by name for consistent ordering
    discovered_paths.sort(key=lambda x: x.name)
    
    log.info(f"Discovered {len(discovered_paths)} input files")
    for path in discovered_paths:
        log.debug(f"  - {path}")
    
    return discovered_paths


def ingest_documents(paths: List[Path]) -> List[Dict[str, Any]]:
    """
    Ingest documents from the given paths.
    
    Args:
        paths: List of file paths to ingest
        
    Returns:
        List[Dict[str, Any]]: List of document records with metadata
    """
    log = get_logger()
    log.info(f"Starting document ingestion for {len(paths)} files")
    
    all_records = []
    
    # Load documents using the loaders
    for path, elements in load_many(paths):
        log.info(f"Processing {path.name}: {len(elements)} elements")
        
            # Structure chunks for each element
    chunks = structure_chunks(elements, str(path))
    
    # Handle case where structure_chunks returns None
    if chunks is None:
        chunks = []
    
    # Attach metadata to chunks
    for chunk in chunks:
            record = attach_metadata(chunk)
            all_records.append(record)
    
    log.info(f"Ingestion completed: {len(all_records)} records created")
    return all_records


def convert_records_to_documents(records: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert document records to LangChain Document objects.
    
    Args:
        records: List of document records from ingestion
        
    Returns:
        List[Document]: List of LangChain Document objects
    """
    log = get_logger()
    log.info(f"Converting {len(records)} records to Document objects")
    
    documents = []
    
    for record in records:
        try:
            # Extract content and metadata
            content = record.get("page_content", "")
            metadata = record.get("metadata", {})
            
            # Create Document object
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
            
        except Exception as e:
            log.warning(f"Failed to convert record: {e}")
            continue
    
    log.info(f"Conversion completed: {len(documents)} Document objects created")
    return documents


from RAG.app.Evaluation_Analysis.validate import validate_document_pages


def process_documents_pipeline(paths: List[Path]) -> List[Document]:
    """
    Complete document processing pipeline.
    
    This function combines ingestion, chunking, and metadata attachment
    into a single pipeline step.
    
    Args:
        paths: List of input file paths
        
    Returns:
        List[Document]: Processed documents ready for indexing
    """
    log = get_logger()
    log.info("Starting document processing pipeline")
    
    # Step 1: Ingest documents
    records = ingest_documents(paths)
    
    # Step 2: Convert to Document objects
    documents = convert_records_to_documents(records)
    
    # Step 3: Validate documents
    if not validate_document_pages(documents):
        log.warning("Document validation failed, but continuing with pipeline")
    
    log.info(f"Document processing pipeline completed: {len(documents)} documents")
    return documents


def get_ingestion_summary(documents: List[Document]) -> Dict[str, Any]:
    """
    Generate a summary of the ingestion process.
    
    Args:
        documents: List of processed documents
        
    Returns:
        Dict[str, Any]: Summary statistics
    """
    log = get_logger()
    
    # Count documents by type
    type_counts = {}
    file_counts = {}
    page_counts = set()
    
    for doc in documents:
        metadata = doc.metadata or {}
        
        # Count by section type
        section = metadata.get('section', 'Unknown')
        type_counts[section] = type_counts.get(section, 0) + 1
        
        # Count by file
        file_name = metadata.get('file_name', 'Unknown')
        file_counts[file_name] = file_counts.get(file_name, 0) + 1
        
        # Count unique pages
        page = metadata.get('page', 1)
        if isinstance(page, int):
            page_counts.add(page)
    
    summary = {
        'total_documents': len(documents),
        'unique_files': len(file_counts),
        'unique_pages': len(page_counts),
        'type_distribution': type_counts,
        'file_distribution': file_counts
    }
    
    log.info("Ingestion summary:")
    log.info(f"  Total documents: {summary['total_documents']}")
    log.info(f"  Unique files: {summary['unique_files']}")
    log.info(f"  Unique pages: {summary['unique_pages']}")
    log.info(f"  Type distribution: {summary['type_distribution']}")
    
    return summary
