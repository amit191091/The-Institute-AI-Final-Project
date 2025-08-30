#!/usr/bin/env python3
"""
Document Service
===============

Handles document loading, processing, and cleanup operations.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
import shutil

from langchain.schema import Document

from RAG.app.config import settings
from RAG.app.logger import get_logger
from RAG.app.loaders import load_many
from RAG.app.chunking import structure_chunks
from RAG.app.Data_Management.metadata import attach_metadata
from RAG.app.Data_Management.normalized_loader import load_normalized_docs
from RAG.app.Evaluation_Analysis.validate import validate_min_pages

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentService:
    """Service for document loading and processing operations."""

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the document service.
        
        Args:
            project_root: Path to project root directory
        """
        if project_root is None:
            self.project_root = Path.cwd()
        else:
            self.project_root = Path(project_root)
        
        self.log = get_logger()

    def _clean_run_outputs(self) -> None:
        """Delete prior run artifacts so new extraction overwrites files."""
        flag = os.getenv("RAG_CLEAN_RUN", "1").lower() not in ("0", "false", "no")
        if not flag:
            return
            
        # Directories
        for d in (settings.paths.DATA_DIR / "images", settings.paths.DATA_DIR / "elements"):
            try:
                if d.exists():
                    shutil.rmtree(d)
            except Exception as e:
                pass
                
        # Logs: queries.jsonl and logs/elements dumps
        try:
            q = settings.paths.LOGS_DIR / "queries.jsonl"
            if q.exists():
                q.unlink(missing_ok=True)
                
            elements_log = settings.paths.LOGS_DIR / "elements"
            if elements_log.exists():
                shutil.rmtree(elements_log)
        except Exception as e:
            pass

    def load_documents(self, use_normalized: bool = False) -> List[Document]:
        """
        Load documents from various sources.
        
        Args:
            use_normalized: Whether to use normalized documents
            
        Returns:
            List[Document]: Loaded documents
        """
        try:
            if use_normalized:
                logger.info("Loading normalized documents...")
                chunks_path = settings.paths.LOGS_DIR / "normalized" / "chunks.jsonl"
                docs = load_normalized_docs(chunks_path)
            else:
                logger.info("Loading documents from data directory...")
                # Use the same document discovery logic as the pipeline
                from RAG.app.pipeline_modules.pipeline_ingestion import discover_input_paths
                paths = discover_input_paths()
                if not paths:
                    logger.warning("No input files found")
                    return []
                # load_many returns a generator, so we need to collect the results
                docs_generator = load_many(paths)
                docs = []
                for path, elements in docs_generator:
                    # Convert elements to documents
                    for element in elements:
                        if hasattr(element, 'text') and element.text:
                            from langchain.schema import Document
                            doc = Document(
                                page_content=element.text,
                                metadata={
                                    'file_name': path.name,
                                    'page': getattr(element.metadata, 'page_number', 1) if hasattr(element, 'metadata') else 1,
                                    'section': getattr(element, 'category', 'Text'),
                                    'anchor': getattr(element.metadata, 'id', '') if hasattr(element, 'metadata') else ''
                                }
                            )
                            docs.append(doc)
            
            logger.info(f"Loaded {len(docs)} documents")
            return docs
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise

    def process_documents(self, docs: List[Document]) -> List[Document]:
        """
        Process documents through chunking and metadata attachment.
        
        Args:
            docs: Raw documents
            
        Returns:
            List[Document]: Processed documents
        """
        try:
            logger.info("Processing documents...")
            
            # Validate minimum pages - count unique pages across all documents
            unique_pages = set()
            for doc in docs:
                if hasattr(doc, 'metadata') and doc.metadata:
                    page = doc.metadata.get('page', 1)
                    if isinstance(page, int):
                        unique_pages.add(page)
            num_pages = len(unique_pages) if unique_pages else 1
            validate_min_pages(num_pages)
            
            # Structure chunks
            logger.info("Structuring chunks...")
            # For normalized documents, we don't need to restructure chunks
            # Just convert them to the expected format
            chunks = []
            for doc in docs:
                chunk = {
                    'content': doc.page_content,
                    'file_name': doc.metadata.get('file_name', 'unknown'),
                    'page': doc.metadata.get('page', 1),
                    'section': doc.metadata.get('section', 'Text'),
                    'anchor': doc.metadata.get('anchor', ''),
                    'figure_number': doc.metadata.get('figure_number'),
                    'figure_label': doc.metadata.get('figure_label'),
                    'table_number': doc.metadata.get('table_number'),
                    'table_label': doc.metadata.get('table_label'),
                    'image_path': doc.metadata.get('image_path'),
                    'table_md_path': doc.metadata.get('table_md_path'),
                    'table_csv_path': doc.metadata.get('table_csv_path')
                }
                chunks.append(chunk)
            
            # Attach metadata
            logger.info("Attaching metadata...")
            processed_docs = []
            for chunk in chunks:
                processed_chunk = attach_metadata(chunk)
                from langchain.schema import Document
                doc = Document(
                    page_content=processed_chunk["page_content"],
                    metadata=processed_chunk["metadata"]
                )
                processed_docs.append(doc)
            
            logger.info(f"Processed {len(processed_docs)} document chunks")
            return processed_docs
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise
