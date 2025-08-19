"""
Document validation utilities for RAG system
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DocumentValidator:
    """Validates documents before processing"""
    
    def __init__(self):
        self.supported_extensions = {'.txt', '.pdf', '.docx', '.md', '.html', '.json'}
        self.max_file_size_mb = 50  # 50MB limit
        self.min_pages = 1  # Minimum pages for validation
        
    def validate_file(self, file_path: str) -> bool:
        """
        Validate a single file
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                logger.warning(f"File does not exist: {file_path}")
                return False
                
            # Check file extension
            if path.suffix.lower() not in self.supported_extensions:
                logger.warning(f"Unsupported file type: {path.suffix}")
                return False
                
            # Check file size
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.max_file_size_mb:
                logger.warning(f"File too large: {size_mb:.1f}MB > {self.max_file_size_mb}MB")
                return False
                
            # Check if file is readable
            try:
                with open(file_path, 'rb') as f:
                    # Try to read first few bytes
                    f.read(1024)
            except (IOError, OSError) as e:
                logger.warning(f"Cannot read file {file_path}: {e}")
                return False
                
            logger.info(f"File validation passed: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            return False
    
    def validate_files(self, file_paths: List[str]) -> Dict[str, bool]:
        """
        Validate multiple files
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            Dict mapping file paths to validation results
        """
        results = {}
        for file_path in file_paths:
            results[file_path] = self.validate_file(file_path)
        return results
    
    def validate_content(self, content: str) -> bool:
        """
        Validate document content
        
        Args:
            content: Text content to validate
            
        Returns:
            bool: True if content is valid
        """
        if not content or not content.strip():
            logger.warning("Empty content")
            return False
            
        # Check minimum length
        if len(content.strip()) < 10:
            logger.warning("Content too short")
            return False
            
        # Check for binary content
        try:
            content.encode('utf-8')
        except UnicodeEncodeError:
            logger.warning("Content contains invalid characters")
            return False
            
        return True
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict with file information
        """
        try:
            path = Path(file_path)
            stat = path.stat()
            
            return {
                'name': path.name,
                'extension': path.suffix.lower(),
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'modified': stat.st_mtime,
                'is_valid': self.validate_file(file_path)
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {
                'name': Path(file_path).name,
                'extension': '',
                'size_bytes': 0,
                'size_mb': 0,
                'modified': 0,
                'is_valid': False,
                'error': str(e)
            }
