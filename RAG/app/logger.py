import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "rag", level: int = logging.INFO) -> logging.Logger:
    """Get a logger instance for the RAG system."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
    
    return logger


def setup_file_logging(logger: logging.Logger, log_file: Optional[Path] = None) -> None:
    """Setup file logging for the logger."""
    if log_file is None:
        log_file = Path("logs/rag.log")
    
    # Create logs directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
