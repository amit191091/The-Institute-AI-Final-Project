"""
Dynamic figure content processing module.

This module provides functionality to improve figure descriptions by extracting
actual figure descriptions from document text chunks and linking them to figures.
"""

import re
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from app.logger import get_logger

app_logger = get_logger()

def extract_figure_descriptions_from_chunks(chunks: List[Dict]) -> Dict[int, str]:
    """
    Extract figure descriptions from document text chunks.
    
    Args:
        chunks: List of document chunks
        
    Returns:
        Dictionary mapping figure numbers to their descriptions
    """
    figure_descriptions = {}
    
    for chunk in chunks:
        content = chunk.get('content', '')
        if not content:
            continue
            
        # Look for "Figure N: description" patterns in text
        figure_patterns = [
            r'Figure\s+(\d+):\s*([^.]+(?:\[[^\]]+\][^.]*)*[^.]*\.?)',
            r'Figure\s+(\d+)\s*-\s*([^.]+(?:\[[^\]]+\][^.]*)*[^.]*\.?)',
            r'Fig\.?\s+(\d+):\s*([^.]+(?:\[[^\]]+\][^.]*)*[^.]*\.?)',
        ]
        
        for pattern in figure_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                try:
                    fig_num = int(match[0])
                    description = match[1].strip()
                    
                    # Clean up the description
                    description = re.sub(r'\s+', ' ', description)  # Normalize whitespace
                    description = description.rstrip('.,;')  # Remove trailing punctuation
                    
                    if len(description) > 10:  # Only keep meaningful descriptions
                        figure_descriptions[fig_num] = description
                        app_logger.debug(f"Found Figure {fig_num}: {description[:50]}...")
                except (ValueError, IndexError):
                    continue
    
    return figure_descriptions

def extract_table_descriptions_from_chunks(chunks: List[Dict]) -> Dict[int, str]:
    """
    Extract table descriptions from document text chunks.
    
    Args:
        chunks: List of document chunks
        
    Returns:
        Dictionary mapping table numbers to their descriptions
    """
    table_descriptions = {}
    
    for chunk in chunks:
        content = chunk.get('content', '')
        if not content:
            continue
            
        # Look for "Table N: description" patterns in text
        table_patterns = [
            r'Table\s+(\d+):\s*([^.]+(?:\[[^\]]+\][^.]*)*[^.]*\.?)',
            r'Table\s+(\d+)\s*-\s*([^.]+(?:\[[^\]]+\][^.]*)*[^.]*\.?)',
            r'Tab\.?\s+(\d+):\s*([^.]+(?:\[[^\]]+\][^.]*)*[^.]*\.?)',
        ]
        
        for pattern in table_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                try:
                    table_num = int(match[0])
                    description = match[1].strip()
                    
                    # Clean up the description
                    description = re.sub(r'\s+', ' ', description)  # Normalize whitespace
                    description = description.rstrip('.,;')  # Remove trailing punctuation
                    
                    if len(description) > 5:  # Only keep meaningful descriptions
                        table_descriptions[table_num] = description
                        app_logger.debug(f"Found Table {table_num}: {description[:50]}...")
                except (ValueError, IndexError):
                    continue
    
    return table_descriptions

def detect_garbled_ocr(text: str) -> bool:
    """
    Detect if text appears to be garbled OCR output.
    
    Args:
        text: Text to analyze
        
    Returns:
        True if text appears garbled, False otherwise
    """
    if not text or len(text) < 5:
        return False
        
    # Count problematic patterns
    problematic_patterns = [
        r'[^\w\s\[\]().,;:!?%-]',     # Unusual characters
        r'\b[a-zA-Z]{1}\b',           # Single letter words
        r'[A-Z]{4,}',                 # Long sequences of capitals
        r'[^a-zA-Z\s]{3,}',          # Sequences of non-letters
        r'\s{3,}',                    # Multiple spaces
    ]
    
    total_issues = 0
    for pattern in problematic_patterns:
        matches = re.findall(pattern, text)
        total_issues += len(matches)
    
    # Also check for specific OCR garbage patterns
    ocr_garbage_patterns = [
        r'[=§¢&]+',                   # Multiple special chars
        r'\b[a-zA-Z]{1,2}\s[a-zA-Z]{1,2}\b',  # Scattered short words
        r'[A-Za-z]\s*[^\w\s]\s*[A-Za-z]',     # Letter-symbol-letter patterns
    ]
    
    for pattern in ocr_garbage_patterns:
        matches = re.findall(pattern, text)
        total_issues += len(matches) * 2  # Weight these more heavily
    
    # Consider garbled if issues > 10% of text length (lowered threshold)
    issue_ratio = total_issues / max(len(text), 1)
    return issue_ratio > 0.1

def extract_figure_number(content: str) -> Optional[int]:
    """Extract figure number from content."""
    patterns = [
        r'[Ff]igure\s*(\d+)',
        r'[Ff]ig\.?\s*(\d+)',
        r'figure_number["\']?\s*:\s*(\d+)',
        r'anchor["\']?\s*:\s*["\']?figure-(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return int(match.group(1))
    return None

def extract_table_number(content: str) -> Optional[int]:
    """Extract table number from content."""
    patterns = [
        r'[Tt]able\s*(\d+)',
        r'[Tt]ab\.?\s*(\d+)',
        r'table_number["\']?\s*:\s*(\d+)',
        r'anchor["\']?\s*:\s*["\']?table-(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return int(match.group(1))
    return None

def enhance_figure_content(content: str, figure_number: Optional[int] = None, 
                         all_chunks: Optional[List[Dict]] = None) -> str:
    """
    Enhance figure content by replacing garbled OCR with actual descriptions from document.
    
    Args:
        content: Original figure content (may contain garbled OCR)
        figure_number: Figure number if known
        all_chunks: All document chunks to search for descriptions
        
    Returns:
        Enhanced content with better descriptions
    """
    if not content:
        return content
    
    # Extract figure number if not provided
    if figure_number is None:
        figure_number = extract_figure_number(content)
    
    if figure_number is None:
        return content
    
    # Extract descriptions from document chunks if provided
    figure_descriptions = {}
    if all_chunks:
        figure_descriptions = extract_figure_descriptions_from_chunks(all_chunks)
    
    # Check for garbled OCR in the content
    ocr_text_match = re.search(r'OCR Text:\s*(.+?)(?:\nContext|$)', content, re.IGNORECASE | re.DOTALL)
    
    if ocr_text_match:
        ocr_text = ocr_text_match.group(1).strip()
        
        if detect_garbled_ocr(ocr_text):
            app_logger.info(f"Detected garbled OCR for Figure {figure_number}: '{ocr_text[:30]}...'")
            
            # Try to find description from document
            if figure_number in figure_descriptions:
                enhanced_desc = figure_descriptions[figure_number]
                app_logger.info(f"Replacing with document description: '{enhanced_desc[:50]}...'")
                
                # Replace OCR text with actual description
                enhanced_content = re.sub(
                    r'OCR Text:\s*.+?(?=\nContext|\n|$)', 
                    f"Document Description: {enhanced_desc}", 
                    content, 
                    flags=re.IGNORECASE | re.DOTALL
                )
                return enhanced_content
            else:
                app_logger.warning(f"No description found for Figure {figure_number}")
    
    return content

def enhance_table_content(content: str, table_number: Optional[int] = None, 
                        all_chunks: Optional[List[Dict]] = None) -> str:
    """
    Enhance table content by finding actual table descriptions from document.
    
    Args:
        content: Original table content
        table_number: Table number if known
        all_chunks: All document chunks to search for descriptions
        
    Returns:
        Enhanced content with better descriptions
    """
    if not content:
        return content
    
    # Extract table number if not provided
    if table_number is None:
        table_number = extract_table_number(content)
    
    if table_number is None:
        return content
    
    # Extract descriptions from document chunks if provided
    table_descriptions = {}
    if all_chunks:
        table_descriptions = extract_table_descriptions_from_chunks(all_chunks)
    
    # Try to find description from document
    if table_number in table_descriptions:
        enhanced_desc = table_descriptions[table_number]
        app_logger.info(f"Found description for Table {table_number}: '{enhanced_desc[:50]}...'")
        
        # Add description to content
        enhanced_content = f"Table {table_number}: {enhanced_desc}\n\n{content}"
        return enhanced_content
    
    return content
