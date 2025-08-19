"""
Document chunking with structure-aware splitting, distillation, and token budget management
"""
from typing import Dict, List, Optional, Tuple, Any
import logging
import re

from .utils import simple_summarize, approx_token_len, truncate_to_tokens, clean_text
from .metadata import classify_section_type, extract_keywords, attach_metadata
from .config import settings

logger = logging.getLogger(__name__)

class StructureAwareChunker:
    """
    Advanced document chunker that respects document structure and applies
    distillation while maintaining token budgets
    """
    
    def __init__(self):
        self.avg_token_range = settings.CHUNK_TOK_AVG_RANGE
        self.max_tokens = settings.CHUNK_TOK_MAX
        self.overlap_size = settings.OVERLAP_SIZE
    
    def chunk_document(self, elements: List[Any], file_path: str, 
                      client_id: str = None, case_id: str = None) -> List[Dict[str, Any]]:
        """
        Process document elements into structured chunks with metadata
        
        Args:
            elements: Document elements from parser
            file_path: Source file path
            client_id: Client identifier
            case_id: Case identifier
            
        Returns:
            List of processed chunks with full metadata
        """
        chunks = []
        
        for element in elements:
            try:
                processed_chunks = self._process_element(element, file_path)
                for chunk in processed_chunks:
                    # Attach comprehensive metadata
                    chunk_with_metadata = attach_metadata(chunk, client_id, case_id)
                    chunks.append(chunk_with_metadata)
            except Exception as e:
                logger.error(f"Failed to process element: {e}")
                continue
        
        # Post-process for overlaps and merging
        chunks = self._apply_overlaps(chunks)
        
        return chunks
    
    def _process_element(self, element: Any, file_path: str) -> List[Dict[str, Any]]:
        """Process individual document element"""
        # Extract element properties
        kind = getattr(element, "category", getattr(element, "type", "Text"))
        page = self._extract_page_number(element)
        anchor = self._extract_anchor(element)
        raw_text = self._extract_text(element)
        
        if not raw_text or not raw_text.strip():
            return []
        
        # Clean text
        raw_text = clean_text(raw_text)
        
        # Classify section type
        section_type = classify_section_type(kind, raw_text)
        
        # Process based on element type
        if kind.lower() == "table":
            return self._process_table(raw_text, file_path, page, anchor, section_type)
        elif kind.lower() in ("figure", "image"):
            return self._process_figure(raw_text, file_path, page, anchor, section_type)
        else:
            return self._process_text(raw_text, file_path, page, anchor, section_type)
    
    def _process_table(self, text: str, file_path: str, page: int, 
                      anchor: str, section_type: str) -> List[Dict[str, Any]]:
        """Process table elements with special handling"""
        # Apply distillation to get core 5% information
        distilled_summary = simple_summarize(text, ratio=0.05, min_length=100)
        
        # Extract table structure information
        row_range, col_names = self._analyze_table_structure(text)
        
        # Format table content with summary and raw data
        content = f"[TABLE]\\nSUMMARY:\\n{distilled_summary}\\n\\nRAW DATA:\\n{text}"
        
        # Apply token budget (max 800 for tables)
        token_count = approx_token_len(content)
        if token_count > self.max_tokens:
            content = truncate_to_tokens(content, self.max_tokens)
        
        chunk = {
            "file_name": file_path,
            "page": page,
            "section_type": section_type,
            "anchor": anchor,
            "table_row_range": row_range,
            "table_col_names": col_names,
            "content": content.strip(),
            "keywords": extract_keywords(content),
        }
        
        return [chunk]
    
    def _process_figure(self, text: str, file_path: str, page: int, 
                       anchor: str, section_type: str) -> List[Dict[str, Any]]:
        """Process figure/image elements"""
        # For figures, keep more context (50% ratio)
        caption = text or "Figure"
        distilled = simple_summarize(caption, ratio=0.5, min_length=50)
        
        content = f"[FIGURE]\\n{distilled}"
        
        # Apply token budget
        token_count = approx_token_len(content)
        if token_count > self.max_tokens:
            content = truncate_to_tokens(content, self.max_tokens)
        
        chunk = {
            "file_name": file_path,
            "page": page,
            "section_type": section_type,
            "anchor": anchor,
            "content": content.strip(),
            "keywords": extract_keywords(content),
        }
        
        return [chunk]
    
    def _process_text(self, text: str, file_path: str, page: int, 
                     anchor: str, section_type: str) -> List[Dict[str, Any]]:
        """Process textual elements with smart chunking"""
        # Apply distillation to get core 5% information
        distilled = simple_summarize(text, ratio=0.05, min_length=100)
        
        # Check if distilled content fits in one chunk
        token_count = approx_token_len(distilled)
        
        if token_count <= self.avg_token_range[1]:
            # Single chunk
            content = truncate_to_tokens(distilled, self.avg_token_range[1])
            
            chunk = {
                "file_name": file_path,
                "page": page,
                "section_type": section_type,
                "anchor": anchor,
                "content": content.strip(),
                "keywords": extract_keywords(content),
            }
            
            return [chunk]
        else:
            # Split into multiple chunks
            return self._split_long_text(distilled, file_path, page, anchor, section_type)
    
    def _split_long_text(self, text: str, file_path: str, page: int, 
                        anchor: str, section_type: str) -> List[Dict[str, Any]]:
        """Split long text into multiple chunks with overlap"""
        chunks = []
        
        # Split by sentences for better chunk boundaries
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk = ""
        chunk_sentences = []
        
        for sentence in sentences:
            test_chunk = current_chunk + (" " if current_chunk else "") + sentence + "."
            
            if approx_token_len(test_chunk) <= self.avg_token_range[1]:
                current_chunk = test_chunk
                chunk_sentences.append(sentence)
            else:
                # Finalize current chunk
                if current_chunk:
                    chunk = {
                        "file_name": file_path,
                        "page": page,
                        "section_type": section_type,
                        "anchor": f"{anchor}_chunk_{len(chunks)}" if anchor else f"chunk_{len(chunks)}",
                        "content": current_chunk.strip(),
                        "keywords": extract_keywords(current_chunk),
                    }
                    chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = chunk_sentences[-2:] if len(chunk_sentences) >= 2 else chunk_sentences
                current_chunk = ". ".join(overlap_sentences + [sentence]) + "."
                chunk_sentences = overlap_sentences + [sentence]
        
        # Add final chunk
        if current_chunk:
            chunk = {
                "file_name": file_path,
                "page": page,
                "section_type": section_type,
                "anchor": f"{anchor}_chunk_{len(chunks)}" if anchor else f"chunk_{len(chunks)}",
                "content": current_chunk.strip(),
                "keywords": extract_keywords(current_chunk),
            }
            chunks.append(chunk)
        
        return chunks
    
    def _analyze_table_structure(self, table_text: str) -> Tuple[Optional[Tuple[int, int]], Optional[List[str]]]:
        """Analyze table structure to extract row ranges and column names"""
        lines = table_text.strip().split('\\n')
        
        # Try to identify table structure
        row_count = len([line for line in lines if line.strip()])
        
        # Extract potential column headers from first line
        if lines:
            first_line = lines[0]
            # Look for common table separators
            if '|' in first_line:
                col_names = [col.strip() for col in first_line.split('|') if col.strip()]
            elif '\\t' in first_line:
                col_names = [col.strip() for col in first_line.split('\\t') if col.strip()]
            else:
                # Try to split by multiple spaces
                col_names = [col.strip() for col in re.split(r'\\s{2,}', first_line) if col.strip()]
        else:
            col_names = []
        
        row_range = (1, row_count) if row_count > 0 else None
        
        return row_range, col_names[:10]  # Limit to 10 columns
    
    def _extract_page_number(self, element: Any) -> int:
        """Extract page number from element metadata"""
        metadata = getattr(element, "metadata", {}) or {}
        return metadata.get("page_number", 1)
    
    def _extract_anchor(self, element: Any) -> Optional[str]:
        """Extract anchor/ID from element metadata"""
        metadata = getattr(element, "metadata", {}) or {}
        return metadata.get("id")
    
    def _extract_text(self, element: Any) -> str:
        """Extract text content from element"""
        if hasattr(element, 'text'):
            return element.text or ""
        elif isinstance(element, dict):
            return element.get('text', '')
        else:
            return str(element)
    
    def _apply_overlaps(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply overlap between consecutive chunks for better retrieval"""
        if len(chunks) <= 1:
            return chunks
        
        # This is a placeholder for more sophisticated overlap logic
        # For now, we rely on the overlap created during text splitting
        return chunks

def create_chunker() -> StructureAwareChunker:
    """Factory function to create a configured chunker"""
    return StructureAwareChunker()
