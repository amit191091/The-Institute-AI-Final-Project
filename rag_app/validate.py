"""
Document validation and data criteria checking
"""
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

from .config import settings
from .loaders import DocumentLoader
from .utils import approx_token_len

logger = logging.getLogger(__name__)

class DocumentValidator:
    """
    Validates documents against data criteria and ingestion requirements
    """
    
    def __init__(self):
        self.loader = DocumentLoader()
        self.min_pages = settings.MIN_PAGES
        self.chunk_token_range = settings.CHUNK_TOK_AVG_RANGE
        self.max_chunk_tokens = settings.CHUNK_TOK_MAX
    
    def validate_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Comprehensive document validation
        
        Args:
            file_path: Path to document file
            
        Returns:
            Validation results dictionary
        """
        validation_result = {
            "file_path": str(file_path),
            "is_valid": False,
            "warnings": [],
            "errors": [],
            "metadata": {},
            "recommendations": []
        }
        
        try:
            # Basic file checks
            if not file_path.exists():
                validation_result["errors"].append("File does not exist")
                return validation_result
            
            if file_path.suffix.lower() not in settings.SUPPORTED_EXTENSIONS:
                validation_result["errors"].append(f"Unsupported file format: {file_path.suffix}")
                return validation_result
            
            # Load and analyze document
            elements = self.loader.load_elements(file_path)
            
            if not elements:
                validation_result["errors"].append("Document contains no readable content")
                return validation_result
            
            # Perform validation checks
            page_validation = self._validate_page_count(elements)
            content_validation = self._validate_content_structure(elements)
            metadata_validation = self._validate_metadata_requirements(elements)
            technical_validation = self._validate_technical_content(elements)
            
            # Compile results
            validation_result["metadata"] = {
                "page_count": page_validation["page_count"],
                "element_count": len(elements),
                "has_tables": content_validation["has_tables"],
                "has_figures": content_validation["has_figures"],
                "content_types": content_validation["content_types"],
                "technical_indicators": technical_validation["indicators"]
            }
            
            # Collect warnings and errors
            all_validations = [page_validation, content_validation, metadata_validation, technical_validation]
            for validation in all_validations:
                validation_result["warnings"].extend(validation.get("warnings", []))
                validation_result["errors"].extend(validation.get("errors", []))
                validation_result["recommendations"].extend(validation.get("recommendations", []))
            
            # Determine overall validity
            validation_result["is_valid"] = len(validation_result["errors"]) == 0
            
            # Add success message if valid
            if validation_result["is_valid"]:
                validation_result["message"] = "Document passes all validation criteria"
            
            return validation_result
            
        except Exception as e:
            validation_result["errors"].append(f"Validation failed: {str(e)}")
            logger.error(f"Document validation error for {file_path}: {e}")
            return validation_result
    
    def _validate_page_count(self, elements: List[Any]) -> Dict[str, Any]:
        """Validate document has minimum required pages"""
        result = {"warnings": [], "errors": [], "recommendations": []}
        
        # Try multiple methods to get page count
        page_count = 1  # Default fallback
        
        # Method 1: Check element metadata for page numbers
        page_numbers = set()
        for element in elements:
            metadata = getattr(element, "metadata", {}) or {}
            page_num = metadata.get("page_number")
            if page_num:
                page_numbers.add(page_num)
        
        if page_numbers:
            page_count = len(page_numbers)
        else:
            # Method 2: Try to get page count from file directly if available
            try:
                # Look for filename in first element metadata
                first_element = elements[0] if elements else None
                if first_element:
                    metadata = getattr(first_element, "metadata", {}) or {}
                    filename = metadata.get("filename")
                    
                    if filename and str(filename).lower().endswith('.pdf'):
                        import PyPDF2
                        from pathlib import Path
                        
                        pdf_path = Path(filename) if Path(filename).exists() else None
                        if pdf_path and pdf_path.exists():
                            with open(pdf_path, 'rb') as file:
                                pdf_reader = PyPDF2.PdfReader(file)
                                page_count = len(pdf_reader.pages)
            except Exception:
                pass  # Keep default page_count
        
        result["page_count"] = page_count
        
        # More lenient validation - only error if truly problematic
        if page_count == 0:
            result["errors"].append(f"Document appears to have no readable pages")
        elif page_count < 3:
            result["warnings"].append(f"Document has only {page_count} page(s) - this may be sufficient for analysis")
        else:
            result["message"] = f"Page count validation passed: {page_count} pages"
        
        return result
    
    def _validate_content_structure(self, elements: List[Any]) -> Dict[str, Any]:
        """Validate document structure and content organization"""
        result = {"warnings": [], "errors": [], "recommendations": []}
        
        # Analyze content types
        content_types = {}
        has_tables = False
        has_figures = False
        has_headers = False
        total_text_length = 0
        
        for element in elements:
            element_type = getattr(element, "category", getattr(element, "type", "unknown"))
            text = getattr(element, "text", "") or ""
            
            content_types[element_type] = content_types.get(element_type, 0) + 1
            
            if element_type.lower() == "table":
                has_tables = True
            elif element_type.lower() in ("figure", "image"):
                has_figures = True
            elif element_type.lower() in ("title", "header"):
                has_headers = True
            
            total_text_length += len(text)
        
        result.update({
            "content_types": content_types,
            "has_tables": has_tables,
            "has_figures": has_figures,
            "has_headers": has_headers,
            "total_text_length": total_text_length
        })
        
        # Validation checks
        if total_text_length < 1000:
            result["warnings"].append("Document appears to have very little text content")
        
        if not has_headers:
            result["warnings"].append("Document lacks clear section headers/structure")
            result["recommendations"].append("Add section headers to improve document structure")
        
        if not has_tables and "analysis" in str(elements).lower():
            result["warnings"].append("Technical analysis document without tables may lack detailed data")
        
        return result
    
    def _validate_metadata_requirements(self, elements: List[Any]) -> Dict[str, Any]:
        """Validate metadata requirements for anchoring"""
        result = {"warnings": [], "errors": [], "recommendations": []}
        
        elements_with_anchors = 0
        elements_with_pages = 0
        
        for element in elements:
            metadata = getattr(element, "metadata", {}) or {}
            
            if metadata.get("page_number"):
                elements_with_pages += 1
            
            if metadata.get("id") or metadata.get("anchor"):
                elements_with_anchors += 1
        
        total_elements = len(elements)
        
        # Page number coverage
        page_coverage = elements_with_pages / total_elements if total_elements > 0 else 0
        if page_coverage < 0.8:
            result["warnings"].append(f"Only {page_coverage:.1%} of elements have page numbers")
            result["recommendations"].append("Ensure document parser preserves page number information")
        
        # Anchor coverage
        anchor_coverage = elements_with_anchors / total_elements if total_elements > 0 else 0
        if anchor_coverage < 0.5:
            result["warnings"].append(f"Only {anchor_coverage:.1%} of elements have anchor IDs")
            result["recommendations"].append("Consider using a parser that generates element IDs for better anchoring")
        
        return result
    
    def _validate_technical_content(self, elements: List[Any]) -> Dict[str, Any]:
        """Validate technical content specific to gear/bearing analysis"""
        result = {"warnings": [], "errors": [], "recommendations": [], "indicators": {}}
        
        # Combine all text for analysis
        full_text = ""
        for element in elements:
            text = getattr(element, "text", "") or ""
            full_text += " " + text.lower()
        
        # Check for technical indicators
        technical_indicators = {
            "measurements": self._count_measurements(full_text),
            "timestamps": self._count_timestamps(full_text),
            "gear_terms": self._count_gear_terms(full_text),
            "failure_terms": self._count_failure_terms(full_text),
            "figures_mentioned": self._count_figure_references(full_text)
        }
        
        result["indicators"] = technical_indicators
        
        # Validation based on indicators
        if technical_indicators["measurements"] < 5:
            result["warnings"].append("Document contains few measurement values for technical analysis")
        
        if technical_indicators["gear_terms"] < 3:
            result["warnings"].append("Document may not be gear/bearing related - few domain terms found")
        
        if technical_indicators["timestamps"] == 0:
            result["recommendations"].append("Consider adding timeline information for better analysis")
        
        if technical_indicators["figures_mentioned"] == 0:
            result["recommendations"].append("Visual aids (figures/charts) would enhance technical documentation")
        
        return result
    
    def _count_measurements(self, text: str) -> int:
        """Count measurement values in text"""
        patterns = [
            r'\\d+(?:\\.\\d+)?\\s*(?:μm|um|mm|cm|m|MPa|GPa|RPM|Hz|°C|N|kN)',
            r'\\d+(?:\\.\\d+)?\\s*(?:microns?|millimeters?|centimeters?|meters?)',
            r'\\d+(?:\\.\\d+)?\\s*(?:degrees?|rpm|hertz)'
        ]
        
        total_count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            total_count += len(matches)
        
        return total_count
    
    def _count_timestamps(self, text: str) -> int:
        """Count timestamp/date references in text"""
        patterns = [
            r'\\b\\d{4}[-/]\\d{1,2}[-/]\\d{1,2}\\b',  # YYYY-MM-DD
            r'\\b\\d{1,2}[-/]\\d{1,2}[-/]\\d{4}\\b',  # MM-DD-YYYY
            r'\\b\\d{1,2}:\\d{2}(?::\\d{2})?\\b',     # Time stamps
        ]
        
        total_count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text)
            total_count += len(matches)
        
        return total_count
    
    def _count_gear_terms(self, text: str) -> int:
        """Count gear/bearing related terms"""
        gear_terms = [
            'gear', 'bearing', 'tooth', 'teeth', 'shaft', 'housing', 'lubrication',
            'mesh', 'contact', 'rotation', 'torque', 'load', 'stress', 'vibration'
        ]
        
        total_count = 0
        for term in gear_terms:
            count = len(re.findall(r'\\b' + term + r's?\\b', text, re.IGNORECASE))
            total_count += count
        
        return total_count
    
    def _count_failure_terms(self, text: str) -> int:
        """Count failure analysis related terms"""
        failure_terms = [
            'wear', 'fatigue', 'crack', 'fracture', 'failure', 'damage', 'deterioration',
            'fault', 'defect', 'breakdown', 'seizure', 'overheat', 'corrosion'
        ]
        
        total_count = 0
        for term in failure_terms:
            count = len(re.findall(r'\\b' + term + r's?\\b', text, re.IGNORECASE))
            total_count += count
        
        return total_count
    
    def _count_figure_references(self, text: str) -> int:
        """Count figure/chart references"""
        patterns = [
            r'\\bfig(?:ure)?\\s*\\d+\\b',
            r'\\btable\\s*\\d+\\b',
            r'\\bchart\\s*\\d+\\b',
            r'\\bdiagram\\s*\\d+\\b'
        ]
        
        total_count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            total_count += len(matches)
        
        return total_count
    
    def validate_batch(self, file_paths: List[Path]) -> Dict[str, Any]:
        """
        Validate multiple documents
        
        Args:
            file_paths: List of document paths
            
        Returns:
            Batch validation results
        """
        batch_results = {
            "total_files": len(file_paths),
            "valid_files": 0,
            "invalid_files": 0,
            "files_with_warnings": 0,
            "results": [],
            "summary": {}
        }
        
        for file_path in file_paths:
            result = self.validate_document(file_path)
            batch_results["results"].append(result)
            
            if result["is_valid"]:
                batch_results["valid_files"] += 1
            else:
                batch_results["invalid_files"] += 1
                
            if result["warnings"]:
                batch_results["files_with_warnings"] += 1
        
        # Generate summary
        batch_results["summary"] = {
            "validation_rate": batch_results["valid_files"] / batch_results["total_files"] if batch_results["total_files"] > 0 else 0,
            "common_issues": self._analyze_common_issues(batch_results["results"]),
            "recommendations": self._generate_batch_recommendations(batch_results["results"])
        }
        
        return batch_results
    
    def _analyze_common_issues(self, results: List[Dict]) -> List[str]:
        """Analyze common issues across validation results"""
        issue_counts = {}
        
        for result in results:
            for error in result["errors"]:
                issue_counts[error] = issue_counts.get(error, 0) + 1
            for warning in result["warnings"]:
                issue_counts[warning] = issue_counts.get(warning, 0) + 1
        
        # Return most common issues
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [issue for issue, count in sorted_issues[:5]]
    
    def _generate_batch_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate recommendations for batch processing"""
        recommendations = set()
        
        for result in results:
            recommendations.update(result["recommendations"])
        
        return list(recommendations)[:10]  # Limit to top 10

def create_validator() -> DocumentValidator:
    """Factory function to create document validator"""
    return DocumentValidator()
