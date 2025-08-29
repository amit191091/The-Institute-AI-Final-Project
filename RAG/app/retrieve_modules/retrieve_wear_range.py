"""
Specialized Wear Range Retriever
Handles wear depth range queries with high accuracy
"""
import re
from typing import List, Dict, Any, Tuple
from langchain.schema import Document


def extract_wear_data_from_document(doc: Document) -> List[Tuple[str, float]]:
    """
    Extract all wear cases and their depths from a document.
    Returns list of (case_id, depth) tuples.
    """
    content = doc.page_content
    wear_data = []
    
    # Pattern to match wear cases with depths
    # Handles various formats: W1 40, W21 510, etc.
    patterns = [
        r'W(\d+)\s+(\d+(?:\.\d+)?)\s*μm',  # W1 40 μm
        r'W(\d+)\s+(\d+(?:\.\d+)?)',       # W1 40
        r'W(\d+)\s*\|?\s*(\d+(?:\.\d+)?)', # W1 | 40 (table format)
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for case_num, depth_str in matches:
            try:
                case_id = f"W{case_num}"
                depth = float(depth_str)
                wear_data.append((case_id, depth))
            except ValueError:
                continue
    
    return wear_data


def find_wear_cases_in_range(min_depth: float, max_depth: float, docs: List[Document]) -> List[str]:
    """
    Find all wear cases within the specified depth range.
    Returns sorted list of case IDs.
    """
    all_wear_data = []
    
    # Extract wear data from all documents
    for doc in docs:
        wear_data = extract_wear_data_from_document(doc)
        all_wear_data.extend(wear_data)
    
    # Remove duplicates (keep first occurrence)
    seen = set()
    unique_wear_data = []
    for case_id, depth in all_wear_data:
        if case_id not in seen:
            seen.add(case_id)
            unique_wear_data.append((case_id, depth))
    
    # Filter by range and sort
    cases_in_range = []
    for case_id, depth in unique_wear_data:
        if min_depth <= depth <= max_depth:
            cases_in_range.append((case_id, depth))
    
    # Sort by case number
    cases_in_range.sort(key=lambda x: int(x[0][1:]))
    
    return [case_id for case_id, _ in cases_in_range]


def parse_range_query(query: str) -> Tuple[float, float]:
    """
    Parse a range query to extract min and max depths.
    Handles various formats:
    - "greater than 500 μm and less than 650 μm"
    - "between 100 and 450 μm"
    - "500-650 μm"
    """
    query_lower = query.lower()
    
    # Extract numbers from the query
    numbers = re.findall(r'\d+(?:\.\d+)?', query)
    if len(numbers) < 2:
        raise ValueError("Could not find two numbers in range query")
    
    depths = [float(num) for num in numbers]
    
    # Determine which is min and which is max based on query structure
    if any(word in query_lower for word in ["greater than", "more than", "above"]):
        # Format: "greater than X and less than Y"
        if "less than" in query_lower or "below" in query_lower:
            min_depth = depths[0]
            max_depth = depths[1]
        else:
            min_depth = depths[0]
            max_depth = float('inf')
    elif any(word in query_lower for word in ["between", "and"]):
        # Format: "between X and Y"
        min_depth = min(depths)
        max_depth = max(depths)
    else:
        # Default: assume first number is min, second is max
        min_depth = min(depths)
        max_depth = max(depths)
    
    return min_depth, max_depth


def wear_range_retriever(query: str, docs: List[Document]) -> str:
    """
    Main function to handle wear range queries.
    Returns comma-separated list of case IDs.
    """
    try:
        # Parse the range from the query
        min_depth, max_depth = parse_range_query(query)
        
        # Find cases in range
        cases = find_wear_cases_in_range(min_depth, max_depth, docs)
        
        if not cases:
            return "No wear cases found in the specified range"
        
        return ", ".join(cases)
        
    except Exception as e:
        return f"Error processing range query: {str(e)}"


def is_wear_range_query(query: str) -> bool:
    """
    Determine if a query is asking about wear depth ranges.
    """
    query_lower = query.lower()
    
    # Check for range indicators
    range_indicators = [
        "greater than", "less than", "more than", "above", "below",
        "between", "range", "μm", "um", "micron"
    ]
    
    # Check for wear case indicators
    wear_indicators = ["wear case", "wear cases", "case", "cases"]
    
    has_range = any(indicator in query_lower for indicator in range_indicators)
    has_wear = any(indicator in query_lower for indicator in wear_indicators)
    
    return has_range and has_wear
