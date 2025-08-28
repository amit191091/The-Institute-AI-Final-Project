import os
from typing import List, Dict, Any
from langchain.schema import Document
from RAG.app.logger import get_logger
from RAG.app.retrieve_modules.retrieve_query_analyzer import query_analyzer
from RAG.app.config import settings


def _add_wear_depth_fallback(q: str, candidates: List[Document]) -> List[Document]:
    """Add wear depth table data if missing from candidates."""
    analysis = query_analyzer(q)
    if analysis.get("question_type") == "wear_depth_question" or "wear depth" in q.lower():
        # Check if we have any table data in candidates
        has_table_data = any(
            any(case in c.page_content.lower() for case in settings.query_analysis.WEAR_CASES_MAIN)
            for c in candidates
        )
        
        # Check if we have the specific measurement data
        has_measurement_data = any(
            any(measurement in c.page_content.lower() for measurement in settings.fallback.WEAR_MEASUREMENTS)
            for c in candidates
        )
        
        if not has_table_data or not has_measurement_data:
            # Import the global docs to get table data
            try:
                from RAG.app.pipeline import build_pipeline
                from RAG.app.pipeline_modules.pipeline_ingestion import discover_input_paths
                paths = discover_input_paths()
                all_docs, _, _ = build_pipeline(paths)
                
                # Find table documents with wear depth data
                table_docs = []
                for doc in all_docs:
                    content = doc.page_content.lower()
                    if any(case in content for case in settings.query_analysis.WEAR_CASES_MAIN):  # Use main wear cases
                        # Check if it contains the specific wear case from the question
                        for case in settings.query_analysis.WEAR_CASES:
                            if case in q.lower() and case in content:
                                table_docs.append(doc)
                                break
                
                # Also look specifically for measurement data
                measurement_docs = []
                for doc in all_docs:
                    content = doc.page_content.lower()
                    if any(measurement in content for measurement in settings.fallback.WEAR_MEASUREMENTS):
                        measurement_docs.append(doc)
                
                # Add the table docs to candidates if found
                if table_docs:
                    candidates.extend(table_docs[:settings.fallback.MAX_TABLE_DOCS])
                if measurement_docs:
                    candidates.extend(measurement_docs[:settings.fallback.MAX_MEASUREMENT_DOCS])
            except Exception as e:
                # If fallback fails, continue with original candidates
                pass
    
    return candidates


def _add_speed_fallback(q: str, candidates: List[Document]) -> List[Document]:
    """Add speed data if missing from candidates."""
    if any(speed_term in q.lower() for speed_term in settings.fallback.SPEED_QUERY_TERMS):
        # Check if we have speed data in candidates
        has_speed_data = any(
            any(speed in c.page_content.lower() for speed in settings.fallback.SPEED_DATA)
            for c in candidates
        )
        
        if not has_speed_data:
            try:
                from RAG.app.pipeline import build_pipeline
                from RAG.app.pipeline_modules.pipeline_ingestion import discover_input_paths
                paths = discover_input_paths()
                all_docs, _, _ = build_pipeline(paths)
                
                # Find documents with speed data
                speed_docs = []
                for doc in all_docs:
                    content = doc.page_content.lower()
                    if any(speed in content for speed in settings.fallback.SPEED_DATA):
                        speed_docs.append(doc)
                
                if speed_docs:
                    candidates.extend(speed_docs[:settings.fallback.MAX_SPEED_DOCS])
            except Exception as e:
                pass
    
    return candidates


def _add_accelerometer_fallback(q: str, candidates: List[Document]) -> List[Document]:
    """Add accelerometer data if missing from candidates."""
    if any(accel_term in q.lower() for accel_term in settings.fallback.ACCELEROMETER_QUERY_TERMS):
        # Check if we have accelerometer data in candidates
        has_accel_data = any(
            any(accel in c.page_content.lower() for accel in settings.fallback.ACCELEROMETER_DATA)
            for c in candidates
        )
        
        if not has_accel_data:
            try:
                from RAG.app.pipeline import build_pipeline
                from RAG.app.pipeline_modules.pipeline_ingestion import discover_input_paths
                paths = discover_input_paths()
                all_docs, _, _ = build_pipeline(paths)
                
                # Find documents with accelerometer data
                accel_docs = []
                for doc in all_docs:
                    content = doc.page_content.lower()
                    if any(accel in content for accel in settings.fallback.ACCELEROMETER_DATA):
                        accel_docs.append(doc)
                
                if accel_docs:
                    candidates.extend(accel_docs[:settings.fallback.MAX_ACCELEROMETER_DOCS])
            except Exception as e:
                pass
    
    return candidates


def _add_threshold_fallback(q: str, candidates: List[Document]) -> List[Document]:
    """Add threshold data if missing from candidates."""
    if any(threshold_term in q.lower() for threshold_term in settings.fallback.THRESHOLD_QUERY_TERMS):
        # Check if we have threshold data in candidates
        has_threshold_data = any(
            any(threshold in c.page_content.lower() for threshold in settings.fallback.THRESHOLD_DATA)
            for c in candidates
        )
        
        if not has_threshold_data:
            try:
                from RAG.app.pipeline import build_pipeline
                from RAG.app.pipeline_modules.pipeline_ingestion import discover_input_paths
                paths = discover_input_paths()
                all_docs, _, _ = build_pipeline(paths)
                
                # Find documents with threshold data
                threshold_docs = []
                for doc in all_docs:
                    content = doc.page_content.lower()
                    if any(threshold in content for threshold in settings.fallback.THRESHOLD_DATA):
                        threshold_docs.append(doc)
                
                if threshold_docs:
                    candidates.extend(threshold_docs[:settings.fallback.MAX_THRESHOLD_DOCS])
            except Exception as e:
                pass
    
    return candidates
