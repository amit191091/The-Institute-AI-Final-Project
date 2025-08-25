#!/usr/bin/env python3
"""
Backup Database Module
=====================

This module handles backup database functionality for the RAG system.
It provides fallback data when the main report doesn't contain needed information.
"""

import sys
import os
import re
from pathlib import Path
from typing import List, Dict

from langchain.schema import Document
from app.logger import get_logger


def load_backup_database_pdf(pdf_path: str = "../Pictures and Vibrations database/Database figures and tables.pdf") -> List[Document]:
	"""
	Load the backup database PDF containing figures and tables.
	This is used as a fallback when the main report doesn't contain the needed information.
	
	Args:
		pdf_path: Path to the database PDF file
		
	Returns:
		List of Document objects with extracted content
	"""
	try:
		# Check if file exists
		if not Path(pdf_path).exists():
			get_logger().warning(f"Backup database PDF not found at {pdf_path}")
			return []
		
		documents = []
		
		# Open and read the PDF
		with open(pdf_path, 'rb') as file:
			import pypdf
			reader = pypdf.PdfReader(file)
			
			for page_num, page in enumerate(reader.pages, 1):
				text = page.extract_text()
				if not text.strip():
					continue
				
				# Create document with backup metadata
				doc = Document(
					page_content=text,
					metadata={
						"file_name": "Database_figures_and_tables.pdf",
						"page": page_num,
						"section": "Backup_Database",
						"source": "backup_pdf",
						"content_type": "database_tables",
						"anchor": f"backup_page_{page_num}"
					}
				)
				documents.append(doc)
				
				get_logger().info(f"Loaded backup database page {page_num} with {len(text)} characters")
		
		get_logger().info(f"Successfully loaded {len(documents)} pages from backup database PDF")
		return documents
		
	except Exception as e:
		get_logger().error(f"Error loading backup database PDF: {e}")
		return []


def load_backup_database_tables(pdf_path: str = "../Pictures and Vibrations database/Database figures and tables.pdf") -> List[Document]:
	"""
	Extract specific tables from the backup database PDF.
	This creates focused documents for table data.
	
	Args:
		pdf_path: Path to the database PDF file
		
	Returns:
		List of Document objects with table content
	"""
	try:
		# Check if file exists
		if not Path(pdf_path).exists():
			get_logger().warning(f"Backup database PDF not found at {pdf_path}")
			return []
		
		documents = []
		
		# Open and read the PDF
		with open(pdf_path, 'rb') as file:
			import pypdf
			reader = pypdf.PdfReader(file)
			
			for page_num, page in enumerate(reader.pages, 1):
				text = page.extract_text()
				if not text.strip():
					continue
				
				# Extract tables from the page
				lines = text.split('\n')
				current_table = []
				table_started = False
				
				for line in lines:
					line = line.strip()
					
					# Check if this line starts a table
					if re.search(r'Table \d+:|Case.*Wear.*depth|Record.*Number', line, re.I):
						if current_table:
							# Save previous table
							table_text = '\n'.join(current_table)
							if len(table_text.strip()) > 50:  # Minimum table size
								doc = Document(
									page_content=table_text,
									metadata={
										"file_name": "Database_figures_and_tables.pdf",
										"page": page_num,
										"section": "Backup_Table",
										"source": "backup_pdf",
										"content_type": "table_data",
										"anchor": f"backup_table_page_{page_num}"
									}
								)
								documents.append(doc)
						
						# Start new table
						current_table = [line]
						table_started = True
					
					elif table_started and line:
						# Continue current table
						current_table.append(line)
					
					elif table_started and not line:
						# End of table
						table_started = False
						if current_table:
							table_text = '\n'.join(current_table)
							if len(table_text.strip()) > 50:
								doc = Document(
									page_content=table_text,
									metadata={
										"file_name": "Database_figures_and_tables.pdf",
										"page": page_num,
										"section": "Backup_Table",
										"source": "backup_pdf",
										"content_type": "table_data",
										"anchor": f"backup_table_page_{page_num}"
									}
								)
								documents.append(doc)
							current_table = []
				
				# Handle last table on page
				if current_table:
					table_text = '\n'.join(current_table)
					if len(table_text.strip()) > 50:
						doc = Document(
							page_content=table_text,
							metadata={
								"file_name": "Database_figures_and_tables.pdf",
								"page": page_num,
								"section": "Backup_Table",
								"source": "backup_pdf",
								"content_type": "table_data",
								"anchor": f"backup_table_page_{page_num}"
							}
						)
						documents.append(doc)
		
		get_logger().info(f"Successfully extracted {len(documents)} tables from backup database PDF")
		return documents
		
	except Exception as e:
		get_logger().error(f"Error extracting tables from backup database PDF: {e}")
		return []


def load_backup_database() -> List[Document]:
	"""
	Load the backup database PDF for fallback retrieval.
	This is called once during system initialization.
	"""
	try:
		# Load both full pages and extracted tables
		backup_pages = load_backup_database_pdf()
		backup_tables = load_backup_database_tables()
		
		# Combine both types of backup documents
		backup_docs = backup_pages + backup_tables
		
		print(f"✅ Loaded {len(backup_docs)} backup database documents")
		return backup_docs
		
	except Exception as e:
		print(f"⚠️ Warning: Could not load backup database: {e}")
		return []


def should_use_backup_database(query: str) -> bool:
	"""
	Determine if the query should trigger backup database search.
	This is used when the main report doesn't contain the needed information.
	"""
	from app.keywords import get_backup_database_keywords
	
	query_lower = query.lower()
	backup_keywords = get_backup_database_keywords()
	
	# Check if query contains backup keywords
	for keyword in backup_keywords:
		if keyword in query_lower:
			return True
	
	# Check for specific measurement queries
	if any(term in query_lower for term in ["what is", "how much", "value", "measurement", "data"]):
		return True
	
	return False


def search_backup_database(query: str, backup_docs: List[Document], top_k: int = 5) -> List[Document]:
	"""
	Search the backup database PDF for relevant information.
	This is used as a fallback when the main report doesn't contain needed data.
	"""
	if not backup_docs:
		return []
	
	# Simple keyword-based search for backup database
	query_terms = set(re.findall(r'\b\w+\b', query.lower()))
	scored_docs = []
	
	for doc in backup_docs:
		content_lower = doc.page_content.lower()
		content_terms = set(re.findall(r'\b\w+\b', content_lower))
		
		# Calculate overlap
		overlap = len(query_terms & content_terms)
		if overlap > 0:
			# Score based on overlap and content relevance
			score = overlap / len(query_terms) if query_terms else 0
			
			# Boost score for table data if query is about measurements
			if doc.metadata.get("content_type") == "table_data" and any(term in query.lower() for term in ["measurement", "data", "value", "table"]):
				score *= 1.5
			
			scored_docs.append((score, doc))
	
	# Sort by score and return top results
	scored_docs.sort(key=lambda x: x[0], reverse=True)
	return [doc for score, doc in scored_docs[:top_k]]


def retrieve_with_backup_fallback(query: str, main_retriever, backup_docs: List[Document], top_k: int = 5) -> List[Document]:
	"""
	Retrieve documents with fallback to backup database.
	First tries main retriever, then falls back to backup database if needed.
	"""
	# Try main retriever first
	main_results = main_retriever.get_relevant_documents(query)
	
	# Check if main results are sufficient
	if main_results and len(main_results) > 0:
		# Check if main results contain relevant information
		main_content = " ".join([doc.page_content.lower() for doc in main_results])
		query_terms = set(re.findall(r'\b\w+\b', query.lower()))
		
		# If main results don't contain key query terms, try backup
		missing_terms = [term for term in query_terms if term not in main_content]
		if len(missing_terms) > len(query_terms) * 0.5:  # If more than 50% of terms are missing
			get_logger().info(f"Main results missing key terms: {missing_terms}, trying backup database")
			
			# Check if backup database should be used
			if should_use_backup_database(query):
				backup_results = search_backup_database(query, backup_docs, top_k=3)
				if backup_results:
					get_logger().info(f"Found {len(backup_results)} relevant results in backup database")
					# Combine main and backup results, prioritizing main results
					combined_results = main_results[:top_k-2] + backup_results[:2]
					return combined_results[:top_k]
	
	# If main results are sufficient or backup not needed, return main results
	return main_results[:top_k]
