from pathlib import Path
from typing import List, Dict
import re
import os
from types import SimpleNamespace
from app.logger import get_logger

# Constants
TABLE_HEADER_TERMS = ["case", "wear", "depth", "sensor", "value", "feature"]
TECHNICAL_TERMS = ["gear", "vibration", "wear", "failure", "damage", "spall", "fatigue", "bearing", "shaft", "tooth"]
FIGURE_CAPTION_MIN_LENGTH = 10
FIGURE_CAPTION_MAX_LENGTH = 100
TABLE_PREVIEW_LINES = 3
DEFAULT_TABLE_SUMMARY = "Data table"
WEAR_TABLE_SUMMARY = "Wear measurement data table"
SENSOR_TABLE_SUMMARY = "Sensor configuration table"
CASE_TABLE_SUMMARY = "Case study data table"


def _generate_table_summary(text: str) -> str:
	"""Generate a semantic summary for a table based on its content."""
	import re
	lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
	if not lines:
		return DEFAULT_TABLE_SUMMARY
	
	# Look for table title or caption
	for line in lines[:TABLE_PREVIEW_LINES]:
		if re.search(r"\btable\s*\d+", line, re.I):
			# Extract everything after "Table X:"
			match = re.search(r"\btable\s*\d+[:\.\-\s]*(.+)", line, re.I)
			if match:
				return f"Table: {match.group(1).strip()}"
	
	# Look for column headers to infer content
	header_indicators = []
	for line in lines[:TABLE_PREVIEW_LINES]:
		if any(term in line.lower() for term in TABLE_HEADER_TERMS):
			header_indicators.extend(re.findall(r"\b(case|wear|depth|sensor|value|feature|measurement|data)\w*\b", line.lower()))
	
	if "wear" in header_indicators:
		return WEAR_TABLE_SUMMARY
	elif "sensor" in header_indicators:
		return SENSOR_TABLE_SUMMARY
	elif "case" in header_indicators:
		return CASE_TABLE_SUMMARY
	elif header_indicators:
		return f"Data table ({', '.join(set(header_indicators[:3]))})"
	
	return DEFAULT_TABLE_SUMMARY


def _generate_figure_summary(page_text: str, page_num: int, img_index: int) -> str:
	"""Generate a semantic summary for a figure based on surrounding text."""
	import re
	
	# First, try to find the actual figure number from the page text
	# Look for "Figure X:" patterns and extract the actual figure number
	fig_number_patterns = [
		r"\bfigure\s*(\d+)[:\.\-\s]*([^\n\r]+)",
		r"\bfig\.?\s*(\d+)[:\.\-\s]*([^\n\r]+)"
	]
	
	for pattern in fig_number_patterns:
		matches = re.findall(pattern, page_text, re.I)
		if matches:
			# Use the first match found (most likely the correct figure)
			fig_num = matches[0][0]
			caption = matches[0][1].strip()
			if len(caption) > FIGURE_CAPTION_MIN_LENGTH:
				return f"Figure {fig_num}: {caption[:FIGURE_CAPTION_MAX_LENGTH]}"
	
	# Fallback: Look for figure captions near this image
	fig_patterns = [
		rf"\bfig\.?\s*{img_index}\b[:\.\-\s]*([^\n\r]+)",
		rf"\bfigure\s*{img_index}\b[:\.\-\s]*([^\n\r]+)",
		r"\bfig\.?\s*\d+[:\.\-\s]*([^\n\r]+)",
		r"\bfigure\s*\d+[:\.\-\s]*([^\n\r]+)"
	]
	
	for pattern in fig_patterns:
		matches = re.findall(pattern, page_text, re.I)
		if matches:
			caption = matches[0].strip()
			if len(caption) > FIGURE_CAPTION_MIN_LENGTH:  # Reasonable caption length
				return f"Figure {img_index}: {caption[:FIGURE_CAPTION_MAX_LENGTH]}"
	
	# Look for technical context on the page
	technical_terms = re.findall(r"\b(" + "|".join(TECHNICAL_TERMS) + r")\w*\b", page_text.lower())
	if technical_terms:
		unique_terms = list(set(technical_terms[:3]))
		return f"Figure {img_index}: Technical diagram ({', '.join(unique_terms)})"
	
	return f"Figure {img_index}: Image from page {page_num}"


# Placeholder for future LlamaIndex integration
def _try_llamaindex_extraction(path: Path):
	"""Future: Use LlamaIndex/LlamaParse for enhanced PDF parsing.
	This function is a placeholder for when we want to integrate LlamaIndex
	as an alternative to the current extraction pipeline.
	"""
	# TODO: Implement LlamaIndex integration
	# - Check for LLAMA_CLOUD_API_KEY
	# - Use LlamaParse for better table/figure detection
	# - Convert results to our element format
	# - Fall back to current extractors if unavailable
	return []

try:
	from unstructured.partition.pdf import partition_pdf
	from unstructured.partition.docx import partition_docx
	from unstructured.partition.text import partition_text
except Exception:  # pragma: no cover - allow import even if extras missing
	partition_pdf = partition_docx = partition_text = None  # type: ignore


def _export_tables_to_files(elements, path: Path) -> None:
	"""Persist detected table elements to data/elements as Markdown and CSV files.
	Attach file paths back into element.metadata as table_md_path/table_csv_path when possible.
	"""
	base_dir = Path("data") / "elements"
	base_dir.mkdir(parents=True, exist_ok=True)
	for i, e in enumerate(elements, start=1):
		if str(getattr(e, "category", "")).lower() != "table":
			continue
		text = (getattr(e, "text", "") or "").strip()
		if not text:
			continue
		# Determine file stems
		stem = f"{path.stem}-table-{i}"
		md_file = base_dir / f"{stem}.md"
		csv_file = base_dir / f"{stem}.csv"
		# Heuristic: if looks like markdown table, write to md; else if CSV-like, write csv; otherwise write md anyway
		looks_markdown = text.lstrip().startswith("|") and "|" in text
		looks_csv = "," in text and "\n" in text and not looks_markdown
		try:
			if looks_markdown:
				md_file.write_text(text, encoding="utf-8")
			elif looks_csv:
				csv_file.write_text(text, encoding="utf-8")
			else:
				md_file.write_text(text, encoding="utf-8")
		except Exception:
			continue
		# Attach paths to element metadata if possible
		md_obj = getattr(e, "metadata", None)
		if md_obj is not None:
			try:
				# Some metadata are SimpleNamespace; set attributes dynamically
				if looks_markdown or not looks_csv:
					setattr(md_obj, "table_md_path", str(md_file))
				if looks_csv:
					setattr(md_obj, "table_csv_path", str(csv_file))
			except Exception:
				pass


def _pdf_fallback_elements(path: Path):
	"""Lightweight PDF parsing via pypdf as a fallback for Windows.
	Produces simple elements with .text, .category, and .metadata(page_number,id).
	"""
	try:
		from pypdf import PdfReader
	except Exception as e:  # pragma: no cover
		raise RuntimeError("pypdf not installed; cannot parse PDF without unstructured") from e

	reader = PdfReader(str(path))
	elements = []
	for pi, page in enumerate(reader.pages, start=1):
		try:
			text = page.extract_text() or ""
		except Exception:
			text = ""
		# split by blank lines into blocks
		blocks = re.split(r"\n\s*\n", text) if text else []
		if not blocks:
			# single block fallback
			blocks = [text]
		for bi, block in enumerate(blocks, start=1):
			b = block.strip()
			if not b:
				continue
			head = b.splitlines()[0].strip().lower()
			cat = "Text"
			if head.startswith("table") or "|" in b or re.search(r"\btable\b", b, re.I):
				cat = "Table"
			elif head.startswith("figure") or re.search(r"\bfigure\b|\bfig\.\b", b, re.I):
				cat = "Figure"
			el = SimpleNamespace(
				text=b,
				category=cat,
				metadata=SimpleNamespace(page_number=pi, id=f"{path.name}-p{pi}-b{bi}"),
			)
			elements.append(el)
	if not elements:
		# ensure at least one element exists
		elements.append(SimpleNamespace(text="", category="Text", metadata=SimpleNamespace(page_number=1, id=f"{path.name}-p1-b1")))
	return elements


def load_jsonl(path: Path):
	"""Yield elements from a JSONL with keys like page_content / text / metadata."""
	import json
	elements = []
	with path.open("r", encoding="utf-8", errors="ignore") as f:
		for i, line in enumerate(f, 1):
			try:
				rec = json.loads(line)
			except Exception:
				continue
			txt = rec.get("page_content") or rec.get("text") or ""
			meta = rec.get("metadata") or {}
			elements.append(SimpleNamespace(
				text=str(txt),
				category="Text",
				metadata=SimpleNamespace(**meta, id=f"jsonl-{path.name}-{i}")
			))
	return elements


def load_csv(path: Path):
	"""Load CSV files and convert them to document elements."""
	import csv
	import pandas as pd
	
	elements = []
	
	try:
		# Try using pandas first for better handling
		df = pd.read_csv(path)
		
		# Convert DataFrame to text representation
		csv_text = df.to_string(index=False)
		
		# Create a single element for the entire CSV
		elements.append(SimpleNamespace(
			text=csv_text,
			category="Table",
			metadata=SimpleNamespace(
				id=f"csv-{path.name}",
				file_name=path.name,
				page=1,
				section="Table",
				source="csv",
				rows=len(df),
				columns=len(df.columns),
				column_names=list(df.columns)
			)
		))
		
		# Also create individual row elements for better searchability
		for i, row in df.iterrows():
			row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
			if row_text.strip():
				elements.append(SimpleNamespace(
					text=row_text,
					category="Text",
					metadata=SimpleNamespace(
						id=f"csv-{path.name}-row-{i+1}",
						file_name=path.name,
						page=1,
						section="Table",
						source="csv",
						row_number=i+1
					)
				))
				
	except Exception as e:
		# Fallback: basic CSV reading
		pass
	
	return elements


def load_elements(path: Path):
	"""Return Unstructured elements for PDF/DOCX/TXT with page metadata kept."""
	# Import settings at the top of the function
	try:
		from app.pdf_extractions_settings import pdf_settings
	except ImportError:
		# Fallback to environment variables
		class FallbackSettings:
			RAG_PDF_HI_RES = os.getenv("RAG_PDF_HI_RES", "1").lower() not in ("0", "false", "no")
			RAG_USE_TABULA = os.getenv("RAG_USE_TABULA", "").lower() in ("1", "true", "yes")
			RAG_USE_CAMELOT = os.getenv("RAG_USE_CAMELOT", "").lower() in ("1", "true", "yes")
			RAG_SYNTH_TABLES = os.getenv("RAG_SYNTH_TABLES", "").lower() in ("1", "true", "yes")
			RAG_EXTRACT_IMAGES = os.getenv("RAG_EXTRACT_IMAGES", "").lower() in ("1", "true", "yes")
			RAG_USE_PDFPLUMBER = os.getenv("RAG_USE_PDFPLUMBER", "").lower() in ("1", "true", "yes")
		pdf_settings = FallbackSettings()
	
	ext = path.suffix.lower()
	if ext == ".pdf":
		els = None
		
		if partition_pdf is not None:
			# Optional env toggle to skip hi_res completely
			use_hi_res = pdf_settings.RAG_PDF_HI_RES
			if use_hi_res:
				# Pass OCR language(s) using preferred 'languages' kw when available; fallback to 'ocr_languages'
				lang = (os.getenv("RAG_OCR_LANG", "eng") or "eng").strip()
				lang_pref = [s.strip() for s in lang.replace("+", ",").split(",") if s.strip()]
				try:
					els = partition_pdf(
						filename=str(path),
						strategy="hi_res",
						infer_table_structure=True,
						languages=lang_pref,
					)
				except TypeError:
					try:
						els = partition_pdf(
							filename=str(path),
							strategy="hi_res",
							infer_table_structure=True,
							ocr_languages=lang,
						)
					except Exception as e:
						pass
				except Exception as e:
					pass
			# Try standard parser with/without table inference flags
			if els is None:
				for kwargs in (
					{"filename": str(path), "pdf_infer_table_structure": True},
					{"filename": str(path)},
				):
					try:
						els = partition_pdf(**kwargs)  # type: ignore[arg-type]
						break
					except TypeError:
						# Ignore bad kw on older versions
						continue
					except Exception as e:
						pass
		
		# IMPORTANT: Always try PDFPlumber first, even if unstructured failed
		# This ensures we get tables even when unstructured fails
		if pdf_settings.RAG_USE_PDFPLUMBER:
			_pdfp = _try_pdfplumber_tables(path)
			if _pdfp:
				if els is None:
					els = []
				els.extend(_pdfp)
		
		# Fallback: basic text extraction per page using pypdf
		if els is None:
			try:
				from pypdf import PdfReader
			except Exception as e:  # pragma: no cover
				raise RuntimeError("No PDF parser available. Install unstructured[all-docs] or pypdf.") from e
			reader = PdfReader(str(path))
			els = []
			for i, page in enumerate(reader.pages, start=1):
				text = page.extract_text() or ""
				# Create a minimal shim object with .text, .category, .metadata
				class _Shim:
					def __init__(self, text, page_number):
						self.text = text
						self.category = "Text"
						class MD:
							def __init__(self, page_number):
								self.page_number = page_number
								self.id = f"p{i}"
						self.metadata = MD(page_number)
				els.append(_Shim(text, i))
		
		# Optional: enrich with Tabula tables
		if pdf_settings.RAG_USE_TABULA:
			_added = _try_tabula_tables(path)
			if _added:
				els.extend(_added)

		# Optional: Camelot table extraction
		if pdf_settings.RAG_USE_CAMELOT:
			_cam = _try_camelot_tables(path)
			if _cam:
				els.extend(_cam)

		# Optional: synthesize table-like elements from text blocks if none detected
		if pdf_settings.RAG_SYNTH_TABLES:
			try:
				_synth = _synthesize_tables_from_text(els, path)
				if _synth:
					els.extend(_synth)
			except Exception:
				pass
		
		# Optional: extract images as figures via PyMuPDF
		if pdf_settings.RAG_EXTRACT_IMAGES:
			_figs = _try_extract_images(path)
			if _figs:
				els.extend(_figs)

		# Optional: export tables to markdown/csv files and attach paths in metadata
		try:
			_export_tables_to_files(els, path)
		except Exception:
			pass
		
		# Log what we extracted
		if els:
			categories = {}
			for el in els:
				cat = getattr(el, "category", "Unknown")
				categories[cat] = categories.get(cat, 0) + 1
		
		return els
	if ext in (".docx", ".doc"):
		if partition_docx is None:
			raise RuntimeError("unstructured[all-docs] not installed for DOCX parsing")
		return partition_docx(filename=str(path))
	if ext in (".txt",):
		if partition_text is None:
			raise RuntimeError("unstructured not installed for text parsing")
		return partition_text(filename=str(path))
	if ext in (".jsonl",):
		return load_jsonl(path)
	if ext in (".csv",):
		return load_csv(path)
	raise ValueError(f"Unsupported format: {ext}")


def load_many(paths: List[Path]):
	for p in paths:
		els = load_elements(p)
		yield p, els


def _synthesize_tables_from_text(els, path: Path):
	"""Heuristically detect table-like blocks in text elements and produce Table elements.
	Useful when Tabula/hi_res are unavailable or PDF is mostly text with delimited rows.
	Now with improved filtering to avoid page headers and simple text.
	"""
	try:
		import re as _re
	except Exception:
		return []
	out = []
	count = 0
	for e in els:
		cat = str(getattr(e, "category", "") or "").lower()
		if cat in ("table",):
			continue
		text = (getattr(e, "text", "") or "").strip()
		if not text:
			continue
		
		# Skip obvious page headers and simple text
		lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
		if not lines or len(lines) < 2:  # Reduced minimum lines for technical tables
			continue
			
		# Skip page headers like "5 | P a g e"
		if len(lines) <= 2 and any(_re.match(r"^\d+\s*\|\s*P\s+a\s+g\s+e\s*$", ln, _re.I) for ln in lines):
			continue
			
		# Skip if most content is just page numbers or simple headers
		non_header_lines = [ln for ln in lines if not _re.match(r"^\d+\s*\|\s*P\s+a\s+g\s+e", ln, _re.I)]
		if len(non_header_lines) < 1:  # Reduced minimum
			continue
			
		head = lines[0].lower()
		# More specific table header detection
		looks_like_header = _re.search(r"\btable\s*\d+\b", head) and ":" in text
		
		# Count meaningful separators (ignore single | in page headers)
		sep_counts = sum(("\t" in ln) + ("," in ln) + (ln.count("|") >= 2) for ln in lines[:8])  # Reduced pipe count
		wide_cols = sum(1 for ln in lines[:8] if _re.search(r"\S\s{2,}\S.*\S\s{2,}\S", ln))  # Reduced spacing requirement
		
		# More data-like patterns (numbers, units, technical terms)
		has_data_patterns = any(_re.search(r"\b\d+\.?\d*\s*(mm|μm|MPa|Hz|°C|N|kN|rpm|%|ratio|module)\b", ln, _re.I) for ln in lines[:5])
		has_multiple_columns = any(ln.count("|") >= 2 or ln.count("\t") >= 1 for ln in lines)  # Reduced requirements
		
		# Look for specific technical terms that indicate tables
		has_technical_terms = any(_re.search(r"\b(gear|transmission|ratio|module|teeth|wear|depth|parameter|value|measurement)\b", ln, _re.I) for ln in lines[:3])
		
		# Only synthesize if we have strong evidence of tabular data
		if (looks_like_header or 
		    (sep_counts >= 2 and has_multiple_columns) or  # Reduced requirements
		    (wide_cols >= 1 and has_data_patterns) or  # Reduced requirements
		    (has_data_patterns and has_multiple_columns and len(lines) >= 2) or  # Reduced requirements
		    (has_technical_terms and has_data_patterns)):  # New condition for technical tables
			
			count += 1
			md = getattr(e, "metadata", None)
			page_no = getattr(md, "page_number", None) if md else None
			
			# Generate a better summary for the table
			summary = _generate_table_summary(text)
			
			out.append(
				SimpleNamespace(
					text=text,
					category="Table",
					metadata=SimpleNamespace(
						page_number=page_no, 
						id=f"synth-{path.name}-{count}",
						table_summary=summary
					),
				)
			)
	return out


def _try_tabula_tables(path: Path):
	"""Extract tables using tabula-py into shim elements. Requires Java + tabula.
	Returns a list of SimpleNamespace(text, category='Table', metadata.page_number,id).
	"""
	try:
		# Import the function explicitly to satisfy static analysis across tabula-py versions
		from tabula import read_pdf as tabula_read_pdf  # type: ignore
	except Exception:
		return []
	try:
		dfs = tabula_read_pdf(str(path), pages="all", multiple_tables=True, guess=True, stream=True)
	except Exception as e1:
		try:
			dfs = tabula_read_pdf(str(path), pages="all", multiple_tables=True, guess=True, lattice=True)
		except Exception as e2:
			return []
	elements = []
	idx = 0
	for df in dfs or []:
		try:
			# Prefer markdown; fallback to CSV
			try:
				md_text = df.to_markdown(index=False)  # type: ignore[attr-defined]
			except Exception:
				# pandas DataFrame supports to_csv; add ignore for static checkers
				md_text = df.to_csv(index=False)  # type: ignore[call-arg]
			idx += 1
			elements.append(
				SimpleNamespace(
					text=md_text,
					category="Table",
					metadata=SimpleNamespace(page_number=None, id=f"tabula-{path.name}-{idx}"),
				)
			)
		except Exception:
			continue
	return elements


def _try_pdfplumber_tables(path: Path):
	"""Extract simple tables using pdfplumber per page. No external deps beyond pdfminer.six.
	Returns shim elements with markdown/CSV-ish text.
	"""
	try:
		import pdfplumber  # type: ignore
	except Exception:
		return []
	
	elements = []
	
	try:
		with pdfplumber.open(str(path)) as pdf:
			for pno, page in enumerate(pdf.pages, 1):
				page_tables = []
				
				# Method 1: Try standard PDFPlumber table extraction
				try:
					page_tables = page.extract_tables()
				except Exception:
					pass
				
				# Method 2: Try to find tables by looking for table-like structures
				if not page_tables:
					text = page.extract_text() or ""
					# More aggressive table detection for technical documents
					if any(keyword in text.lower() for keyword in ["table", "parameter", "value", "measurement", "gear", "module", "ratio", "transmission", "teeth", "wear", "depth"]):
						lines = text.split('\n')
						table_data = []
						
						for line in lines:
							# Look for lines with multiple columns (separated by spaces or tabs)
							if len(line.split()) >= 3 and any(char in line for char in ['|', '\t', '  ']):
								table_data.append(line)
						
						if len(table_data) >= 2:
							page_tables.append(table_data)
				
				# Method 3: Look for specific technical table patterns
				if not page_tables:
					text = page.extract_text() or ""
					table_patterns = [
						r"Table\s+\d+[:\-]\s*(.+?)(?:\n|$)",
						r"Parameter\s+Value\s+Unit",
						r"Gear\s+Module\s+Teeth",
						r"Transmission\s+Ratio",
						r"Wear\s+Depth\s+Measurement"
					]
					for pattern in table_patterns:
						matches = re.findall(pattern, text, re.I)
						if matches:
							# Extract surrounding lines as table data
							lines = text.split('\n')
							table_data = []
							for i, line in enumerate(lines):
								if any(match.lower() in line.lower() for match in matches):
									# Add this line and a few surrounding lines
									start = max(0, i-2)
									end = min(len(lines), i+3)
									table_data.extend(lines[start:end])
									break
							if table_data:
								page_tables.append(table_data)
				
				# Process extracted tables
				for idx, table_rows in enumerate(page_tables):
					if len(table_rows) < 2 or len(table_rows[0]) < 2:
						continue  # Skip very small tables
					
					try:
						# Convert table to text representation
						table_text = "\n".join(table_rows)
						
						# Generate table summary
						table_summary = _generate_table_summary(table_text)
						
						# Create table element
						table_element = SimpleNamespace(
							text=table_text,
							category="Table",
							metadata=SimpleNamespace(
								page_number=pno,
								id=f"table-{path.name}-p{pno}-t{idx}",
								table_summary=table_summary,
								source="pdfplumber"
							)
						)
						elements.append(table_element)
					except Exception as e:
						pass
		
		return elements
	
	except Exception as e:
		return []


def _try_camelot_tables(path: Path):
	"""Extract tables using Camelot if available. Requires Ghostscript/Poppler; optional.
	"""
	try:
		import camelot  # type: ignore
	except Exception:
		return []
	elements = []
	idx = 0
	for flavor in ("lattice", "stream"):
		try:
			# pages='all' can be slow; try '1-end' style
			tables = camelot.read_pdf(str(path), pages="all", flavor=flavor)
		except Exception as e:
			continue
		for t in getattr(tables, "df", []) or []:
			try:
				# t is DataFrame when accessed as tables[i].df; iterate tables properly
				pass
			except Exception:
				pass
		# Camelot returns a TableList; iterate explicitly
		for ti, tbl in enumerate(tables, start=1):
			try:
				df = getattr(tbl, "df", None)
				if df is None:
					continue
				try:
					md_text = df.to_markdown(index=False)  # type: ignore[attr-defined]
				except Exception:
					md_text = df.to_csv(index=False)
				idx += 1
				elements.append(
					SimpleNamespace(
						text=md_text,
						category="Table",
						metadata=SimpleNamespace(page_number=None, id=f"camelot-{flavor}-{path.name}-{idx}"),
					)
				)
			except Exception:
				continue
	return elements


def _try_extract_images(path: Path):
	"""Extract images from PDF pages as Figure elements using PyMuPDF (fitz).
	Saves images to data/images and creates proper Figure elements with OCR summaries.
	"""
	try:
		import fitz  # PyMuPDF
	except Exception:
		return []
	out_dir = Path("data") / "images"
	out_dir.mkdir(parents=True, exist_ok=True)
	elements = []
	try:
		doc = fitz.open(str(path))
		for pno in range(len(doc)):
			page = doc[pno]
			for img_index, img in enumerate(page.get_images(full=True), start=1):
				xref = img[0]
				pix = fitz.Pixmap(doc, xref)
				# Save as PNG with higher resolution
				fname = f"{path.stem}-p{pno+1}-img{img_index}.png"
				fpath = out_dir / fname
				try:
					if pix.alpha:  # convert RGBA to RGB
						pix = fitz.Pixmap(fitz.csRGB, pix)
					
					# Save with original resolution for now (scaling can cause issues)
					pix.save(fpath)
				finally:
					pix = None  # release
				
				# Try to get text around the image for context
				page_text = page.get_text()
				figure_summary = _generate_figure_summary(page_text, pno + 1, img_index)
				
				elements.append(
					SimpleNamespace(
						text=f"[FIGURE] {figure_summary}\nImage file: {fpath}",
						category="Figure",
						metadata=SimpleNamespace(
							page_number=pno + 1, 
							id=f"img-{path.name}-{pno+1}-{img_index}", 
							image_path=str(fpath),
							figure_summary=figure_summary
						),
					)
				)
	except Exception as e:
		return []
	return elements