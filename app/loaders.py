from pathlib import Path
from typing import List
import re
import os
from types import SimpleNamespace
from app.logger import get_logger

try:
	from unstructured.partition.pdf import partition_pdf
	from unstructured.partition.docx import partition_docx
	from unstructured.partition.text import partition_text
except Exception:  # pragma: no cover - allow import even if extras missing
	partition_pdf = partition_docx = partition_text = None  # type: ignore


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
	print(f"[INFO] Using pypdf fallback for {path.name}: produced {len(elements)} elements")
	return elements


def load_elements(path: Path):
	"""Return Unstructured elements for PDF/DOCX/TXT with page metadata kept."""
	ext = path.suffix.lower()
	if ext == ".pdf":
		els = None
		if partition_pdf is not None:
			log = get_logger()
			# Log feature toggles
			log.info(
				"PDF parse toggles: hi_res=%s, tabula=%s, extract_images=%s",
				os.getenv("RAG_PDF_HI_RES", "1"), os.getenv("RAG_USE_TABULA", "0"), os.getenv("RAG_EXTRACT_IMAGES", "0")
			)
			# Optional env toggle to skip hi_res completely
			use_hi_res = (os.getenv("RAG_PDF_HI_RES", "1").lower() not in ("0", "false", "no"))
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
						log.warning("hi_res PDF parsing failed (%s); falling back to standard parser", e.__class__.__name__)
				except Exception as e:
					log.warning("hi_res PDF parsing failed (%s); falling back to standard parser", e.__class__.__name__)
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
						log.warning("standard PDF parsing attempt failed (%s); trying next/fallback", e.__class__.__name__)
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
			get_logger().warning("Using pypdf fallback for PDF parsing (limited structure detection).")
		# Optional: enrich with Tabula tables
		if os.getenv("RAG_USE_TABULA", "").lower() in ("1", "true", "yes"):
			_added = _try_tabula_tables(path)
			if _added:
				els.extend(_added)
				get_logger().info("%s: Tabula extracted %d tables", path.name, len(_added))
		# Optional: synthesize table-like elements from text blocks if none detected
		if os.getenv("RAG_SYNTH_TABLES", "").lower() in ("1", "true", "yes"):
			try:
				_synth = _synthesize_tables_from_text(els, path)
				if _synth:
					els.extend(_synth)
					get_logger().info("%s: Synthesized %d table-like elements from text", path.name, len(_synth))
			except Exception:
				pass
		# Optional: extract images as figures via PyMuPDF
		if os.getenv("RAG_EXTRACT_IMAGES", "").lower() in ("1", "true", "yes"):
			_figs = _try_extract_images(path)
			if _figs:
				els.extend(_figs)
				get_logger().info("%s: Extracted %d images as figures", path.name, len(_figs))
		return els
	if ext in (".docx", ".doc"):
		if partition_docx is None:
			raise RuntimeError("unstructured[all-docs] not installed for DOCX parsing")
		return partition_docx(filename=str(path))
	if ext in (".txt",):
		if partition_text is None:
			raise RuntimeError("unstructured not installed for text parsing")
		return partition_text(filename=str(path))
	raise ValueError(f"Unsupported format: {ext}")


def load_many(paths: List[Path]):
	log = get_logger()
	for p in paths:
		els = load_elements(p)
		# Log tables/figures present in the parsed elements (best-effort)
		try:
			cats = [str(getattr(e, "category", "")).lower() for e in els]
			# histogram by category
			hist = {}
			for c in cats:
				hist[c] = hist.get(c, 0) + 1
			if hist:
				log.info(f"{p.name}: element categories -> {sorted(hist.items(), key=lambda x: (-x[1], x[0]))}")
			tables = [getattr(e, "text", "").strip()[:80] for e in els if str(getattr(e, "category", "")).lower() == "table"]
			figures = [getattr(e, "text", "").strip()[:80] for e in els if str(getattr(e, "category", "")).lower() in ("figure", "image")]
			if tables:
				log.info(f"{p.name}: detected {len(tables)} table elements (sample): {tables[:2]}")
			if figures:
				log.info(f"{p.name}: detected {len(figures)} figure elements (sample): {figures[:2]}")
			# Optional raw elements dump for deep debugging
			import os, json
			if os.getenv("RAG_DUMP_ELEMENTS", "").lower() in ("1", "true", "yes"):
				dump_dir = Path("logs") / "elements"
				dump_dir.mkdir(parents=True, exist_ok=True)
				out_path = dump_dir / f"{p.stem}.jsonl"
				with open(out_path, "w", encoding="utf-8") as f:
					for e in els:
						md = getattr(e, "metadata", None)
						rec = {
							"category": str(getattr(e, "category", "")),
							"page_number": getattr(md, "page_number", None) if md else None,
							"id": getattr(md, "id", None) if md else None,
							"text_head": (getattr(e, "text", "") or "").strip()[:200],
						}
						f.write(json.dumps(rec, ensure_ascii=False) + "\n")
		except Exception:
			pass
		yield p, els


def _synthesize_tables_from_text(els, path: Path):
	"""Heuristically detect table-like blocks in text elements and produce Table elements.
	Useful when Tabula/hi_res are unavailable or PDF is mostly text with delimited rows.
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
		# Heuristics: presence of row delimiters, multiple consecutive spaces forming columns, or a 'Table X' header
		lines = [ln for ln in text.splitlines() if ln.strip()]
		if not lines or len(lines) < 2:
			continue
		head = lines[0].lower()
		looks_like_header = _re.search(r"\btable\s*\d+\b", head)
		sep_counts = sum(("\t" in ln) + ("," in ln) + ("|" in ln) for ln in lines[:8])
		wide_cols = sum(1 for ln in lines[:8] if _re.search(r"\S\s{2,}\S", ln))
		if looks_like_header or sep_counts >= 2 or wide_cols >= 3:
			count += 1
			md = getattr(e, "metadata", None)
			page_no = getattr(md, "page_number", None) if md else None
			out.append(
				SimpleNamespace(
					text=text,
					category="Table",
					metadata=SimpleNamespace(page_number=page_no, id=f"synth-{path.name}-{count}"),
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
		get_logger().debug("Tabula not available; skip table extraction")
		return []
	try:
		dfs = tabula_read_pdf(str(path), pages="all", multiple_tables=True, guess=True, stream=True)
	except Exception as e1:
		get_logger().warning("Tabula stream mode failed (%s); trying lattice", e1.__class__.__name__)
		try:
			dfs = tabula_read_pdf(str(path), pages="all", multiple_tables=True, guess=True, lattice=True)
		except Exception as e2:
			get_logger().warning("Tabula failed to extract tables (%s)", e2.__class__.__name__)
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


def _try_extract_images(path: Path):
	"""Extract images from PDF pages as Figure elements using PyMuPDF (fitz).
	Saves images to logs/images and creates placeholder Figure elements.
	"""
	try:
		import fitz  # PyMuPDF
	except Exception:
		get_logger().debug("PyMuPDF not available; skip image extraction")
		return []
	out_dir = Path("logs") / "images"
	out_dir.mkdir(parents=True, exist_ok=True)
	elements = []
	try:
		doc = fitz.open(str(path))
		for pno in range(len(doc)):
			page = doc[pno]
			for img_index, img in enumerate(page.get_images(full=True), start=1):
				xref = img[0]
				pix = fitz.Pixmap(doc, xref)
				# Save as PNG
				fname = f"{path.stem}-p{pno+1}-img{img_index}.png"
				fpath = out_dir / fname
				try:
					if pix.alpha:  # convert RGBA to RGB
						pix = fitz.Pixmap(fitz.csRGB, pix)
					pix.save(fpath)
				finally:
					pix = None  # release
				elements.append(
					SimpleNamespace(
						text=f"[FIGURE] Extracted image saved at {fpath}",
						category="Figure",
						metadata=SimpleNamespace(page_number=pno + 1, id=f"img-{path.name}-{pno+1}-{img_index}", image_path=str(fpath)),
					)
				)
	except Exception as e:
		get_logger().warning("Image extraction failed (%s)", e.__class__.__name__)
		return []
	return elements

