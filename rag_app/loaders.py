from pathlib import Path
from typing import List
import re
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
		if partition_pdf is not None:
			# Try stronger table detection paths if available
			try:
				return partition_pdf(filename=str(path), strategy="hi_res", infer_table_structure=True)
			except TypeError:
				# Fallback for different versions
				try:
					return partition_pdf(filename=str(path), pdf_infer_table_structure=True)
				except Exception:
					return partition_pdf(filename=str(path))
		# Fallback: basic text extraction per page using pypdf
		try:
			from pypdf import PdfReader
		except Exception as e:  # pragma: no cover
			raise RuntimeError("No PDF parser available. Install unstructured[all-docs] or pypdf.") from e
		reader = PdfReader(str(path))
		elements = []
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
			elements.append(_Shim(text, i))
		get_logger().warning("Using pypdf fallback for PDF parsing (limited structure detection).")
		return elements
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

