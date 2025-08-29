"""
Reusable UI components and utilities for the Gradio interface.
"""
import pandas as pd
from pathlib import Path
import gradio as gr


def _rows_to_df(rows):
	"""Convert list of row lists to a pandas DataFrame with stable columns."""
	cols = [
		"file",
		"page",
		"section",
		"anchor",
		"words",
		"figure_number",
		"figure_order",
		"table_md",
		"table_csv",
		"image",
		"preview",
	]
	try:
		return pd.DataFrame(rows or [], columns=cols)
	except Exception:
		# Fallback: best-effort DataFrame without strict columns
		return pd.DataFrame(rows or [])


def _extract_table_figure_context(docs):
	"""Return a Markdown-ready preview of the top Table/Figure contexts.
	If a table markdown file was generated, embed its content; otherwise fall back to the chunk text.
	"""
	subset = [d for d in docs if d.metadata.get("section") in ("Table", "Figure")]
	if not subset:
		return "(no table/figure contexts in top candidates)"
	out = []
	for d in subset[:3]:
		head = f"[{d.metadata.get('file_name')} p{d.metadata.get('page')}]"
		md_path = d.metadata.get("table_md_path")
		csv_path = d.metadata.get("table_csv_path")
		if md_path and Path(str(md_path)).exists():
			try:
				content = Path(str(md_path)).read_text(encoding="utf-8")
				link_line = f"(table files: [markdown]({md_path})" + (f" | [csv]({csv_path})" if csv_path else "") + ")"
				out.append(f"{head}\n{link_line}\n\n{content}")
				continue
			except Exception:
				pass
		# Fallback to the chunk page content
		out.append(f"{head}\n{d.page_content[:1000]}")
	return "\n\n---\n\n".join(out)


def _fmt_docs(docs, max_items=8):
	out = []
	for d in docs[:max_items]:
		out.append(f"[{d.metadata.get('file_name')} p{d.metadata.get('page')} {d.metadata.get('section')}]\n{d.page_content[:300]}")
	return "\n\n---\n\n".join(out) if out else "(none)"


def _render_router_info(route: str, top_docs):
	heads = [f"[{d.metadata.get('file_name')} p{d.metadata.get('page')} {d.metadata.get('section')}]" for d in top_docs[:3]]
	return f"Route: {route} | Top contexts: {'; '.join(heads)}"


def _rows_from_docs(_docs, limit: int = 300):
	rows = []
	for d in _docs[:limit]:
		md = d.metadata or {}
		txt = d.page_content or ""
		sec = md.get("section") or md.get("section_type")
		# Build preview consistent with snapshot: Figure/Table label preferred
		prev = ""
		if sec == "Figure":
			prev = str(md.get("figure_label") or "")
			if not prev:
				prev = txt[:200]
		elif sec == "Table":
			prev = str(md.get("table_label") or "")
			if not prev:
				# Try compose from first SUMMARY line
				lines = (txt or "").splitlines()
				summ = None
				for i, ln in enumerate(lines):
					if ln.strip().upper() == "SUMMARY:" and i + 1 < len(lines):
						summ = lines[i + 1].strip(); break
				if summ:
					no = md.get('table_number')
					prev = f"Table {no}: {summ}" if no is not None and str(no).strip() != "" else summ
				else:
					prev = txt[:200]
		else:
			prev = txt[:200]
		rows.append([
			md.get("file_name"),
			md.get("page"),
			md.get("section"),
			md.get("anchor"),
			0 if not txt else len(txt.split()),
			md.get("figure_number") or "",
			md.get("figure_order") or "",
			md.get("table_md_path") or "",
			md.get("table_csv_path") or "",
			md.get("image_path") or "",
			prev,
		])
	return rows


def _rows_for_df(docs, filter_section: str | None, q: str | None, limit: int = 300):
	"""Build rows for the DB Explorer table with light filtering."""
	fs = (filter_section or "").strip()
	qq = (q or "").strip().lower()
	rows = []
	for d in docs:
		md = d.metadata or {}
		if fs and md.get("section") != fs:
			continue
		txt = d.page_content or ""
		sec = md.get("section") or md.get("section_type")
		if sec == "Figure":
			prev = md.get("figure_label") or (txt[:200])
		elif sec == "Table":
			if md.get("table_label"):
				prev = md.get("table_label")
			else:
				lines = (txt or "").splitlines()
				summ = None
				for i, ln in enumerate(lines):
					if ln.strip().upper() == "SUMMARY:" and i + 1 < len(lines):
						summ = lines[i + 1].strip(); break
				prev = (f"Table {md.get('table_number')}: {summ}" if summ and md.get('table_number') else (summ or txt[:200]))
		else:
			prev = (txt[:200])
		if qq and qq not in (txt.lower() + " " + " ".join(map(str, md.values())).lower()):
			continue
		rows.append([
			md.get("file_name"),
			md.get("page"),
			md.get("section"),
			md.get("anchor"),
			0 if not txt else len(txt.split()),
			md.get("figure_number") or "",
			md.get("figure_order") or "",
			md.get("table_md_path") or "",
			md.get("table_csv_path") or "",
			md.get("image_path") or "",
			prev,
		])
		if len(rows) >= limit:
			break
	return rows


def _fig_sort_key(d):
    """Sort key for figures by number, order, and page."""
    md = d.metadata or {}
    fn = md.get("figure_number")
    fo = md.get("figure_order")
    pg = md.get("page")
    try:
        fnv = int(fn) if fn is not None and str(fn).strip().isdigit() else 10**9
    except Exception:
        fnv = 10**9
    try:
        fov = int(fo) if fo is not None and str(fo).strip().isdigit() else 10**9
    except Exception:
        fov = 10**9
    try:
        pgv = int(pg) if pg is not None and str(pg).strip().isdigit() else 10**9
    except Exception:
        pgv = 10**9
    an = str(md.get("anchor") or "")
    return (fnv, fov, pgv, an)


def sort_figure_docs(fig_docs):
    """Sort figure documents by number, order, and page."""
    try:
        return sorted(fig_docs, key=_fig_sort_key)
    except Exception:
        return fig_docs
