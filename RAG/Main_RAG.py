from __future__ import annotations

import os
import sys

# Enable better PDF table extraction - MUST be set before any other imports
os.environ["RAG_USE_PDFPLUMBER"] = "1"
os.environ["RAG_SYNTH_TABLES"] = "1"
os.environ["RAG_PDF_HI_RES"] = "1"
os.environ["RAG_USE_TABULA"] = "0"
os.environ["RAG_USE_CAMELOT"] = "0"
os.environ["RAG_EXTRACT_IMAGES"] = "0"

# Force reload of any already-imported modules that might cache env vars
import importlib
for module_name in ['app.loaders', 'app.pdf_extractions_settings']:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

from pathlib import Path
from typing import List, Tuple
import json
from datetime import datetime

from dotenv import load_dotenv

from app.config import settings
from app.loaders import load_many
from app.chunking import structure_chunks
from app.metadata import attach_metadata
from app.indexing import build_dense_index, build_sparse_retriever, to_documents, dump_chroma_snapshot
from app.retrieve import apply_filters, build_hybrid_retriever, query_analyzer, rerank_candidates
from app.agents import answer_needle, answer_summary, answer_table, route_question
from app.hierarchical import _ask_table_hierarchical
from app.ui_gradio import build_ui
from app.validate import validate_min_pages
from app.logger import get_logger
from app.retrieve import lexical_overlap

def _clean_run_outputs():
    """Delete prior run artifacts so new extraction overwrites files.
    Cleans: data/images, data/elements, logs/queries.jsonl, logs/elements/*.jsonl
    Controlled by env RAG_CLEAN_RUN (default: true).
    """
    flag = os.getenv("RAG_CLEAN_RUN", "1").lower() not in ("0", "false", "no")
    if not flag:
        return
    import shutil
    # Directories
    for d in (Path("data")/"images", Path("data")/"elements"):
        try:
            if d.exists():
                shutil.rmtree(d)
        except Exception:
            pass
    # Optional: clean Chroma persist dir
    try:
        chroma_dir = os.getenv("RAG_CHROMA_DIR")
        if chroma_dir:
            d = Path(chroma_dir)
            if d.exists():
                shutil.rmtree(d)
    except Exception:
        pass
    # Logs: queries.jsonl and logs/elements dumps
    try:
        q = Path("logs")/"queries.jsonl"
        if q.exists():
            q.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass
    try:
        ed = Path("logs")/"elements"
        if ed.exists():
            shutil.rmtree(ed)
    except Exception:
        pass


def _discover_input_paths() -> List[Path]:
    """
    Discover inputs from multiple sources:
    - Main report: "Gear wear Failure.pdf"
    - JSONL sidecar files (db_snapshot.jsonl, queries.jsonl)
    - Picture analysis data (CSV files)
    - Vibration analysis data (CSV files)
    """
    candidates: List[Path] = []
    
    # Main report
    gear_pdf = settings.DATA_DIR / "Gear wear Failure.pdf"
    if gear_pdf.exists():
        candidates.append(gear_pdf)
    else:
        print(f"❌ Gear wear Failure.pdf not found in {settings.DATA_DIR}")
        return []
    
    # JSONL files
    for name in ("db_snapshot.jsonl", "queries.jsonl"):
        p = settings.DATA_DIR / name
        if p.exists():
            candidates.append(p)
    
    # Picture analysis data
    # Try both relative paths to handle different working directories
    picture_data_dirs = [
        Path("../Pictures and Vibrations database/Picture"),
        Path("Pictures and Vibrations database/Picture")
    ]
    
    picture_data_dir = None
    for pdir in picture_data_dirs:
        if pdir.exists():
            picture_data_dir = pdir
            break
    
    if picture_data_dir:
        picture_csv_files = [
            "all_teeth_results.csv",
            "single_tooth_results.csv"
        ]
        for csv_file in picture_csv_files:
            csv_path = picture_data_dir / csv_file
            if csv_path.exists():
                candidates.append(csv_path)
        
        # Ground truth measurements
        ground_truth_path = picture_data_dir / "Picture Tools" / "ground_truth_measurements.csv"
        if ground_truth_path.exists():
            candidates.append(ground_truth_path)
    
    # Vibration analysis data
    # Try both relative paths to handle different working directories
    vibration_data_dirs = [
        Path("../Pictures and Vibrations database/Vibration"),
        Path("Pictures and Vibrations database/Vibration")
    ]
    
    vibration_data_dir = None
    for vdir in vibration_data_dirs:
        if vdir.exists():
            vibration_data_dir = vdir
            break
    
    if vibration_data_dir:
        vibration_csv_files = [
            "RMS15.csv",
            "RMS45.csv", 
            "FME Values.csv",
            "Records.csv"
        ]
        for csv_file in vibration_csv_files:
            csv_path = vibration_data_dir / csv_file
            if csv_path.exists():
                candidates.append(csv_path)
    
    return candidates


def _get_embeddings():
    """Prefer Google embeddings, fallback to OpenAI."""
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except Exception:
        GoogleGenerativeAIEmbeddings = None  # type: ignore
    try:
        from langchain_openai import OpenAIEmbeddings
    except Exception:
        OpenAIEmbeddings = None  # type: ignore

    if settings.GOOGLE_API_KEY and GoogleGenerativeAIEmbeddings is not None:
        return GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL_GOOGLE)
    if settings.OPENAI_API_KEY and OpenAIEmbeddings is not None:
        return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL_OPENAI)
    raise RuntimeError(
        "No embedding backend available. Set GOOGLE_API_KEY or OPENAI_API_KEY in config.py and install langchain-google-genai or langchain-openai."
    )


class _LLM:
    """Simple callable LLM wrapper with configurable preference and fallback via LangChain chat models."""

    def __init__(self) -> None:
        self._backend = None
        self._which = None
        
        # Use preferred LLM based on configuration
        if settings.PREFERRED_LLM == "openai" and settings.OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI
                self._backend = ChatOpenAI(
                    model=settings.OPENAI_MODEL, 
                    temperature=settings.OPENAI_TEMPERATURE
                )
                self._which = "openai"
            except Exception:
                self._backend = None
        
        elif settings.PREFERRED_LLM == "google" and settings.GOOGLE_API_KEY:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._backend = ChatGoogleGenerativeAI(
                    model=settings.GOOGLE_MODEL, 
                    temperature=settings.GOOGLE_TEMPERATURE
                )
                self._which = "google"
            except Exception:
                self._backend = None
        
        # Fallback to the other LLM if preferred one failed
        elif settings.PREFERRED_LLM == "openai" and settings.GOOGLE_API_KEY and self._backend is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._backend = ChatGoogleGenerativeAI(
                    model=settings.GOOGLE_MODEL, 
                    temperature=settings.GOOGLE_TEMPERATURE
                )
                self._which = "google"
            except Exception:
                self._backend = None
        
        elif settings.PREFERRED_LLM == "google" and settings.OPENAI_API_KEY and self._backend is None:
            try:
                from langchain_openai import ChatOpenAI
                self._backend = ChatOpenAI(
                    model=settings.OPENAI_MODEL, 
                    temperature=settings.OPENAI_TEMPERATURE
                )
                self._which = "openai"
            except Exception:
                self._backend = None

    def __call__(self, prompt: str) -> str:
        if self._backend is not None:
            try:
                resp = self._backend.invoke(prompt)
                # LangChain AIMessage has .content
                return getattr(resp, "content", str(resp))
            except Exception as e:  # pragma: no cover
                return f"[LLM error] {e}\n\n{prompt[-400:]}"
        # Final fallback echo
        return "[LLM not configured] " + prompt[-400:]


def build_pipeline(paths: List[Path]):
    log = get_logger()
    records = []
    # ingest
    for path, elements in load_many(paths):
        
        
        # basic ingestion validation: min pages (skip for CSV files)
        if path.suffix.lower() != ".csv":
            try:
                pages_list = [
                    int(getattr(getattr(e, "metadata", None), "page_number", 0))
                    for e in elements
                    if getattr(getattr(e, "metadata", None), "page_number", None) is not None
                ]
                pages = sorted(set(pages_list))
                ok, msg = validate_min_pages(len(set(pages)), settings.MIN_PAGES)
                if not ok:
                    pass
            except Exception:
                pass
        chunks = structure_chunks(elements, str(path))
        
        for ch in chunks:
            records.append(attach_metadata(ch, client_id=os.getenv("CLIENT_ID"), case_id=path.stem))
    # Section histogram after metadata attachment
    sec_hist = {}
    for r in records:
        sec = (r.get("metadata", {}) or {}).get("section")
        sec_hist[sec] = sec_hist.get(sec, 0) + 1
    

    # vectorization
    docs = to_documents(records)
    # Write a quick DB snapshot for debugging
    try:
        Path("logs").mkdir(exist_ok=True)
        snap_path = Path("logs")/"db_snapshot.jsonl"
        with open(snap_path, "w", encoding="utf-8") as f:
            for d in docs:
                md = (d.metadata or {})
                rec = {
                    "file": md.get("file_name"),
                    "page": md.get("page"),
                    "section": md.get("section"),
                    "anchor": md.get("anchor"),
                    "table_md_path": md.get("table_md_path"),
                    "table_csv_path": md.get("table_csv_path"),
                    "image_path": md.get("image_path"),
                    "words": len((d.page_content or "").split()),
                    "preview": (d.page_content or "")[:200],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
    tbl_cnt = sum(1 for d in docs if (d.metadata or {}).get("section") == "Table")
    fig_cnt = sum(1 for d in docs if (d.metadata or {}).get("section") == "Figure")

    embeddings = _get_embeddings()
    dense = build_dense_index(docs, embeddings)
    sparse = build_sparse_retriever(docs, k=settings.SPARSE_K)
    hybrid = build_hybrid_retriever(dense, sparse, dense_k=settings.DENSE_K)
    # If Chroma is persisted, try writing a snapshot
    try:
        if os.getenv("RAG_CHROMA_DIR"):
            dump_chroma_snapshot(dense, Path("logs")/"chroma_snapshot.jsonl")
    except Exception:
        pass
    # expose per-retriever diagnostics
    try:
        dense_ret = dense.as_retriever(search_kwargs={"k": settings.DENSE_K})
    except Exception:
        dense_ret = None
    debug = {"dense": dense_ret, "sparse": sparse}
    return docs, hybrid, debug


def ask(docs, hybrid, llm: _LLM, question: str, ground_truth: str | None = None) -> str:
	qa = query_analyzer(question)
	route = route_question(question)
	
	# For table questions, implement hierarchical retrieval
	if route == "table":
		return _ask_table_hierarchical(docs, hybrid, llm, question, qa)
	
	# For other routes, use standard retrieval
	# Compatible retrieval call
	try:
		candidates = hybrid.get_relevant_documents(question)
	except Exception:
		candidates = hybrid.invoke(question)
	# enforce rerank pool size
	candidates = candidates[: settings.RERANK_TOP_K]
	
	filtered = apply_filters(candidates, qa["filters"])  # metadata filters
	
	# Fallback: if section filter applied but nothing left, pull from all docs with that section
	try:
		sec = qa["filters"].get("section")
	except Exception:
		sec = None
	if sec and not filtered:
		filtered = [d for d in docs if (d.metadata or {}).get("section") == sec]
	
	# Additional fallback: if no documents after filtering, use all candidates
	if not filtered and candidates:
		filtered = candidates
	
	top_docs = rerank_candidates(question, filtered, top_n=settings.CONTEXT_TOP_N)
	
	# Logging: route, filters, counts, scores
	def _doc_head(d):
		md = getattr(d, "metadata", {}) or {}
		return f"{md.get('file_name')} p{md.get('page')} {md.get('section')}#{md.get('anchor', '')}"

	def _score(d):
		base = lexical_overlap(question, d.page_content)
		meta_text = " ".join(map(str, (getattr(d, "metadata", {}) or {}).values()))
		boost = 0.2 * lexical_overlap(" ".join(qa["keywords"]), meta_text)
		return round(base + boost, 4)

	# Route to sub-agent
	if route == "summary":
		ans = answer_summary(llm, top_docs, question)
	else:
		ans = answer_needle(llm, top_docs, question)
	# Append JSONL trace (lightweight)
	try:
		log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
		entry = {
			"ts": datetime.utcnow().isoformat() + "Z",
			"question": question,
			"route": route,
			"keywords": qa["keywords"],
			"filters": qa["filters"],
			"contexts": [
				{
					"file": d.metadata.get("file_name"),
					"page": d.metadata.get("page"),
					"section": d.metadata.get("section"),
					"anchor": d.metadata.get("anchor"),
					"score": _score(d),
				}
				for d in top_docs
			],
			"answer_preview": (ans or "")[:400],
		}
		with open(log_dir / "queries.jsonl", "a", encoding="utf-8") as f:
			f.write(json.dumps(entry, ensure_ascii=False) + "\n")
	except Exception:
		pass
	return ans


def main() -> None:
    load_dotenv()
    
    # Check for evaluation mode
    if len(sys.argv) > 1 and sys.argv[1] == "--evaluate":
        print("Running RAG evaluation mode...")
        _run_evaluation_mode()
        return
    
    _clean_run_outputs()
    paths = _discover_input_paths()
    if not paths:
        print("No input files found. Place PDFs/DOCs under data/ or the root PDF.")
        return
    docs, hybrid, debug = build_pipeline(paths)
    llm = _LLM()
    
    # Launch Gradio UI
    if os.getenv("RAG_HEADLESS", "").lower() in ("1", "true", "yes"):
        print("[HEADLESS] Ingestion complete. Skipping UI launch.")
        return
    else:
        try:
            ui = build_ui(docs, hybrid, llm, debug)
            share = os.getenv("GRADIO_SHARE", "").lower() in ("1", "true", "yes")
            
            # Auto-open browser
            import webbrowser
            import time
            import signal
            
            def signal_handler(signum, frame):
                print("\n🛑 Shutdown signal received. Closing server gracefully...")
                sys.exit(0)
            
            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            print("🚀 Starting RAG system...")
            print("📖 Opening web interface in your browser...")
            print("💡 Press Ctrl+C to stop the server")
            
            # Launch UI in background and open browser after a short delay
            ui.launch(share=share, inbrowser=True)
            
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received. Shutting down gracefully...")
        except Exception as e:
            print(f"UI failed to launch: {e}")
            # fallback single query demo


def _run_evaluation_mode():
    """Run evaluation scripts from command line."""
    from app.evaluation_wrapper import (
        EVAL_AVAILABLE, 
        test_google_api, 
        test_ragas, 
        generate_ground_truth, 
        evaluate_rag
    )
    
    if not EVAL_AVAILABLE:
        print("❌ Evaluation scripts not available")
        print("Make sure all evaluation scripts are in the data/ directory")
        return
    
    print("Available evaluation commands:")
    print("1. test-google-api")
    print("2. test-ragas")
    print("3. generate-ground-truth <num_questions>")
    print("4. evaluate-rag")
    
    if len(sys.argv) < 3:
        print("Usage: python Main_RAG.py --evaluate <command> [args]")
        return
    
    command = sys.argv[2]
    
    if command == "test-google-api":
        print("Testing Google API setup...")
        result = test_google_api()
        print(result)
    
    elif command == "test-ragas":
        print("Testing RAGAS with Google...")
        result = test_ragas()
        print(result)
    
    elif command == "generate-ground-truth":
        if len(sys.argv) < 4:
            print("Usage: python Main_RAG.py --evaluate generate-ground-truth <num_questions>")
            return
        try:
            num_questions = int(sys.argv[3])
            print(f"Generating ground truth dataset with {num_questions} questions...")
            # Note: This would need the pipeline to be built first
            print("⚠️  This requires the RAG pipeline to be built first")
            print("Run without --evaluate flag to build the pipeline first")
        except ValueError:
            print("Invalid number of questions")
    
    elif command == "evaluate-rag":
        print("Evaluating RAG system...")
        # Note: This would need the pipeline to be built first
        print("⚠️  This requires the RAG pipeline to be built first")
        print("Run without --evaluate flag to build the pipeline first")
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: test-google-api, test-ragas, generate-ground-truth, evaluate-rag")


if __name__ == "__main__":
    main()