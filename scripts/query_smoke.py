from pathlib import Path
import os
import sys

# Ensure project root is on sys.path when running from scripts/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline import build_pipeline, _LLM, ask, answer_with_contexts
from app.agents import route_question_ex  # for transparent router tracing


def main():
    # Prefer headless behavior
    os.environ.setdefault("RAG_HEADLESS", "1")
    # Ensure orchestrator-first answering during smoke tests
    os.environ.setdefault("RAG_USE_ORCHESTRATOR", "1")
    os.environ["RAG_USE_NORMALIZED"] = "0"
    # Load .env if present
    try:
        from dotenv import load_dotenv
        try:
            from dotenv import dotenv_values, find_dotenv  # type: ignore
            env_path = find_dotenv(usecwd=True, raise_error_if_not_found=False)
            if env_path:
                for k, v in (dotenv_values(env_path) or {}).items():
                    if v is not None:
                        os.environ[k] = v
        except Exception:
            pass
    except Exception:
        pass
    # Prefer Google if available; otherwise drop invalid OpenAI key to force FakeEmbeddings fallback
    if not os.getenv("GOOGLE_API_KEY"):
        os.environ.pop("OPENAI_API_KEY", None)
    paths = []
    root_pdf = Path("Gear wear Failure.pdf")
    if root_pdf.exists():
        paths.append(root_pdf)
    data_dir = Path("data")
    if data_dir.exists():
        paths.extend(sorted(data_dir.glob("*.pdf")))
    if not paths:
        print("No PDFs found for smoke test.")
        return
    docs, hybrid, _ = build_pipeline(paths)
    llm = _LLM()

    # Allow passing a single question via CLI to test specific cases
    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = [
            "whats the transmission ratio",
            "what is the gear ratio",
            "show me table 3",
            "give me a list of all the cases where the wear depth is an odd number and greater than 234",
            "whats the sensitivity and sample rate of the instrumentation",
            "what is the maximum allowable wear depth",
            "what speed were used in the testings",
        ]
    for q in questions:
        print("\nQ:", q)
        try:
            ans, ctx, tr = answer_with_contexts(docs, hybrid, llm, q)
        except Exception as e:
            ans = f"[error] {e}"
            ctx = []
            tr = None
        print("A:", str(ans)[:400])

        # Router (heuristic) trace for visibility regardless of orchestrator
        try:
            r_route, r_trace = route_question_ex(q)
            print("ROUTER:", r_route, "| matched=", r_trace.get("matched"))
            simp = r_trace.get("simplified", {}) or {}
            if simp:
                print("  canonical:", simp.get("canonical"))
                # compact signals summary
                sig = (r_trace.get("signals") or {})
                if sig:
                    on = [k for k, v in sig.items() if v]
                    if on:
                        print("  signals:", ", ".join(on))
        except Exception:
            pass

        # Orchestrator reasoning trace (if present)
        if isinstance(tr, dict):
            try:
                if tr.get("route"):
                    print("ORCHESTRATOR route:", tr.get("route"))
                steps = tr.get("steps") or tr.get("plan") or []
                if isinstance(steps, list) and steps:
                    print("  steps:", len(steps))
                    # Print a compact preview of the first 3 steps
                    for i, s in enumerate(steps[:3], start=1):
                        if isinstance(s, dict):
                            kind = s.get("type") or s.get("name") or "step"
                            summary = s.get("summary") or s.get("action") or s.get("note") or ""
                            print(f"    {i}. {kind}: {str(summary)[:120]}")
            except Exception:
                pass

        # List all selected contexts (not only the first) with metadata
        if ctx:
            print("CONTEXTS:")
            for i, d in enumerate(ctx, start=1):
                md = d.metadata or {}
                print(f"  {i}. {md.get('file_name')} p{md.get('page')} {md.get('section')} #{md.get('anchor')}")
            # Show a short content head for the top context
            d0 = ctx[0]
            print("--- CTX[1] content head ---\n", (d0.page_content or "")[:800])


if __name__ == "__main__":
    main()
