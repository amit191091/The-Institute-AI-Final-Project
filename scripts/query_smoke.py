from pathlib import Path
import os
import sys

# Ensure project root is on sys.path when running from scripts/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline import build_pipeline, _LLM, ask, answer_with_contexts


def main():
    # Prefer headless behavior
    os.environ.setdefault("RAG_HEADLESS", "1")
    os.environ["RAG_USE_NORMALIZED"] = "0"
    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
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

    questions = [
        "whats the transmission ratio",
        "what is the gear ratio",
        "show me table 3",
    ]
    for q in questions:
        print("\nQ:", q)
        try:
            ans, ctx = answer_with_contexts(docs, hybrid, llm, q)
        except Exception as e:
            ans = f"[error] {e}"
            ctx = []
        print("A:", str(ans)[:400])
        if ctx:
            d0 = ctx[0]
            md = d0.metadata or {}
            print("CTX0:", md.get("file_name"), "p", md.get("page"), md.get("section"), "#", md.get("anchor"))
            print("--- content head ---\n", (d0.page_content or "")[:800])


if __name__ == "__main__":
    main()
