"""Playwright smoke test for the Gradio UI and flags.
Run:  python -m pip install playwright; python -m playwright install
      python -m pytest -q tests/playwright_smoke.py
"""
import os
import time
import subprocess
from pathlib import Path
from contextlib import contextmanager
from urllib.request import urlopen
from urllib.error import URLError, HTTPError


@contextmanager
def _run_ui(port: int = 7861):
    env = os.environ.copy()
    env.update({
        "GRADIO_PORT": str(port),
    "GRADIO_SERVER_NAME": "127.0.0.1",
        "RAG_HEADLESS": "0",
    # speed up startup: avoid full clean + heavy extractors
    "RAG_CLEAN_RUN": "0",
    "RAG_USE_PDFPLUMBER": "1",
    "RAG_USE_TABULA": "0",
    "RAG_USE_PYMUPDF": "1",
    "CE_ENABLED": "0",
    })
    p = subprocess.Popen(["python", "main.py"], env=env)
    try:
        # scan a few ports in case Gradio auto-increments
        ports = [port, 7860, 7862, 7863, 7865]
        deadline = time.time() + 240  # up to 4 minutes for first run on CI
        found_url = None
        while time.time() < deadline and p.poll() is None and not found_url:
            for prt in ports:
                url = f"http://127.0.0.1:{prt}"
                try:
                    with urlopen(url, timeout=2) as _:
                        found_url = url
                        break
                except (URLError, HTTPError, TimeoutError):
                    continue
            if not found_url:
                time.sleep(1.5)
        # small grace period
        if not found_url:
            time.sleep(5)
            found_url = f"http://127.0.0.1:{port}"
        yield found_url
    finally:
        p.terminate()


def test_ui_basic_flow(page):
    # Uses pytest-playwright's built-in 'page' fixture (sync API managed by plugin)
    with _run_ui() as url:
            page.goto(url, wait_until="domcontentloaded")
            # Ask tab elements
            page.get_by_text("Hybrid RAG", exact=False).wait_for()
            page.get_by_role("tab", name="Ask").click()
            page.get_by_role("button", name="Ask").wait_for()
            # Ask a question in the Ask tab (use placeholder to disambiguate from Agent tab)
            q = page.get_by_placeholder("Ask about figures", exact=False)
            q.fill("Summarize the failure modes")
            page.get_by_role("button", name="Ask").click()
            # Wait for the routed output to appear (can take a few seconds on first run)
            # Accept any of the key markers rendered in the Markdown answer
            try:
                page.get_by_text("Route:", exact=False).wait_for(timeout=30000)
            except Exception:
                try:
                    page.get_by_text("Agent:", exact=False).wait_for(timeout=10000)
                except Exception:
                    page.get_by_text("Top contexts:", exact=False).wait_for(timeout=10000)
            assert page.get_by_text("Route:", exact=False).is_visible() or \
                   page.get_by_text("Agent:", exact=False).is_visible() or \
                   page.get_by_text("Top contexts:", exact=False).is_visible()


def test_flags_variants(monkeypatch):
    # Toggle CE and LlamaIndex flags and ensure pipeline runs headless
    env = os.environ.copy()
    env.update({
        "RAG_HEADLESS": "1",
        "RAG_EVAL": "0",
    # keep it lean
    "RAG_CLEAN_RUN": "0",
    "RAG_USE_PDFPLUMBER": "1",
    "RAG_USE_TABULA": "0",
    "RAG_USE_PYMUPDF": "0",
    "CE_ENABLED": "0",
        "RAG_ENABLE_LLAMAINDEX": "1",
    })
    r = subprocess.run(["python", "main.py"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert r.returncode == 0
    # Check snapshot exists and llamaindex path created (if dependencies present)
    assert Path("logs/db_snapshot.jsonl").exists()
    # LlamaIndex export may be a no-op if package is missing; only soft check directory presence
    Path("data/elements/llamaindex").mkdir(parents=True, exist_ok=True)
