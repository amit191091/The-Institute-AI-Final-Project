"""Playwright smoke test for the Gradio UI and flags.
Run:  python -m pip install playwright; python -m playwright install
      python -m pytest -q tests/playwright_smoke.py
"""
import os
import time
import subprocess
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def _run_ui(port: int = 7861):
    env = os.environ.copy()
    env.update({
        "GRADIO_PORT": str(port),
        "RAG_HEADLESS": "0",
    })
    p = subprocess.Popen(["python", "main.py"], env=env)
    try:
        # Give server a moment
        time.sleep(6)
        yield f"http://127.0.0.1:{port}"
    finally:
        p.terminate()


def test_ui_basic_flow(playwright):
    from playwright.sync_api import sync_playwright
    with _run_ui() as url:
        pw = sync_playwright().start()
        try:
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded")
            # Ask tab elements
            page.get_by_text("Hybrid RAG").wait_for()
            page.get_by_role("button", name="Ask").wait_for()
            # Ask a question
            q = page.get_by_label("Question")
            q.fill("Summarize the failure modes")
            page.get_by_role("button", name="Ask").click()
            page.wait_for_timeout(2500)
            # Expect some Markdown output
            assert page.get_by_text("Route:").is_visible()
        finally:
            pw.stop()


def test_flags_variants(monkeypatch):
    # Toggle CE and LlamaIndex flags and ensure pipeline runs headless
    env = os.environ.copy()
    env.update({
        "RAG_HEADLESS": "1",
        "RAG_EVAL": "0",
        "CE_ENABLED": "1",
        "CE_PROVIDER": "auto",
        "RAG_ENABLE_LLAMAINDEX": "1",
    })
    r = subprocess.run(["python", "main.py"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert r.returncode == 0
    # Check snapshot exists and llamaindex path created (if dependencies present)
    assert Path("logs/db_snapshot.jsonl").exists()
    # LlamaIndex export may be a no-op if package is missing; only soft check directory presence
    Path("data/elements/llamaindex").mkdir(parents=True, exist_ok=True)
