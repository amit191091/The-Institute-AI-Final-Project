"""
End-to-end Graph tab test using Playwright (Python).

Prereqs (Windows PowerShell):
  pip install pytest playwright pytest-playwright
  python -m playwright install chromium

Run:
  $env:GRADIO_SERVER_NAME='127.0.0.1'; $env:GRADIO_PORT='7860'; pytest -q tests/e2e/test_graph_playwright.py
"""
import os
import time
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright, expect


def ensure_server():
    # If the server is already up, reuse it; otherwise start it.
    import socket
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.environ.get("GRADIO_PORT", "7860"))
    s = socket.socket(); s.settimeout(0.2)
    try:
        s.connect((host, port))
        s.close()
        return f"http://{host}:{port}"
    except Exception:
        s.close()
    # Start the app
    env = os.environ.copy()
    env.setdefault("GRADIO_SERVER_NAME", "127.0.0.1")
    env.setdefault("GRADIO_PORT", "7860")
    env.setdefault("RAG_HEADLESS", "false")
    proc = subprocess.Popen([os.path.join(".venv", "Scripts", "python.exe"), "main.py"], env=env)
    # Give it a few seconds to spin up
    for _ in range(60):
        time.sleep(0.5)
        try:
            s = socket.socket(); s.settimeout(0.2); s.connect((env["GRADIO_SERVER_NAME"], int(env["GRADIO_PORT"]))); s.close()
            return f"http://{env['GRADIO_SERVER_NAME']}:{env['GRADIO_PORT']}"
        except Exception:
            pass
    proc.terminate()
    raise RuntimeError("Server failed to start")


def test_graph_tab_generates_html(tmp_path):
    base = ensure_server()
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(base, wait_until="domcontentloaded")
        page.get_by_role("tab", name="Graph").click()
        # Click the button to generate the graph
        page.get_by_role("button", name="Generate / Refresh Graph").click()
        # Wait briefly for the server function
        page.wait_for_timeout(2000)
        # Validate that the fallback message is not present, or graph.html exists
        status = page.locator("text=(failed to build graph)")
        # Allow some retries
        ok = False
        for _ in range(10):
            if status.count() == 0:
                ok = True
                break
            page.wait_for_timeout(500)
        # As an additional assertion, check that logs/graph.html exists and is non-empty
        gh = Path("logs")/"graph.html"
        assert gh.exists(), "graph.html not created"
        assert gh.stat().st_size > 0, "graph.html is empty"
        assert ok or gh.read_text(encoding="utf-8").strip() != "", "Graph generation failed"
        browser.close()
