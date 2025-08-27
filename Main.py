from __future__ import annotations

"""Thin entrypoint delegating to RAG.app.pipeline for cleaner structure."""

from RAG.app.pipeline import run

def main() -> None:
    run()

if __name__ == "__main__":
    main()
