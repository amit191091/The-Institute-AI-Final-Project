# UI Smoke (Playwright manual checklist)

1. Start the app:
   - Ensure `.env` has `RAG_HEADLESS=false` and optional `GRADIO_SHARE=true`.
   - Ensure figures/tables toggles as desired (see below).
2. Open http://127.0.0.1:7860 in the browser.
3. Ask: "show me figure 1" — expect route `table` and Figure contexts; gallery in Figures tab should show items.
4. Ask: "what's inside table 1" — expect route `table`, Table contexts in preview. If not present, enable more extractors.

Recommended toggles on Windows without Poppler:
- RAG_PDF_HI_RES=false
- RAG_EXTRACT_IMAGES=true
- RAG_USE_PDFPLUMBER=true
- RAG_SYNTH_TABLES=true

Optional:
- RAG_USE_TABULA=true (requires Java; subprocess mode OK)
- RAG_USE_CAMELOT=true (requires Ghostscript/Poppler)
