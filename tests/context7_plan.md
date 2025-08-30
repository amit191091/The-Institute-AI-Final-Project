Context7 plan for Graph feature validation

Goals
- Verify graph rendering path works even without PyVis (fallback HTML)
- Verify nodes and edges count is non-zero after indexing a sample doc

Manual steps
- Run the app and click Graph → Generate / Refresh Graph
- Inspect logs/graph.html for vis-network script and container

Automated hooks (to implement)
- Parse logs/graph.html; check presence of `vis-network` and a non-empty nodes array
- Optionally, expose a lightweight `/graph.json` API returning nodes/edges for easier assertions

Ideas
- Extract entity list from a small known document and assert presence of terms (e.g., W26, RMS)
- Record a before/after nodes count when adding another file to the dataset
