#!/usr/bin/env python3
"""Force a complete clean run - deletes ChromaDB and elements before running."""

import os
import sys
import shutil
from pathlib import Path

print("🧹 FORCING COMPLETE CLEAN RUN")
print("=" * 50)

# Set environment variables to force cleaning
os.environ["RAG_CLEAN_RUN"] = "1"          # Clean data/images, data/elements, logs
os.environ["RAG_CLEAN_CHROMA"] = "1"       # Clean ChromaDB persistence directory

print("Environment flags set:")
print(f"  RAG_CLEAN_RUN = {os.environ.get('RAG_CLEAN_RUN')}")
print(f"  RAG_CLEAN_CHROMA = {os.environ.get('RAG_CLEAN_CHROMA')}")

# Additional manual cleanup to ensure everything is deleted
directories_to_clean = [
    "data/images",
    "data/elements", 
    "logs/elements",
    "index/chroma",
    "index/chroma_llamaparse"
]

files_to_clean = [
    "logs/queries.jsonl",
    "logs/graph.html",
    "logs/db_snapshot.jsonl",
    "logs/db_snapshot_full.jsonl"
]

print("\nManually cleaning directories:")
for dir_path in directories_to_clean:
    path = Path(dir_path)
    if path.exists():
        try:
            shutil.rmtree(path)
            print(f"  ✅ Deleted: {dir_path}")
        except Exception as e:
            print(f"  ❌ Failed to delete {dir_path}: {e}")
    else:
        print(f"  ℹ️  Not found: {dir_path}")

print("\nManually cleaning files:")
for file_path in files_to_clean:
    path = Path(file_path)
    if path.exists():
        try:
            path.unlink()
            print(f"  ✅ Deleted: {file_path}")
        except Exception as e:
            print(f"  ❌ Failed to delete {file_path}: {e}")
    else:
        print(f"  ℹ️  Not found: {file_path}")

print("\n🚀 Clean run completed! Now running the pipeline...")
print("=" * 50)

# Now run the main pipeline
sys.path.append('.')
try:
    from app.pipeline import run
    run()
except Exception as e:
    print(f"❌ Error running pipeline: {e}")
    import traceback
    traceback.print_exc()
