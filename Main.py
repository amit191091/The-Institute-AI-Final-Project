from __future__ import annotations

"""Thin entrypoint delegating to app.pipeline for cleaner structure."""

from app.pipeline import run
def main() -> None:
    run()


if __name__ == "__main__":
    main()
# from dotenv import load_dotenv
# import os

# load_dotenv()  # Load .env file


# # def Full_pipeline():
# #     print("starting full pipeline")
# #     # Placeholder for the full pipeline logic
# #     #1.file extraction + Parsing+ chunking avg chunk size :250-500, 800 tokens for table\diagram
# #     #2.metadata generation - filename, pagenumber, chunk_summary, keywords, section_type clientID\CaseID etc..
# #     #3.indexing - tables to csv\markdown , tableid, pagenum, anchor saving + small text summarization of table, vector metadate to filter retrival etc
# #     #4.Hybrid retrieval
# #     #5.Multi document support
# #     #6.gradio QA agent
# #     print("pipeline ended")




# def main():
#     print("hello world bitches")
#     return

# if __name__ == "__main__":
#     main()