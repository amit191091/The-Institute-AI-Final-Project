import gradio as gr
from pathlib import Path

from RAG.app.logger import get_logger

# Import functions from modular Gradio apps
from RAG.app.Gradio_apps.ui_qa_handlers import on_ask


def build_ui(docs, hybrid, llm, debug=None) -> gr.Blocks:
	# Load ground truth and QA data first
	from RAG.app.Gradio_apps.ui_data_loader import load_ground_truth_and_qa
	gt_map, qa_map = load_ground_truth_and_qa()
	
	log = get_logger()
	# Precompute unique sections for filters
	section_values = sorted({(d.metadata or {}).get("section") or "" for d in docs})
	section_values = [s for s in section_values if s]


	# Build a sleeker Blocks UI with tabs
	from RAG.app.Gradio_apps.ui_tabs import (
		build_ask_tab, build_figures_tab, build_agent_tab, 
		build_inspect_tab, create_graph_ui_tab, build_evaluation_tab, build_db_explorer_tab
	)
	
	with gr.Blocks(title="RAG System", theme=gr.themes.Soft()) as demo:
		# Header with title and exit button
		with gr.Row():
			gr.Markdown("# 🤖 RAG System - Gear Failure Analysis")
			exit_btn = gr.Button("❌ Exit", variant="stop", size="sm")
		
		gr.Markdown("Ask questions about gear failure analysis, transmission ratios, wear measurements, and more.")
		
		# Exit functionality
		def exit_app():
			print("🔄 Exiting RAG System...")
			try:
				# Use Gradio's built-in shutdown mechanism
				import gradio as gr
				gr.close_all()
				# Also try to stop the server gracefully
				import os
				import signal
				os.kill(os.getpid(), signal.SIGTERM)
			except Exception as e:
				print(f"Exit error: {e}")
				# Fallback to sys.exit
				import sys
				sys.exit(0)
		
		exit_btn.click(exit_app, outputs=[])
		
		with gr.Tabs():
			build_ask_tab(docs, hybrid, llm, debug, gt_map, qa_map, on_ask)
			build_figures_tab(docs)
			build_agent_tab(docs, hybrid, llm)
			build_inspect_tab(docs)
			create_graph_ui_tab(docs)
			build_evaluation_tab(docs, hybrid, llm)
			build_db_explorer_tab(docs)

	return demo

