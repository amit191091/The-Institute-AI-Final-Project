"""
Tab-specific UI builders for the Gradio interface.
"""
import gradio as gr
import re
from pathlib import Path
from RAG.app.Gradio_apps.ui_components import (
    _rows_to_df, _fmt_docs, sort_figure_docs, _rows_for_df
)
from RAG.app.Gradio_apps.ui_qa_handlers import _run_agent
from RAG.app.Gradio_apps.ui_graph_handlers import _gen_graph, _run_cypher_ui
from RAG.app.Gradio_apps.ui_handlers import _on_refresh
from RAG.app.Gradio_apps.ui_evaluation import (
    on_test_google_api, on_test_ragas, on_generate_ground_truth, on_evaluate_rag
)

def build_ask_tab(docs, hybrid, llm, debug, gt_map, qa_map, on_ask_handler):
    """Build the Ask tab."""
    with gr.Tab("Ask"):
        with gr.Row():
            with gr.Column(scale=3):
                q = gr.Textbox(label="Question", placeholder="Ask about figures, tables, procedures, conclusions...", lines=3)
                ground_truth = gr.Textbox(label="Ground Truth (Optional)", placeholder="Expected answer for evaluation...", lines=2)
                dbg = gr.Checkbox(label="Show retrieval debug", value=False)
                
                with gr.Row():
                    btn = gr.Button("🔍 Ask", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear")
                
                with gr.Row():
                    shutdown_btn = gr.Button("🛑 Shutdown Server", variant="stop", size="lg")
                    exit_btn = gr.Button("❌ Exit Application", variant="stop", size="lg")
                
                ans = gr.Markdown()
                metrics = gr.Textbox(label="Evaluation Metrics", lines=8, interactive=False)
                # Inline figure preview for the current answer
                fig_preview = gr.Image(label="Relevant figure", interactive=False, visible=True)
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 System Info")
                gr.Markdown(f"**Total Documents:** {len(docs)}")
                section_values = sorted({(d.metadata or {}).get("section") or "" for d in docs})
                section_values = [s for s in section_values if s]
                gr.Markdown(f"**Sections:** {', '.join(section_values)}")
                
                if debug:
                    gr.Markdown("### 🔧 Debug Tools")
                    gr.Markdown("Debug information available")
        
        with gr.Accordion("Debug (retrieval trace)", open=False, visible=False) as dbg_acc:
            dbg_router = gr.Markdown()
            dbg_filters = gr.JSON()
            dbg_dense = gr.Markdown()
            dbg_sparse = gr.Markdown()
            dbg_hybrid = gr.Markdown()
            dbg_topdocs = gr.Dataframe(interactive=False)
            dbg_compare = gr.JSON(label="Answer vs Reference (tokens)")
        
        btn.click(
            lambda q, ground_truth, dbg: on_ask_handler(q, ground_truth, dbg, docs, hybrid, llm, debug, gt_map, qa_map),
            inputs=[q, ground_truth, dbg],
            outputs=[ans, metrics, dbg_acc, dbg_router, dbg_filters, dbg_dense, dbg_sparse, dbg_hybrid, dbg_topdocs, dbg_compare, fig_preview],
        )
        
        clear_btn.click(
            lambda: ("", "", False, "", "", gr.update(visible=False, open=False), "", "", "", "", "", "", None),
            outputs=[q, ground_truth, dbg, ans, metrics, dbg_acc, dbg_router, dbg_filters, dbg_dense, dbg_sparse, dbg_hybrid, dbg_topdocs, dbg_compare, fig_preview]
        )
        
        shutdown_btn.click(
            lambda: "🔄 Server shutdown initiated...",
            outputs=[ans]
        )
        
        def exit_application():
            """Exit the application by raising a SystemExit exception."""
            import sys
            print("🔄 Exiting application...")
            sys.exit(0)
        
        exit_btn.click(
            exit_application,
            outputs=[]
        )
        
        # Keyboard shortcuts
        q.submit(
            lambda q, ground_truth, dbg: on_ask_handler(q, ground_truth, dbg, docs, hybrid, llm, debug, gt_map, qa_map),
            inputs=[q, ground_truth, dbg],
            outputs=[ans, metrics, dbg_acc, dbg_router, dbg_filters, dbg_dense, dbg_sparse, dbg_hybrid, dbg_topdocs, dbg_compare, fig_preview]
        )

def build_figures_tab(docs):
    """Build the Figures tab."""
    with gr.Tab("Figures"):
        # Build a gallery of extracted figures, sorted by figure_number/figure_order
        fig_docs = [d for d in docs if d.metadata.get("section") == "Figure" and d.metadata.get("image_path")]
        fig_docs = sort_figure_docs(fig_docs)
        fig_paths = [str(Path(d.metadata.get("image_path"))) for d in fig_docs if d.metadata.get("image_path")]
        if fig_paths:
            # Recent Gradio versions preview by default; keep args minimal for compatibility
            gr.Gallery(value=fig_paths, label="Extracted Figures", columns=4, height=400)
        else:
            gr.Markdown("(No extracted figures. Enable RAG_EXTRACT_IMAGES=true and rerun.)")

def build_agent_tab(docs, hybrid, llm):
    """Build the Agent tab."""
    with gr.Tab("Agent"):
        gr.Markdown("### Agent trace (tools + observations)\nRuns retrieval via simple tools for visibility.")
        
        # Question selection section
        gr.Markdown("#### 📋 Question Selection")
        load_questions_btn = gr.Button("Load Questions from File")
        questions_display = gr.Markdown("Click 'Load Questions' to see available questions")
        question_number = gr.Number(label="Enter question number", minimum=1, step=1, placeholder="e.g., 5")
        load_selected_btn = gr.Button("Load Selected Question")
        
        # Original question input
        q2 = gr.Textbox(label="Question", placeholder="E.g., list all figures or show figure 3")
        run_btn = gr.Button("Run Agent")
        trace_json = gr.JSON(label="Trace")
        result_md = gr.Markdown()

        # Maintenance tools
        try:
            from RAG.app.Agent_Components.agent_tools import tool_audit_and_fill_figures as _audit_figs, tool_plan as _plan
        except Exception:
            _audit_figs = None
            _plan = None

        from RAG.app.Gradio_apps.ui_qa_handlers import _do_audit, _do_plan

        # Question loading functions
        def load_questions_from_file():
            """Load questions from the QA file and display them as a numbered list."""
            try:
                import json
                from pathlib import Path
                
                # Try to find the QA file
                qa_file_paths = [
                    Path("RAG/data/gear_wear_qa.jsonl"),
                    Path("data/gear_wear_qa.jsonl"),
                    Path("gear_wear_qa.jsonl"),
                    Path("gear_wear_qa.json")
                ]
                
                qa_file = None
                for path in qa_file_paths:
                    if path.exists():
                        qa_file = path
                        break
                
                if not qa_file:
                    return "❌ No QA file found. Please ensure 'gear_wear_qa.jsonl' exists in the data directory."
                
                questions = []
                with open(qa_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            question_id = data.get('id', f'Q{line_num}')
                            question_text = data.get('question', 'No question text')
                            questions.append(f"{line_num}. **{question_id}**: {question_text}")
                        except json.JSONDecodeError:
                            continue
                
                if not questions:
                    return "❌ No valid questions found in the file."
                
                questions_text = "\n".join(questions)
                return f"✅ **Loaded {len(questions)} questions:**\n\n{questions_text}"
                
            except Exception as e:
                return f"❌ Error loading questions: {str(e)}"
        
        def load_selected_question(question_num):
            """Load a specific question by number into the question textbox."""
            try:
                import json
                from pathlib import Path
                
                # Try to find the QA file
                qa_file_paths = [
                    Path("RAG/data/gear_wear_qa.jsonl"),
                    Path("data/gear_wear_qa.jsonl"),
                    Path("gear_wear_qa.jsonl"),
                    Path("gear_wear_qa.json")
                ]
                
                qa_file = None
                for path in qa_file_paths:
                    if path.exists():
                        qa_file = path
                        break
                
                if not qa_file:
                    return "❌ No QA file found."
                
                if not question_num or question_num < 1:
                    return "❌ Please enter a valid question number (1 or higher)."
                
                with open(qa_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if line_num == int(question_num):
                            line = line.strip()
                            if not line:
                                return "❌ Question number not found."
                            try:
                                data = json.loads(line)
                                question_text = data.get('question', 'No question text')
                                return question_text
                            except json.JSONDecodeError:
                                return "❌ Invalid question format."
                
                return "❌ Question number not found."
                
            except Exception as e:
                return f"❌ Error loading question: {str(e)}"

        # Connect the buttons
        load_questions_btn.click(
            load_questions_from_file,
            inputs=[],
            outputs=[questions_display]
        )
        
        load_selected_btn.click(
            load_selected_question,
            inputs=[question_number],
            outputs=[q2]
        )

        run_btn.click(
            lambda q: _run_agent(q, docs, hybrid, llm), 
            inputs=[q2], 
            outputs=[trace_json, result_md]
        )

        gr.Markdown("---")
        gr.Markdown("#### Maintenance: audit and fill missing figure numbers/orders")
        audit_btn = gr.Button("Audit/Fix Figures (session-only)")
        audit_out = gr.JSON(label="Audit Summary")
        audit_btn.click(lambda: _do_audit(docs), inputs=[], outputs=[audit_out])

        gr.Markdown("#### Planner: propose a plan to fix DB issues")
        obs = gr.Textbox(label="Observations (paste from db_snapshot.jsonl)")
        plan_btn = gr.Button("Generate Plan")
        plan_md = gr.Markdown()
        plan_btn.click(lambda obs: _do_plan(obs, llm), inputs=[obs], outputs=[plan_md])

def build_inspect_tab(docs):
    """Build the Inspect tab."""
    with gr.Tab("Inspect"):
        gr.Markdown("### Top indexed docs (sample)")
        sample_docs = [d for d in docs[:12]]
        gr.Textbox(value=_fmt_docs(sample_docs, max_items=12), label="Sample Contexts", lines=15)

def create_graph_ui_tab(docs):
    """Build the Graph tab."""
    with gr.Tab("Graph"):
        gr.Markdown("### Knowledge Graph (auto-built)")
        # If the Main built a graph, it will be at logs/graph.html
        _graph_view = gr.HTML(value="")
        _graph_status = gr.Markdown()
        src = gr.Dropdown(
            choices=[
                "Docs co-mention (default)",
                "Normalized graph.json",
                "Normalized chunks",
                "Neo4j (live)"
            ],
            value="Docs co-mention (default)",
            label="Graph source"
        )
        btn_graph = gr.Button("Generate / Refresh Graph")
        gr.Markdown("#### Graph DB (Neo4j) Optional")
        # Prefill with a sample so clicks don't pass an empty string on some Gradio builds
        cypher = gr.Textbox(label="Cypher query", value="MATCH (n) RETURN n LIMIT 10", placeholder="Type a Cypher queryΓאª", lines=2)
        btn_cypher = gr.Button("Run Cypher")
        cypher_out = gr.JSON(label="Results")

        # Initial load if file exists
        try:
            from RAG.app.config import settings
            graph_html_path = settings.paths.LOGS_DIR/"graph.html"
            if graph_html_path.exists():
                html_data = graph_html_path.read_text(encoding="utf-8")
                import html as _html
                iframe = f"<p><a href='file:///{graph_html_path.resolve().as_posix()}' target='_blank'>Open graph.html in browser</a></p>" \
                         f"<iframe style='width:100%;height:650px;border:1px solid #ddd' srcdoc=\"{_html.escape(html_data)}\"></iframe>"
                _graph_view.value = iframe
            else:
                _graph_status.value = "(Graph not available yet Γאף click the button to generate it.)"
        except Exception:
            _graph_status.value = "(Graph not available yet Γאף click the button to generate it.)"
        
        btn_graph.click(
            lambda s: _gen_graph(s, docs), 
            inputs=[src], 
            outputs=[_graph_view, _graph_status]
        )

        # Support both click and Enter-to-submit
        btn_cypher.click(_run_cypher_ui, inputs=[cypher], outputs=[cypher_out])
        cypher.submit(_run_cypher_ui, inputs=[cypher], outputs=[cypher_out])

def build_evaluation_tab(docs, hybrid, llm):
    """Build the Evaluation tab."""
    with gr.Tab("📋 Evaluation"):
        gr.Markdown("### RAGAS Evaluation Tools")
        with gr.Row():
            test_google_btn = gr.Button("Test Google API")
            test_ragas_btn = gr.Button("Test RAGAS")
        with gr.Row():
            generate_gt_btn = gr.Button("Generate Ground Truth")
            num_questions = gr.Slider(minimum=5, maximum=50, value=10, step=5, label="Number of Questions")
        evaluate_btn = gr.Button("Evaluate RAG System", variant="primary")
        eval_output = gr.Textbox(label="Evaluation Results", lines=15, interactive=False)
        
        # Evaluation handlers
        test_google_btn.click(on_test_google_api, outputs=[eval_output])
        test_ragas_btn.click(on_test_ragas, outputs=[eval_output])
        generate_gt_btn.click(
            lambda n: on_generate_ground_truth(docs, hybrid, llm, n), 
            inputs=[num_questions], 
            outputs=[eval_output]
        )
        evaluate_btn.click(
            lambda: on_evaluate_rag(docs, hybrid, llm), 
            outputs=[eval_output]
        )

def build_db_explorer_tab(docs):
    """Build the DB Explorer tab."""
    with gr.Tab("DB Explorer"):
        gr.Markdown("### Browse indexed documents (filters below)")
        # Add an '(All)' option to avoid None handling differences across versions
        section_values = sorted({(d.metadata or {}).get("section") or "" for d in docs})
        section_values = [s for s in section_values if s]
        sec_choices = ["(All)"] + section_values
        sec_dd = gr.Dropdown(choices=sec_choices, label="Section filter", value="(All)")
        qbox = gr.Textbox(label="Contains (text or metadata)")
        refresh = gr.Button("Refresh")
        # Initialize the table with data at construction time for v5 compatibility
        _initial_rows = _rows_for_df(docs, None, None)
        df = gr.Dataframe(
            value=_rows_to_df(_initial_rows),
            wrap=True,
            interactive=False,
        )
        refresh.click(lambda fs, qq: _on_refresh(docs, fs, qq), inputs=[sec_dd, qbox], outputs=[df])
        # initial load handled via value above
