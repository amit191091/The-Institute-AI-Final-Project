from typing import List
from app.retrieve import apply_filters, rerank_candidates, lexical_overlap
from app.agents import answer_table, identify_csv_targets, _matches_csv_target
from app.config import settings


def _ask_table_hierarchical(docs, hybrid, llm, question: str, qa: dict) -> str:
    """
    Hierarchical retrieval strategy for table questions:
    1. First search ONLY in the main report (PDF)
    2. If no answer found, search in CSV files based on question type
    3. Fallback to standard retrieval if needed
    """
    
    def _doc_head(d):
        md = getattr(d, "metadata", {}) or {}
        return f"{md.get('file_name')} p{md.get('page')} {md.get('section')}#{md.get('anchor', '')}"

    def _score(d):
        base = lexical_overlap(question, d.page_content)
        meta_text = " ".join(map(str, (getattr(d, "metadata", {}) or {}).values()))
        boost = 0.2 * lexical_overlap(" ".join(qa["keywords"]), meta_text)
        return round(base + boost, 4)
    
    # Step 1: Try to find answer in main report (PDF) ONLY
    pdf_docs = [d for d in docs if "Gear wear Failure.pdf" in d.metadata.get("file_name", "")]
    
    if pdf_docs:
        # Use ONLY PDF documents for search - IGNORE section filter for PDF
        pdf_filtered = pdf_docs  # Use all PDF docs, ignore section filter
        
        pdf_top_docs = rerank_candidates(question, pdf_filtered, top_n=settings.CONTEXT_TOP_N)
        
        # Try to answer with PDF documents using simple table prompt
        from app.agents import TABLE_SYSTEM, TABLE_PROMPT, render_context
        ctx = render_context(pdf_top_docs)
        prompt = TABLE_SYSTEM + "\n" + TABLE_PROMPT.format(table=ctx, question=question)
        ans = llm(prompt).strip()
        
        # Check if we got a meaningful answer
        if not any(phrase in ans.lower() for phrase in ["not found", "not provided", "cannot find", "not available", "no information", "not mentioned", "does not contain", "cannot provide"]):
            return ans
    
    # Step 2: If no answer in PDF, search in CSV files ONLY
    csv_targets = identify_csv_targets(question)
    
    if csv_targets:
        # Get only CSV documents
        csv_docs = [d for d in docs if _matches_csv_target(d.metadata.get("file_name"), csv_targets)]
        
        if csv_docs:
            csv_filtered = apply_filters(csv_docs, qa["filters"])
            if not csv_filtered:
                csv_filtered = csv_docs  # Use all CSV docs if filters don't match
            
            csv_top_docs = rerank_candidates(question, csv_filtered, top_n=settings.CONTEXT_TOP_N)
            
            # Try to answer with CSV documents using simple table prompt
            from app.agents import TABLE_SYSTEM, TABLE_PROMPT, render_context
            ctx = render_context(csv_top_docs)
            prompt = TABLE_SYSTEM + "\n" + TABLE_PROMPT.format(table=ctx, question=question)
            ans = llm(prompt).strip()
            
            # Check if we got a meaningful answer
            if not any(phrase in ans.lower() for phrase in ["not found", "not provided", "cannot find", "not available", "no information", "not mentioned", "does not contain", "cannot provide"]):
                return ans
    
    # Step 3: Fallback to standard retrieval
    try:
        candidates = hybrid.get_relevant_documents(question)
    except Exception:
        candidates = hybrid.invoke(question)
    
    candidates = candidates[:settings.RERANK_TOP_K]
    filtered = apply_filters(candidates, qa["filters"])
    
    try:
        sec = qa["filters"].get("section")
    except Exception:
        sec = None
    if sec and not filtered:
        filtered = [d for d in docs if (d.metadata or {}).get("section") == sec]
    
    top_docs = rerank_candidates(question, filtered, top_n=settings.CONTEXT_TOP_N)
    
    return answer_table(llm, top_docs, question, hybrid)
