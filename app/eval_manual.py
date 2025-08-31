"""
Manual Factual Evaluation System
Implements user-defined thresholds for precise control over evaluation criteria.

Target Metrics (user-specified):
- Answer Correctness: Ground Truth alignment
- Context Precision ≥ 0.75
- Context Recall ≥ 0.70  
- Faithfulness ≥ 0.85
- Table-QA Accuracy ≥ 0.90
"""

import os
import re
import json
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from app.logger import trace_func

# LLM imports for judge-based evaluation
try:
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    ChatOpenAI = None
    ChatGoogleGenerativeAI = None
    SystemMessage = HumanMessage = None

@dataclass
class EvaluationThresholds:
    """User-defined thresholds for manual evaluation"""
    context_precision: float = 0.75
    context_recall: float = 0.70
    faithfulness: float = 0.85
    table_qa_accuracy: float = 0.90
    answer_correctness: float = 0.80  # Ground truth alignment

@dataclass
class QuestionMetrics:
    """Per-question evaluation metrics"""
    question: str
    answer: str
    reference: str
    contexts: List[str]
    
    # Core metrics
    answer_correctness: float
    context_precision: float
    context_recall: float
    faithfulness: float
    table_qa_accuracy: Optional[float] = None
    
    # Derived metrics
    passes_thresholds: bool = False
    is_table_question: bool = False
    reasoning: str = ""

@trace_func
def _setup_llm_judge():
    """Setup LLM for manual evaluation judging"""
    # Priority: Use OpenAI if available and allowed
    if os.getenv("OPENAI_API_KEY") and ChatOpenAI:
        try:
            return ChatOpenAI(
                model=os.getenv("OPENAI_EVAL_MODEL", "gpt-4o-mini"),
                temperature=0.1,
                max_retries=2
            )
        except Exception as e:
            print(f"OpenAI setup failed: {e}")
    
    # Fallback: Google if available
    if os.getenv("GOOGLE_API_KEY") and ChatGoogleGenerativeAI:
        try:
            return ChatGoogleGenerativeAI(
                model=os.getenv("GOOGLE_EVAL_MODEL", "gemini-1.5-pro"),
                temperature=0.1,
                max_retries=2
            )
        except Exception as e:
            print(f"Google setup failed: {e}")
    
    raise RuntimeError("No LLM available for manual evaluation. Please set OPENAI_API_KEY or GOOGLE_API_KEY")

@trace_func
def _is_table_question(question: str) -> bool:
    """Detect if question requires table/numerical data analysis"""
    q_lower = question.lower()
    table_patterns = [
        r"wear\s*depth",
        r"\bwhich\s+wear\s+case\b",
        r"\btransmission\s+ratio\b|\bgear\s+ratio\b",
        r"\bmodule\b",
        r"sampling\s*rate|khz|hz",
        r"\bsensitivity\b",
        r"\btachometer\b|\baccelerometer\b",
        r"\btable\b.*\d|\d.*\btable\b",
        r"case\s*\d+|w\d+",
        r"what.*value|how.*much|specific.*number"
    ]
    
    return any(re.search(pattern, q_lower) for pattern in table_patterns)

@trace_func
def _calculate_answer_correctness(question: str, answer: str, reference: str, llm) -> Tuple[float, str]:
    """Calculate how well the answer matches the ground truth reference"""
    
    system_prompt = """You are an expert evaluator for technical Q&A systems.

Your task is to score how well the ANSWER matches the REFERENCE (ground truth) for the given QUESTION.

Scoring Criteria:
- 1.0: Perfect match - answer is factually identical to reference
- 0.9: Excellent - answer contains reference info with minor additional context
- 0.8: Good - answer contains reference info but with some extra/missing details
- 0.7: Adequate - core reference information present but presentation differs
- 0.6: Partial - some reference information present but incomplete
- 0.5: Minimal - answer touches on reference but misses key details
- 0.0-0.4: Poor - answer contradicts or completely misses reference

Focus on FACTUAL ACCURACY, not presentation style.

Respond with:
SCORE: [0.0-1.0]
REASONING: [Brief explanation of why this score was assigned]"""

    human_prompt = f"""QUESTION: {question}

REFERENCE (Ground Truth): {reference}

ANSWER: {answer}

Evaluate the answer's factual correctness against the reference."""

    try:
        if SystemMessage and HumanMessage:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            response = llm.invoke(messages)
        else:
            # Fallback for older LangChain versions
            response = llm.invoke(f"{system_prompt}\n\n{human_prompt}")
        
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse score and reasoning
        score_match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", content)
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", content, re.DOTALL)
        
        score = float(score_match.group(1)) if score_match else 0.0
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Could not parse reasoning"
        
        # Clamp score to valid range
        score = max(0.0, min(1.0, score))
        
        return score, reasoning
        
    except Exception as e:
        return 0.0, f"Evaluation failed: {str(e)}"

@trace_func
def _calculate_context_precision(question: str, reference: str, contexts: List[str], llm) -> Tuple[float, str]:
    """Calculate how many retrieved contexts are relevant to answering the question"""
    
    if not contexts:
        return 0.0, "No contexts provided"
    
    system_prompt = """You are an expert evaluator for information retrieval systems.

Your task is to determine how many of the RETRIEVED CONTEXTS are actually relevant and useful for answering the QUESTION, given the REFERENCE as ground truth.

For each context, determine:
- RELEVANT: Contains information needed to answer the question
- NOT RELEVANT: Contains unrelated or unhelpful information

Calculate precision as: (Number of Relevant Contexts) / (Total Number of Contexts)

Respond with:
RELEVANT_COUNT: [number]
TOTAL_COUNT: [number]
PRECISION: [0.0-1.0]
REASONING: [Brief explanation of which contexts were relevant/irrelevant]"""

    contexts_text = "\n\n".join([f"CONTEXT {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
    
    human_prompt = f"""QUESTION: {question}

REFERENCE (Ground Truth): {reference}

RETRIEVED CONTEXTS:
{contexts_text}

Evaluate the precision of the retrieved contexts."""

    try:
        if SystemMessage and HumanMessage:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            response = llm.invoke(messages)
        else:
            response = llm.invoke(f"{system_prompt}\n\n{human_prompt}")
        
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse results
        relevant_match = re.search(r"RELEVANT_COUNT:\s*(\d+)", content)
        total_match = re.search(r"TOTAL_COUNT:\s*(\d+)", content)
        precision_match = re.search(r"PRECISION:\s*([0-9]*\.?[0-9]+)", content)
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", content, re.DOTALL)
        
        relevant_count = int(relevant_match.group(1)) if relevant_match else 0
        total_count = int(total_match.group(1)) if total_match else len(contexts)
        precision = float(precision_match.group(1)) if precision_match else (relevant_count / max(1, total_count))
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Could not parse reasoning"
        
        # Ensure precision is valid
        precision = max(0.0, min(1.0, precision))
        
        return precision, reasoning
        
    except Exception as e:
        # Fallback: simple heuristic
        return 0.5, f"LLM evaluation failed, using fallback: {str(e)}"

@trace_func
def _calculate_context_recall(question: str, reference: str, contexts: List[str], llm) -> Tuple[float, str]:
    """Calculate if all information needed to answer the question is present in contexts"""
    
    if not contexts:
        return 0.0, "No contexts provided"
    
    system_prompt = """You are an expert evaluator for information retrieval systems.

Your task is to determine if the RETRIEVED CONTEXTS contain ALL the information needed to answer the QUESTION, using the REFERENCE as the complete ground truth.

Consider:
- Does the context contain all key facts from the reference?
- Is any critical information missing that would prevent a correct answer?
- Focus on completeness, not redundancy

Calculate recall as the proportion of reference information found in contexts.

Respond with:
RECALL: [0.0-1.0]
REASONING: [Brief explanation of what information was found/missing]"""

    contexts_text = "\n\n".join([f"CONTEXT {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
    
    human_prompt = f"""QUESTION: {question}

REFERENCE (Ground Truth): {reference}

RETRIEVED CONTEXTS:
{contexts_text}

Evaluate if contexts contain all information needed to generate the reference answer."""

    try:
        if SystemMessage and HumanMessage:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            response = llm.invoke(messages)
        else:
            response = llm.invoke(f"{system_prompt}\n\n{human_prompt}")
        
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse results
        recall_match = re.search(r"RECALL:\s*([0-9]*\.?[0-9]+)", content)
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", content, re.DOTALL)
        
        recall = float(recall_match.group(1)) if recall_match else 0.0
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Could not parse reasoning"
        
        # Ensure recall is valid
        recall = max(0.0, min(1.0, recall))
        
        return recall, reasoning
        
    except Exception as e:
        return 0.5, f"LLM evaluation failed, using fallback: {str(e)}"

@trace_func
def _calculate_faithfulness(question: str, answer: str, contexts: List[str], llm) -> Tuple[float, str]:
    """Calculate if the answer is faithful to the provided contexts (no hallucination)"""
    
    if not contexts:
        return 0.0, "No contexts to verify against"
    
    system_prompt = """You are an expert evaluator for factual accuracy in Q&A systems.

Your task is to determine if the ANSWER is faithful to the provided CONTEXTS - meaning all facts in the answer can be verified from the contexts.

Consider:
- Does the answer contain facts not present in contexts? (hallucination)
- Are all claims in the answer supported by the contexts?
- Does the answer contradict the contexts?

Calculate faithfulness as the proportion of answer content that is verifiable from contexts.

Respond with:
FAITHFULNESS: [0.0-1.0]
REASONING: [Brief explanation of any unsupported claims or contradictions]"""

    contexts_text = "\n\n".join([f"CONTEXT {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
    
    human_prompt = f"""QUESTION: {question}

ANSWER: {answer}

CONTEXTS:
{contexts_text}

Evaluate if the answer is faithful to the contexts (no hallucination)."""

    try:
        if SystemMessage and HumanMessage:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            response = llm.invoke(messages)
        else:
            response = llm.invoke(f"{system_prompt}\n\n{human_prompt}")
        
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse results
        faithfulness_match = re.search(r"FAITHFULNESS:\s*([0-9]*\.?[0-9]+)", content)
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", content, re.DOTALL)
        
        faithfulness = float(faithfulness_match.group(1)) if faithfulness_match else 0.0
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Could not parse reasoning"
        
        # Ensure faithfulness is valid
        faithfulness = max(0.0, min(1.0, faithfulness))
        
        return faithfulness, reasoning
        
    except Exception as e:
        return 0.5, f"LLM evaluation failed, using fallback: {str(e)}"

@trace_func
def _calculate_table_qa_accuracy(question: str, answer: str, reference: str, llm) -> Tuple[float, str]:
    """Special accuracy calculation for table/numerical questions"""
    
    system_prompt = """You are an expert evaluator for technical table and numerical Q&A.

Your task is to score the accuracy of numerical/tabular answers with high precision.

For table/numerical questions, consider:
- Exact numerical matches: 1.0
- Correct value with minor format differences: 0.95
- Correct order of magnitude, minor error: 0.8
- Partially correct (some digits right): 0.6
- Wrong value but correct units/format: 0.3
- Completely wrong: 0.0

For date/time questions:
- Exact match: 1.0
- Correct date, minor format difference: 0.95
- Correct month/year, wrong day: 0.8
- Wrong date: 0.0

Respond with:
ACCURACY: [0.0-1.0]
REASONING: [Brief explanation focusing on numerical/tabular precision]"""

    human_prompt = f"""QUESTION: {question}

REFERENCE (Ground Truth): {reference}

ANSWER: {answer}

Evaluate the table/numerical accuracy."""

    try:
        if SystemMessage and HumanMessage:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            response = llm.invoke(messages)
        else:
            response = llm.invoke(f"{system_prompt}\n\n{human_prompt}")
        
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse results
        accuracy_match = re.search(r"ACCURACY:\s*([0-9]*\.?[0-9]+)", content)
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", content, re.DOTALL)
        
        accuracy = float(accuracy_match.group(1)) if accuracy_match else 0.0
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Could not parse reasoning"
        
        # Ensure accuracy is valid
        accuracy = max(0.0, min(1.0, accuracy))
        
        return accuracy, reasoning
        
    except Exception as e:
        return 0.5, f"LLM evaluation failed, using fallback: {str(e)}"

@trace_func
def evaluate_question(
    question: str,
    answer: str,
    reference: str,
    contexts: List[str],
    thresholds: EvaluationThresholds,
    llm
) -> QuestionMetrics:
    """Evaluate a single question with manual thresholds"""
    
    is_table = _is_table_question(question)
    
    # Calculate core metrics
    correctness, corr_reasoning = _calculate_answer_correctness(question, answer, reference, llm)
    precision, prec_reasoning = _calculate_context_precision(question, reference, contexts, llm)
    recall, rec_reasoning = _calculate_context_recall(question, reference, contexts, llm)
    faithfulness, faith_reasoning = _calculate_faithfulness(question, answer, contexts, llm)
    
    # Table-specific accuracy
    table_accuracy = None
    table_reasoning = ""
    if is_table:
        table_accuracy, table_reasoning = _calculate_table_qa_accuracy(question, answer, reference, llm)
    
    # Check if passes all thresholds
    passes = (
        correctness >= thresholds.answer_correctness and
        precision >= thresholds.context_precision and
        recall >= thresholds.context_recall and
        faithfulness >= thresholds.faithfulness and
        (not is_table or (table_accuracy is not None and table_accuracy >= thresholds.table_qa_accuracy))
    )
    
    # Combine reasoning
    reasoning_parts = [
        f"Correctness: {corr_reasoning}",
        f"Precision: {prec_reasoning}",
        f"Recall: {rec_reasoning}",
        f"Faithfulness: {faith_reasoning}"
    ]
    if is_table and table_reasoning:
        reasoning_parts.append(f"Table Accuracy: {table_reasoning}")
    
    combined_reasoning = " | ".join(reasoning_parts)
    
    return QuestionMetrics(
        question=question,
        answer=answer,
        reference=reference,
        contexts=contexts,
        answer_correctness=correctness,
        context_precision=precision,
        context_recall=recall,
        faithfulness=faithfulness,
        table_qa_accuracy=table_accuracy,
        passes_thresholds=passes,
        is_table_question=is_table,
        reasoning=combined_reasoning
    )

@trace_func
def run_manual_evaluation(
    dataset: Dict[str, List[Any]],
    thresholds: Optional[EvaluationThresholds] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run complete manual evaluation with user-defined thresholds"""
    
    if thresholds is None:
        thresholds = EvaluationThresholds()
    
    # Setup LLM judge
    try:
        llm = _setup_llm_judge()
    except Exception as e:
        return {"error": f"Failed to setup LLM judge: {e}"}, []
    
    # Extract data
    questions = dataset.get("question", [])
    answers = dataset.get("answer", [])
    references = dataset.get("reference", [])
    contexts_list = dataset.get("contexts", [])
    
    if not questions:
        return {"error": "No questions found in dataset"}, []
    
    # Evaluate each question
    results = []
    for i in range(len(questions)):
        q = questions[i] if i < len(questions) else ""
        a = answers[i] if i < len(answers) else ""
        ref = references[i] if i < len(references) else ""
        ctxs = contexts_list[i] if i < len(contexts_list) else []
        
        try:
            metrics = evaluate_question(q, a, ref, ctxs, thresholds, llm)
            
            # Convert to dict for compatibility
            result = {
                "question": metrics.question,
                "answer": metrics.answer,
                "reference": metrics.reference,
                "contexts": metrics.contexts,
                "answer_correctness": metrics.answer_correctness,
                "context_precision": metrics.context_precision,
                "context_recall": metrics.context_recall,
                "faithfulness": metrics.faithfulness,
                "passes_thresholds": metrics.passes_thresholds,
                "is_table_question": metrics.is_table_question,
                "reasoning": metrics.reasoning
            }
            
            if metrics.table_qa_accuracy is not None:
                result["table_qa_accuracy"] = metrics.table_qa_accuracy
            
            results.append(result)
            
        except Exception as e:
            # Add failed evaluation record
            results.append({
                "question": q,
                "answer": a,
                "reference": ref,
                "contexts": ctxs,
                "answer_correctness": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "faithfulness": 0.0,
                "passes_thresholds": False,
                "is_table_question": _is_table_question(q),
                "reasoning": f"Evaluation failed: {str(e)}"
            })
    
    # Calculate summary statistics
    def safe_mean(values):
        nums = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
        return sum(nums) / len(nums) if nums else 0.0
    
    correctness_scores = [r["answer_correctness"] for r in results]
    precision_scores = [r["context_precision"] for r in results]
    recall_scores = [r["context_recall"] for r in results]
    faithfulness_scores = [r["faithfulness"] for r in results]
    table_scores = [r.get("table_qa_accuracy") for r in results if r.get("table_qa_accuracy") is not None]
    
    total_questions = len(results)
    passed_questions = sum(1 for r in results if r["passes_thresholds"])
    table_questions = sum(1 for r in results if r["is_table_question"])
    
    summary = {
        "total_questions": total_questions,
        "passed_questions": passed_questions,
        "pass_rate": passed_questions / max(1, total_questions),
        "table_questions": table_questions,
        
        # Average scores
        "avg_answer_correctness": safe_mean(correctness_scores),
        "avg_context_precision": safe_mean(precision_scores),
        "avg_context_recall": safe_mean(recall_scores),
        "avg_faithfulness": safe_mean(faithfulness_scores),
        
        # Threshold compliance
        "correctness_above_threshold": sum(1 for s in correctness_scores if s >= thresholds.answer_correctness),
        "precision_above_threshold": sum(1 for s in precision_scores if s >= thresholds.context_precision),
        "recall_above_threshold": sum(1 for s in recall_scores if s >= thresholds.context_recall),
        "faithfulness_above_threshold": sum(1 for s in faithfulness_scores if s >= thresholds.faithfulness),
        
        # Thresholds used
        "thresholds": {
            "answer_correctness": thresholds.answer_correctness,
            "context_precision": thresholds.context_precision,
            "context_recall": thresholds.context_recall,
            "faithfulness": thresholds.faithfulness,
            "table_qa_accuracy": thresholds.table_qa_accuracy
        }
    }
    
    if table_scores:
        summary["avg_table_qa_accuracy"] = safe_mean(table_scores)
        summary["table_accuracy_above_threshold"] = sum(1 for s in table_scores if s >= thresholds.table_qa_accuracy)
    
    return summary, results

@trace_func
def pretty_manual_metrics(summary: Dict[str, Any]) -> str:
    """Format manual evaluation results for display"""
    
    if "error" in summary:
        return f"❌ **Manual Evaluation Failed**: {summary['error']}"
    
    total = summary.get("total_questions", 0)
    passed = summary.get("passed_questions", 0)
    pass_rate = summary.get("pass_rate", 0.0)
    table_count = summary.get("table_questions", 0)
    
    thresholds = summary.get("thresholds", {})
    
    # Main metrics
    corr = summary.get("avg_answer_correctness", 0.0)
    prec = summary.get("avg_context_precision", 0.0)
    rec = summary.get("avg_context_recall", 0.0)
    faith = summary.get("avg_faithfulness", 0.0)
    
    # Threshold compliance
    corr_pass = summary.get("correctness_above_threshold", 0)
    prec_pass = summary.get("precision_above_threshold", 0)
    rec_pass = summary.get("recall_above_threshold", 0)
    faith_pass = summary.get("faithfulness_above_threshold", 0)
    
    result = f"""## 📋 **Manual Factual Evaluation Results**

### 🎯 **Overall Performance**
- **Questions Evaluated**: {total}
- **Passed All Thresholds**: {passed}/{total} ({pass_rate:.1%})
- **Table Questions**: {table_count}

### 📊 **Metric Scores & Compliance**
- **Answer Correctness**: {corr:.3f} (≥{thresholds.get('answer_correctness', 0.8):.2f}: {corr_pass}/{total})
- **Context Precision**: {prec:.3f} (≥{thresholds.get('context_precision', 0.75):.2f}: {prec_pass}/{total})
- **Context Recall**: {rec:.3f} (≥{thresholds.get('context_recall', 0.70):.2f}: {rec_pass}/{total})
- **Faithfulness**: {faith:.3f} (≥{thresholds.get('faithfulness', 0.85):.2f}: {faith_pass}/{total})"""

    # Add table accuracy if present
    if "avg_table_qa_accuracy" in summary:
        table_acc = summary["avg_table_qa_accuracy"]
        table_pass = summary.get("table_accuracy_above_threshold", 0)
        result += f"\n- **Table-QA Accuracy**: {table_acc:.3f} (≥{thresholds.get('table_qa_accuracy', 0.90):.2f}: {table_pass}/{table_count})"
    
    # Status indicators
    if pass_rate >= 0.8:
        result += "\n\n✅ **System Performance: EXCELLENT**"
    elif pass_rate >= 0.6:
        result += "\n\n⚠️ **System Performance: NEEDS IMPROVEMENT**"
    else:
        result += "\n\n❌ **System Performance: REQUIRES ATTENTION**"
    
    return result
