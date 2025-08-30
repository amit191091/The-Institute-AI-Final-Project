# RAGAS imports with robust fallbacks
try:
	from ragas import evaluate
	from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
	try:
		from ragas.run_config import RunConfig  # optional in some versions
	except Exception:  # pragma: no cover
		RunConfig = None  # type: ignore
except Exception:  # pragma: no cover
	evaluate = None  # type: ignore
	faithfulness = answer_relevancy = context_precision = context_recall = None  # type: ignore
	RunConfig = None  # type: ignore

# Optional Google safety enums (available in newer google-generativeai)
try:
	from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore
except Exception:  # pragma: no cover
	HarmCategory = None  # type: ignore
	HarmBlockThreshold = None  # type: ignore

try:
	from datasets import Dataset  # type: ignore
except Exception:  # pragma: no cover
	Dataset = None  # type: ignore

import os
import math
import re
import json
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from difflib import SequenceMatcher
from RAG.app.config import settings
from RAG.app.logger import get_logger


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Configuration is now centralized in config.py
# Access via: settings.evaluation.ANSWER_CORRECTNESS_WEIGHTS
# Access via: settings.evaluation.EVALUATION_TARGETS
# Access via: settings.evaluation.ANSWER_CORRECTNESS_WEIGHT_LIST


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Import from text_processors module
from .text_processors import (
    extract_numbers,
    extract_dates,
    extract_technical_terms,
    extract_citations,
    simple_similarity
)


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

# Import from answer_evaluators module
from .answer_evaluators import (
    calculate_answer_correctness,
    evaluate_answer_simple
)


# ============================================================================
# RAGAS SETUP FUNCTIONS
# ============================================================================

# Import from ragas_setup module
from .ragas_setup import _setup_ragas_llm, _setup_ragas_embeddings


# ============================================================================
# RAGAS EVALUATION FUNCTIONS
# ============================================================================

# Import from ragas_evaluators module
from .ragas_evaluators import (
    run_eval,
    run_eval_detailed,
    pretty_metrics
)


# ============================================================================
# ADVANCED EVALUATION UTILITIES
# ============================================================================

# Import advanced utilities from eval_ragas.py script
from .advanced_evaluation_utils import (
    overlap_prf1,
    _mean_safe,
    _maybe_float,
    _pick,
    _is_table_like_question,
    _table_correct,
    append_eval_footer
)


# ============================================================================
# RAG SYSTEM EVALUATION
# ============================================================================

# Import from system_evaluators module
from .system_evaluators import evaluate_rag_system



# ============================================================================
# TABLE-QA SPECIFIC EVALUATION
# ============================================================================

# Import from table_qa_evaluators module
from .table_qa_evaluators import calculate_table_qa_accuracy

# ============================================================================
# TARGET COMPLIANCE FUNCTIONS
# ============================================================================

# Import from compliance_checkers module
from .compliance_checkers import check_target_compliance, print_target_compliance
