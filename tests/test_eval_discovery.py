import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running tests directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline import _load_json_or_jsonl, _index_ground_truth


def test_gt_qa_files_present_and_indexed():
    gt_path = Path("data/gear_wear_ground_truth_context_free.json")
    qa_path = Path("data/gear_wear_qa_context_free.jsonl")
    assert gt_path.exists(), "expected ground truth file missing"
    assert qa_path.exists(), "expected QA file missing"

    gt_rows = _load_json_or_jsonl(gt_path)
    qa_rows = _load_json_or_jsonl(qa_path)
    assert isinstance(gt_rows, list) and len(gt_rows) > 0
    assert isinstance(qa_rows, list) and len(qa_rows) > 0

    gt_by_id, gt_by_q = _index_ground_truth(gt_rows)
    # Context-free GT will populate by_id, possibly empty by_q
    assert len(gt_by_id) > 0
    # Spot-check a known id from the sample data
    assert "gwf-v2-016" in gt_by_id