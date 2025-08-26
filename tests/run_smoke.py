import sys
from pathlib import Path

# Ensure project root is on sys.path so 'app' package can be imported when running this file directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smoke_chunking_test import test_basic_anchors_and_ids


def main() -> int:
    try:
        test_basic_anchors_and_ids()
    except AssertionError as e:
        print(f"SMOKE TEST: FAIL - {e}")
        return 1
    except Exception as e:
        print(f"SMOKE TEST: ERROR - {e}")
        return 2
    print("SMOKE TEST: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
