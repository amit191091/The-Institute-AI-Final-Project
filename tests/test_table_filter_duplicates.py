import io
import json
import warnings
import pandas as pd

from app.agent_tools import df_to_rows_with_col_index, _flatten_and_uniquify_columns


def test_flatten_and_uniquify_columns_simple_dupes():
    cols = ["A", "B", "A", None, "B"]
    out = _flatten_and_uniquify_columns(cols)
    assert out == ["A", "B", "A_2", "col", "B_2"]


def test_df_to_rows_with_col_index_preserves_order_and_maps_positions():
    # Construct a DataFrame with duplicate and non-string column names
    df = pd.DataFrame(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ]
    )
    df.columns = ["A", "B", "A", None, "B"]

    # Ensure no warnings are raised and that rows contain __col_index__ mapping
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rows = df_to_rows_with_col_index(df, max_rows=2)
        # No warnings expected (we dedupe before to_dict)
        assert not any("columns are not unique" in str(x.message) for x in w)

    assert len(rows) == 2
    # First row values under deduped names
    r0 = rows[0]
    assert "__col_index__" in r0
    idx_map = r0["__col_index__"]
    # Deduped column names should align with positions
    # Expected after dedupe: ["A", "B", "A_2", "col", "B_2"]
    assert idx_map == {"A": 0, "B": 1, "A_2": 2, "col": 3, "B_2": 4}
    # Values preserved per position
    assert r0["A"] == 1
    assert r0["B"] == 2
    assert r0["A_2"] == 3
    assert r0["col"] == 4
    assert r0["B_2"] == 5
