# tests/test_workflow.py
"""
Module 6 — Workflow Tests

Fix applied: get_llm is imported lazily (inside function bodies) in nodes.py
and router.py to avoid circular imports. Therefore we patch at the SOURCE:
    patch("src.utils.llm_router.get_llm")
NOT at the usage site:
    patch("src.graph.nodes.get_llm")  ← this fails — name doesn't exist at module level
"""

import pytest
from unittest.mock import MagicMock, patch

from src.graph.nodes import sql_node, stats_node, explain_node
from src.graph.workflow import run_query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm(response_text: str):
    """Build a mock LLM that always returns a fixed response string."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_text
    mock_llm.invoke.return_value = mock_response
    return mock_llm

EXPECTED_KEYS = {"query_type", "sql_query", "stats_code", "raw_result", "explanation", "error", "cached"}

# Patch target — the one place get_llm actually lives
GET_LLM = "src.utils.llm_router.get_llm"


# ---------------------------------------------------------------------------
# 1. SQL path — end-to-end happy path
# ---------------------------------------------------------------------------

def test_run_query_sql_path_returns_success(tmp_path):
    """Full pipeline, SQL path: should return success=True with explanation."""
    import pandas as pd
    from sqlalchemy import create_engine

    csv_path = tmp_path / "sales.csv"
    db_path  = tmp_path / "test.db"

    df = pd.DataFrame({
        "region":      ["North", "South"],
        "total_sales": [500.0,   800.0],
    })
    df.to_csv(csv_path, index=False)
    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql("data", engine, if_exists="replace", index=False)
    engine.dispose()

    # get_llm is called 3 times in the full pipeline:
    #   1. router_node        → must return "sql"
    #   2. sql_node (SQL gen) → must return a SELECT query
    #   3. explain_node       → must return explanation text
    with patch(GET_LLM) as mock_get_llm:
        mock_get_llm.side_effect = [
            _make_mock_llm("sql"),
            _make_mock_llm("SELECT SUM(total_sales) FROM data"),
            _make_mock_llm("Total sales is $1,300."),
        ]
        result = run_query(
            question="What is the total sales?",
            file_path=str(csv_path),
            db_path=str(db_path),
        )

    assert result["error"] is None
    assert result["explanation"] != ""
    assert result["query_type"] == "sql"
    assert EXPECTED_KEYS.issubset(result.keys())


# ---------------------------------------------------------------------------
# 2. Stats path — end-to-end happy path
# ---------------------------------------------------------------------------

def test_run_query_stats_path_returns_success(tmp_path):
    """Full pipeline, stats path: should return success=True with explanation."""
    import pandas as pd
    from sqlalchemy import create_engine

    csv_path = tmp_path / "sales.csv"
    db_path  = tmp_path / "test.db"

    df = pd.DataFrame({
        "total_sales": [100.0, 120.0, 110.0, 130.0, 115.0]
    })
    df.to_csv(csv_path, index=False)
    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql("data", engine, if_exists="replace", index=False)
    engine.dispose()

    # get_llm called 3 times:
    #   1. router_node  → "stats"
    #   2. stats_node   → valid pandas code
    #   3. explain_node → explanation text
    with patch(GET_LLM) as mock_get_llm:
        mock_get_llm.side_effect = [
            _make_mock_llm("stats"),
            _make_mock_llm("result = df['total_sales'].rolling(3).mean().dropna().tolist()"),
            _make_mock_llm("The moving average shows a steady upward trend."),
        ]
        result = run_query(
            question="What is the 3-period moving average of sales?",
            file_path=str(csv_path),
            db_path=str(db_path),
        )

    assert result["error"] is None
    assert result["explanation"] != ""
    assert result["query_type"] == "stats"
    assert EXPECTED_KEYS.issubset(result.keys())


# ---------------------------------------------------------------------------
# 3–5. Input validation
# ---------------------------------------------------------------------------

def test_run_query_empty_question_returns_error():
    result = run_query(question="", file_path="some/path.csv", db_path="some/db.db")
    assert result["error"] is not None
    assert EXPECTED_KEYS.issubset(result.keys())


def test_run_query_missing_file_path_returns_error():
    result = run_query(question="What is total sales?", file_path="", db_path="some/db.db")
    assert result["error"] is not None
    assert EXPECTED_KEYS.issubset(result.keys())


def test_run_query_missing_db_path_returns_error():
    result = run_query(question="What is total sales?", file_path="some/path.csv", db_path="")
    assert result["error"] is not None
    assert EXPECTED_KEYS.issubset(result.keys())


# ---------------------------------------------------------------------------
# 6–7. Return dict always has all expected keys
# ---------------------------------------------------------------------------

def test_run_query_success_has_all_keys(tmp_path):
    """Success response must always have all expected keys — no KeyError ever."""
    import pandas as pd
    from sqlalchemy import create_engine

    csv_path = tmp_path / "s.csv"
    db_path  = tmp_path / "t.db"
    df = pd.DataFrame({"total_sales": [100.0, 200.0]})
    df.to_csv(csv_path, index=False)
    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql("data", engine, if_exists="replace", index=False)
    engine.dispose()

    with patch(GET_LLM) as mock_get_llm:
        mock_get_llm.side_effect = [
            _make_mock_llm("sql"),
            _make_mock_llm("SELECT SUM(total_sales) FROM data"),
            _make_mock_llm("Total is 300."),
        ]
        result = run_query("Total sales?", str(csv_path), str(db_path))

    assert EXPECTED_KEYS.issubset(result.keys())


def test_run_query_error_has_all_keys():
    """Error response must always have the same keys — no KeyError ever."""
    result = run_query(question="", file_path="", db_path="")
    assert EXPECTED_KEYS.issubset(result.keys())


# ---------------------------------------------------------------------------
# 8. sql_node: success
# ---------------------------------------------------------------------------

def test_sql_node_success_writes_correct_state_fields(tmp_path):
    """sql_node success → sql_query set, sql_valid=True, raw_result set, error=None."""
    import pandas as pd
    from sqlalchemy import create_engine

    db_path = tmp_path / "test.db"
    df = pd.DataFrame({"total_sales": [100.0, 200.0]})
    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql("data", engine, if_exists="replace", index=False)
    engine.dispose()

    with patch(GET_LLM) as mock_get_llm:
        mock_get_llm.return_value = _make_mock_llm("SELECT SUM(total_sales) FROM data")

        state  = {"query": "What is total sales?", "db_path": str(db_path)}
        result = sql_node(state)

    assert "sql_query"   in result
    assert "raw_result"  in result
    assert result["sql_valid"] is True
    assert result["error"] is None


# ---------------------------------------------------------------------------
# 9. sql_node: LLM failure → clean error in state
# ---------------------------------------------------------------------------

def test_sql_node_llm_failure_writes_error():
    """sql_node: if get_llm raises, error written to state — no crash."""
    with patch(GET_LLM) as mock_get_llm:
        mock_get_llm.side_effect = Exception("API down")
        state  = {"query": "Total sales?", "db_path": "finance.db"}
        result = sql_node(state)

    assert result.get("sql_valid") is False
    assert result.get("error") is not None


# ---------------------------------------------------------------------------
# 10. stats_node: success
# ---------------------------------------------------------------------------

def test_stats_node_success_writes_correct_state_fields(tmp_path):
    """stats_node success → stats_code set, raw_result set, error=None."""
    import pandas as pd

    csv_path = tmp_path / "sales.csv"
    df = pd.DataFrame({"total_sales": [100.0, 120.0, 110.0, 130.0, 115.0]})
    df.to_csv(csv_path, index=False)

    with patch(GET_LLM) as mock_get_llm:
        mock_get_llm.return_value = _make_mock_llm(
            "result = df['total_sales'].mean()"
        )
        state  = {"query": "Average sales?", "file_path": str(csv_path)}
        result = stats_node(state)

    assert "stats_code" in result
    assert "raw_result" in result
    assert result.get("error") is None


# ---------------------------------------------------------------------------
# 11. stats_node: missing file_path
# ---------------------------------------------------------------------------

def test_stats_node_missing_file_path_writes_error():
    """stats_node: no file_path in state → clean error, no crash."""
    state  = {"query": "Moving average?", "file_path": ""}
    result = stats_node(state)
    assert result.get("error") is not None
    assert result.get("stats_code") is None


# ---------------------------------------------------------------------------
# 12. explain_node: upstream error → friendly message, no LLM call
# ---------------------------------------------------------------------------

def test_explain_node_upstream_error_writes_friendly_explanation():
    """If state has an error, explain_node writes friendly message — LLM never called."""
    state = {
        "query":      "Total sales?",
        "raw_result": None,
        "query_type": "sql",
        "sql_query":  None,
        "error":      "SQL validation failed: DROP detected",
    }
    with patch(GET_LLM) as mock_get_llm:
        result = explain_node(state)
        mock_get_llm.assert_not_called()

    assert "explanation" in result
    assert len(result["explanation"]) > 0


# ---------------------------------------------------------------------------
# 13. explain_node: no raw_result → graceful message
# ---------------------------------------------------------------------------

def test_explain_node_no_result_writes_graceful_message():
    """explain_node: raw_result=None + no error → graceful 'no results' message."""
    state = {
        "query":      "Something",
        "raw_result": None,
        "query_type": "sql",
        "sql_query":  "SELECT ...",
        "error":      None,
    }
    # raw_result is None → explain_node returns early without calling LLM
    result = explain_node(state)

    assert "explanation" in result
    assert len(result["explanation"]) > 0


# ---------------------------------------------------------------------------
# 14. explain_node: LLM failure → falls back to raw result string
# ---------------------------------------------------------------------------

def test_explain_node_llm_failure_falls_back_to_raw_result():
    """explain_node: if LLM raises, fall back to raw result string — no crash."""
    state = {
        "query":      "Total sales?",
        "raw_result": "[(1300.0,)]",
        "query_type": "sql",
        "sql_query":  "SELECT SUM(total_sales) FROM data",
        "error":      None,
    }
    with patch(GET_LLM) as mock_get_llm:
        mock_get_llm.side_effect = Exception("Groq down")
        result = explain_node(state)

    assert "explanation" in result
    assert len(result["explanation"]) > 0