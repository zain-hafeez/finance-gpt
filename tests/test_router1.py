# tests/test_router.py
"""
Module 5 — Query Router Tests

Tests cover:
  1.  SQL question → "sql" classification
  2.  Stats question → "stats" classification
  3.  LLM returns extra text (e.g. "sql query") → still parses to "sql"
  4.  LLM returns extra text (e.g. "stats analysis") → still parses to "stats"
  5.  LLM returns unexpected garbage → defaults to "sql"
  6.  LLM raises exception → defaults to "sql", no crash
  7.  Empty question → defaults to "sql", no crash
  8.  Case insensitive parsing ("SQL", "STATS", "Stats")
  9.  router_node reads from state dict correctly
  10. router_node handles missing query key gracefully
  11. route_to_engine returns correct node name
  12. route_to_engine handles invalid query_type gracefully
"""

import pytest
from unittest.mock import MagicMock

from src.graph.router import classify_query, router_node, route_to_engine


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_mock_llm(response_text: str):
    """Build a mock LLM that returns a fixed response string."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_text
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# ---------------------------------------------------------------------------
# 1. SQL question → "sql"
# ---------------------------------------------------------------------------

def test_sql_question_classifies_as_sql():
    """A clear aggregation question should return 'sql'."""
    mock_llm = _make_mock_llm("sql")
    result = classify_query("What is the total sales amount?", mock_llm)
    assert result == "sql"


# ---------------------------------------------------------------------------
# 2. Stats question → "stats"
# ---------------------------------------------------------------------------

def test_stats_question_classifies_as_stats():
    """A forecasting question should return 'stats'."""
    mock_llm = _make_mock_llm("stats")
    result = classify_query("What is the 3-month moving average of sales?", mock_llm)
    assert result == "stats"


# ---------------------------------------------------------------------------
# 3. LLM returns "sql" with extra words → still "sql"
# ---------------------------------------------------------------------------

def test_sql_with_extra_words_still_parses():
    """LLM says 'sql query' or 'sql - this needs aggregation' → parse to 'sql'."""
    mock_llm = _make_mock_llm("sql - this is an aggregation query")
    result = classify_query("Show me top 5 products", mock_llm)
    assert result == "sql"


# ---------------------------------------------------------------------------
# 4. LLM returns "stats" with extra words → still "stats"
# ---------------------------------------------------------------------------

def test_stats_with_extra_words_still_parses():
    """LLM says 'stats analysis needed' → parse to 'stats'."""
    mock_llm = _make_mock_llm("stats - correlation analysis required")
    result = classify_query("Is price correlated with quantity?", mock_llm)
    assert result == "stats"


# ---------------------------------------------------------------------------
# 5. LLM returns garbage → defaults to "sql"
# ---------------------------------------------------------------------------

def test_unexpected_llm_response_defaults_to_sql():
    """If LLM returns something we don't understand, default to 'sql'."""
    mock_llm = _make_mock_llm("I need more information to classify this.")
    result = classify_query("Something ambiguous", mock_llm)
    assert result == "sql"


# ---------------------------------------------------------------------------
# 6. LLM raises exception → defaults to "sql", no crash
# ---------------------------------------------------------------------------

def test_llm_exception_returns_sql_no_crash():
    """If Groq is down and LLM raises, classify_query returns 'sql' safely."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("Groq API rate limit exceeded")

    result = classify_query("What is total revenue?", mock_llm)
    
    assert result == "sql"  # Safe default
    # No exception raised — this is the critical guarantee


# ---------------------------------------------------------------------------
# 7. Empty question → defaults to "sql", no crash
# ---------------------------------------------------------------------------

def test_empty_question_returns_sql():
    """Empty question should return 'sql' without calling the LLM at all."""
    mock_llm = MagicMock()
    
    result = classify_query("", mock_llm)
    
    assert result == "sql"
    mock_llm.invoke.assert_not_called()  # LLM never called for empty input


def test_whitespace_only_question_returns_sql():
    """Whitespace-only question treated as empty."""
    mock_llm = MagicMock()
    result = classify_query("   ", mock_llm)
    assert result == "sql"
    mock_llm.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# 8. Case insensitive parsing
# ---------------------------------------------------------------------------

def test_uppercase_sql_response_parses_correctly():
    """LLM returns 'SQL' (uppercase) → should parse to 'sql'."""
    mock_llm = _make_mock_llm("SQL")
    result = classify_query("Total revenue?", mock_llm)
    assert result == "sql"


def test_uppercase_stats_response_parses_correctly():
    """LLM returns 'STATS' → should parse to 'stats'."""
    mock_llm = _make_mock_llm("STATS")
    result = classify_query("Moving average?", mock_llm)
    assert result == "stats"


def test_mixed_case_stats_response_parses_correctly():
    """LLM returns 'Stats' → should parse to 'stats'."""
    mock_llm = _make_mock_llm("Stats")
    result = classify_query("Forecast next month?", mock_llm)
    assert result == "stats"


# ---------------------------------------------------------------------------
# 9. router_node reads from state dict correctly
# ---------------------------------------------------------------------------

def test_router_node_reads_query_from_state(monkeypatch):
    """router_node should read state['query'] and write state['query_type']."""
    # Mock the get_llm import inside router_node
    mock_llm = _make_mock_llm("sql")
    monkeypatch.setattr("src.graph.router.get_llm", lambda task: mock_llm, raising=False)
    
    # Import get_llm into router module's namespace for monkeypatching to work
    import src.graph.router as router_module
    router_module.get_llm = lambda task: mock_llm
    
    state = {
        "query": "What is the total revenue?",
        "file_path": "data/sample.csv",
        "db_path": "finance.db",
        "session_id": "test-session"
    }
    
    result = router_node(state)
    
    assert "query_type" in result
    assert result["query_type"] == "sql"
    assert result["error"] is None


# ---------------------------------------------------------------------------
# 10. router_node handles missing query key gracefully
# ---------------------------------------------------------------------------

def test_router_node_handles_missing_query():
    """If state has no 'query' key, router_node should not crash."""
    state = {}  # Empty state — no query key
    
    result = router_node(state)
    
    # Should return a valid partial state, not crash
    assert "query_type" in result
    assert result["query_type"] == "sql"  # Safe default
    assert "error" in result


# ---------------------------------------------------------------------------
# 11. route_to_engine returns correct node name
# ---------------------------------------------------------------------------

def test_route_to_engine_returns_sql_for_sql_query_type():
    """route_to_engine("sql" in state) → returns "sql"."""
    state = {"query_type": "sql"}
    assert route_to_engine(state) == "sql"


def test_route_to_engine_returns_stats_for_stats_query_type():
    """route_to_engine("stats" in state) → returns "stats"."""
    state = {"query_type": "stats"}
    assert route_to_engine(state) == "stats"


# ---------------------------------------------------------------------------
# 12. route_to_engine handles invalid query_type gracefully
# ---------------------------------------------------------------------------

def test_route_to_engine_handles_invalid_query_type():
    """If query_type is something unexpected, route to 'sql' safely."""
    state = {"query_type": "unknown_value"}
    result = route_to_engine(state)
    assert result == "sql"


def test_route_to_engine_handles_missing_query_type():
    """If query_type key is missing from state, route to 'sql' safely."""
    state = {}  # No query_type key
    result = route_to_engine(state)
    assert result == "sql"