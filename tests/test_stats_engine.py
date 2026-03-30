# tests/test_stats_engine.py
"""
Module 4 — Stats Engine Tests

Strategy: Mock the LLM so tests run fast, offline, and deterministically.
We test:
  1. Happy path — valid code, correct result
  2. Moving average
  3. Correlation
  4. LLM returns markdown fences (cleaning test)
  5. LLM fails (exception) → clean error dict
  6. Generated code has no 'result' variable → clean error
  7. Empty DataFrame → clean error
  8. Empty question → clean error
  9. RestrictedPython blocks dangerous code (os import)
  10. Result normalization (numpy types → python types)
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.engines.stats_engine import run_stats_query, _clean_code_output, _normalize_result
from src.security.restricted_exec import execute_restricted


# ---------------------------------------------------------------------------
# Shared test fixture — a small DataFrame that mimics real financial data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """15-row DataFrame matching the project's sample_sales.csv structure."""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=15, freq='D').astype(str),
        'region': ['North', 'South', 'East', 'West', 'North'] * 3,
        'product': ['Widget', 'Gadget', 'Widget', 'Gadget', 'Widget'] * 3,
        'quantity': [10, 20, 15, 8, 25, 12, 18, 9, 30, 14, 11, 22, 7, 16, 19],
        'unit_price': [5.0, 10.0, 5.0, 10.0, 5.0] * 3,
        'total_sales': [50.0, 200.0, 75.0, 80.0, 125.0,
                        60.0, 90.0, 90.0, 150.0, 140.0,
                        55.0, 220.0, 35.0, 160.0, 95.0],
        'salesperson': ['Alice', 'Bob', 'Carol', 'Alice', 'Bob'] * 3,
    })


def _make_mock_llm(code_response: str):
    """Helper: build a mock LLM that returns a fixed code string."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = code_response
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# ---------------------------------------------------------------------------
# 1. Happy path — simple sum
# ---------------------------------------------------------------------------

def test_simple_calculation_returns_correct_result(sample_df):
    """LLM returns valid code → result is computed correctly."""
    mock_llm = _make_mock_llm("result = float(df['total_sales'].sum())")
    
    output = run_stats_query("What is the total sales?", sample_df, mock_llm)
    
    assert output['success'] is True
    assert output['error'] is None
    assert output['result'] == pytest.approx(1625.0)
    assert 'total_sales' in output['code']


# ---------------------------------------------------------------------------
# 2. Moving average
# ---------------------------------------------------------------------------

def test_moving_average_returns_list(sample_df):
    """Rolling mean returns a list (normalized from Series)."""
    code = "result = df['total_sales'].rolling(3).mean().dropna().tolist()"
    mock_llm = _make_mock_llm(code)
    
    output = run_stats_query("What is the 3-period moving average of sales?", sample_df, mock_llm)
    
    assert output['success'] is True
    assert isinstance(output['result'], list)
    assert len(output['result']) > 0


# ---------------------------------------------------------------------------
# 3. Correlation
# ---------------------------------------------------------------------------

def test_correlation_returns_dict(sample_df):
    """df.corr() returns DataFrame → normalized to list of dicts."""
    code = "result = df[['quantity', 'total_sales']].corr().to_dict()"
    mock_llm = _make_mock_llm(code)
    
    output = run_stats_query("What is the correlation between quantity and sales?", sample_df, mock_llm)
    
    assert output['success'] is True
    assert output['result'] is not None


# ---------------------------------------------------------------------------
# 4. Markdown fences are stripped from LLM output
# ---------------------------------------------------------------------------

def test_markdown_fenced_code_is_cleaned(sample_df):
    """LLM wraps code in ```python...``` — engine should strip it."""
    fenced_code = "```python\nresult = df['total_sales'].mean()\n```"
    mock_llm = _make_mock_llm(fenced_code)
    
    output = run_stats_query("Average sales?", sample_df, mock_llm)
    
    assert output['success'] is True
    # The stored code should NOT contain backticks
    assert '```' not in output['code']


# ---------------------------------------------------------------------------
# 5. LLM exception → clean error dict (no crash)
# ---------------------------------------------------------------------------

def test_llm_failure_returns_clean_error_dict(sample_df):
    """If LLM raises an exception, engine returns error dict — never crashes."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("Groq rate limit exceeded")
    
    output = run_stats_query("Forecast next month sales", sample_df, mock_llm)
    
    assert output['success'] is False
    assert output['error'] is not None
    assert 'LLM error' in output['error']
    assert output['result'] is None


# ---------------------------------------------------------------------------
# 6. Generated code missing 'result' variable → clean error
# ---------------------------------------------------------------------------

def test_code_without_result_variable_returns_error(sample_df):
    """LLM forgets to assign 'result' → clear error message, no crash."""
    bad_code = "x = df['total_sales'].sum()"  # Uses 'x' instead of 'result'
    mock_llm = _make_mock_llm(bad_code)
    
    output = run_stats_query("Sum sales", sample_df, mock_llm)
    
    assert output['success'] is False
    assert "result" in output['error'].lower()


# ---------------------------------------------------------------------------
# 7. Empty DataFrame
# ---------------------------------------------------------------------------

def test_empty_dataframe_returns_error():
    """Empty DataFrame → immediate error before LLM is even called."""
    mock_llm = MagicMock()
    empty_df = pd.DataFrame()
    
    output = run_stats_query("Analyze sales", empty_df, mock_llm)
    
    assert output['success'] is False
    assert output['error'] is not None
    mock_llm.invoke.assert_not_called()  # LLM should NOT be called


# ---------------------------------------------------------------------------
# 8. Empty question
# ---------------------------------------------------------------------------

def test_empty_question_returns_error(sample_df):
    """Empty question → immediate error, LLM not called."""
    mock_llm = MagicMock()
    
    output = run_stats_query("", sample_df, mock_llm)
    
    assert output['success'] is False
    mock_llm.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# 9. RestrictedPython blocks os import (security test)
# ---------------------------------------------------------------------------

def test_dangerous_os_import_is_blocked(sample_df):
    """RestrictedPython must block 'import os' — core security guarantee."""
    dangerous_code = "import os; result = os.listdir('.')"
    
    exec_result = execute_restricted(dangerous_code, sample_df)
    
    # RestrictedPython should block this — either compile error or exec error
    assert exec_result['success'] is False
    assert exec_result['error'] is not None


# ---------------------------------------------------------------------------
# 10. Result normalization — numpy types become Python types
# ---------------------------------------------------------------------------

def test_numpy_types_are_normalized():
    """numpy.float64 and numpy.int64 must be converted to plain Python types."""
    assert isinstance(_normalize_result(np.float64(42.5)), float)
    assert isinstance(_normalize_result(np.int64(7)), int)
    assert isinstance(_normalize_result(np.array([1, 2, 3])), list)


def test_pandas_series_is_normalized_to_list():
    """pandas.Series must be converted to a plain Python list."""
    series = pd.Series([10.0, 20.0, 30.0])
    result = _normalize_result(series)
    assert isinstance(result, list)
    assert result == [10.0, 20.0, 30.0]


# ---------------------------------------------------------------------------
# 11. Return dict always has correct keys
# ---------------------------------------------------------------------------

def test_success_result_has_all_expected_keys(sample_df):
    """Success dict must always have: success, code, result, error."""
    mock_llm = _make_mock_llm("result = df['total_sales'].max()")
    output = run_stats_query("Max sales?", sample_df, mock_llm)
    
    assert 'success' in output
    assert 'code' in output
    assert 'result' in output
    assert 'error' in output


def test_error_result_has_all_expected_keys(sample_df):
    """Error dict must have the same keys as success dict — no KeyError ever."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("API down")
    output = run_stats_query("Forecast sales", sample_df, mock_llm)
    
    assert 'success' in output
    assert 'code' in output
    assert 'result' in output
    assert 'error' in output