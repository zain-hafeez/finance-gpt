# tests/test_ui.py
"""
Module 8 — UI Component Tests
================================
12 tests covering src/ui/components.py pure functions.
No Gradio server required — these are plain function tests.
"""

import pytest
import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Group 1: build_chart (4 tests)
# ---------------------------------------------------------------------------

class TestBuildChart:

    def test_two_column_sql_result_returns_bar_chart(self):
        """[('East', 705.0), ...] should produce a Bar chart."""
        from src.ui.components import build_chart
        raw = [('East', 705.0), ('North', 955.0), ('South', 1315.0)]
        fig = build_chart(raw, 'sql', 'Sales by region')
        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Bar)

    def test_single_value_sql_result_returns_none(self):
        """[(4065.0,)] is a scalar — no chart needed."""
        from src.ui.components import build_chart
        raw = [(4065.0,)]
        fig = build_chart(raw, 'sql', 'Total sales')
        assert fig is None

    def test_list_of_numbers_returns_line_chart(self):
        """A flat list of floats (moving average) should produce a Scatter chart."""
        from src.ui.components import build_chart
        raw = [100.0, 120.0, 115.0, 130.0, 125.0]
        fig = build_chart(raw, 'stats', '3-period moving average')
        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Scatter)

    def test_none_result_returns_none(self):
        """None input should return None without crashing."""
        from src.ui.components import build_chart
        assert build_chart(None, 'sql', 'question') is None


# ---------------------------------------------------------------------------
# Group 2: format_table (4 tests)
# ---------------------------------------------------------------------------

class TestFormatTable:

    def test_two_column_result_returns_dataframe(self):
        """[('East', 705.0), ...] should return a DataFrame with 2 columns."""
        from src.ui.components import format_table
        raw = [('East', 705.0), ('North', 955.0)]
        df = format_table(raw)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_single_value_result_returns_none(self):
        """[(4065.0,)] is scalar — no table needed."""
        from src.ui.components import format_table
        raw = [(4065.0,)]
        result = format_table(raw)
        assert result is None

    def test_dict_result_returns_dataframe(self):
        """{'total_sales': 0.85, 'quantity': 0.85} → 2-row DataFrame."""
        from src.ui.components import format_table
        raw = {'total_sales': 0.85, 'quantity': 0.85}
        df = format_table(raw)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['Metric', 'Value']

    def test_none_result_returns_none(self):
        """None input should return None without crashing."""
        from src.ui.components import format_table
        assert format_table(None) is None


# ---------------------------------------------------------------------------
# Group 3: format_status (2 tests)
# ---------------------------------------------------------------------------

class TestFormatStatus:

    def test_successful_sql_cached_result(self):
        """Cached SQL result should show SQL engine + cached indicator."""
        from src.ui.components import format_status
        result = {'success': True, 'query_type': 'sql', 'cached': True}
        status = format_status(result, 0.003)
        assert '✅' in status
        assert 'SQL' in status
        assert 'Cached' in status

    def test_error_result_shows_error_indicator(self):
        """Failed pipeline result should show error indicator."""
        from src.ui.components import format_status
        result = {'success': False}
        status = format_status(result, 3.1)
        assert '❌' in status


# ---------------------------------------------------------------------------
# Group 4: build_chat_message (2 tests)
# ---------------------------------------------------------------------------

class TestBuildChatMessage:

    def test_successful_result_returns_explanation(self):
        """Successful result should return the explanation text."""
        from src.ui.components import build_chat_message
        result = {
            'success': True,
            'explanation': 'Total sales is $4,065.',
            'raw_result': [(4065.0,)],
        }
        msg = build_chat_message(result)
        assert 'Total sales is $4,065.' in msg

    def test_error_result_returns_error_message(self):
        """Failed result should return a user-friendly error message."""
        from src.ui.components import build_chat_message
        result = {'success': False, 'error': 'LLM call failed'}
        msg = build_chat_message(result)
        assert '❌' in msg
        assert 'LLM call failed' in msg