# src/ui/components.py
"""
UI Component Helpers — FinanceGPT
===================================
Pure functions that transform pipeline results into displayable formats.
Kept separate from app.py so they can be unit-tested without Gradio.

Functions:
  build_chart(raw_result, query_type, question) -> go.Figure | None
  format_table(raw_result)                      -> pd.DataFrame | None
  format_status(result_dict, elapsed)           -> str
  format_sql_display(result_dict)               -> str
  build_chat_message(result_dict)               -> str
"""

import logging
import pandas as pd
import plotly.graph_objects as go
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chart Builder
# ---------------------------------------------------------------------------

def build_chart(raw_result: Any, query_type: str, question: str) -> go.Figure | None:
    """
    Build a Plotly chart from the pipeline's raw_result.

    Returns a go.Figure if the data is chartable, None otherwise.
    Gradio's Plot component accepts go.Figure directly.

    Chart type selection logic:
    - 2-column result with labels + numbers → Bar chart (e.g. sales by region)
    - Single number → no chart (just show the number in text)
    - List of numbers → Line chart (e.g. moving average)
    - Stats result that's already a dict → try bar chart
    """
    if raw_result is None:
        return None

    try:
        # --- Case 1: List of tuples from SQL (most common) ---
        # e.g. [('East', 705.0), ('North', 955.0), ...]
        if isinstance(raw_result, list) and len(raw_result) > 0:
            first = raw_result[0]

            # Single value result: [(4065.0,)] — no chart needed
            if isinstance(first, tuple) and len(first) == 1:
                return None

            # Two-column result: [('label', value), ...] → bar chart
            if isinstance(first, tuple) and len(first) == 2:
                labels = [str(row[0]) for row in raw_result]
                values = [float(row[1]) for row in raw_result]
                fig = go.Figure(go.Bar(
                    x=labels,
                    y=values,
                    marker_color='#4F8EF7',
                    text=[f'{v:,.2f}' for v in values],
                    textposition='auto',
                ))
                fig.update_layout(
                    title=_truncate(question, 60),
                    xaxis_title='Category',
                    yaxis_title='Value',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family='Inter, sans-serif', size=13),
                    margin=dict(l=40, r=40, t=60, b=40),
                )
                return fig

            # List of numbers (stats result like moving average) → line chart
            if isinstance(first, (int, float)):
                fig = go.Figure(go.Scatter(
                    y=raw_result,
                    mode='lines+markers',
                    line=dict(color='#4F8EF7', width=2),
                    marker=dict(size=6),
                ))
                fig.update_layout(
                    title=_truncate(question, 60),
                    xaxis_title='Period',
                    yaxis_title='Value',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family='Inter, sans-serif', size=13),
                    margin=dict(l=40, r=40, t=60, b=40),
                )
                return fig

        # --- Case 2: Dict result from stats engine ---
        # e.g. {'total_sales': 0.85, 'quantity': 0.85}
        if isinstance(raw_result, dict) and len(raw_result) > 0:
            labels = list(raw_result.keys())
            values = [float(v) for v in raw_result.values()]
            fig = go.Figure(go.Bar(
                x=labels,
                y=values,
                marker_color='#4F8EF7',
            ))
            fig.update_layout(
                title=_truncate(question, 60),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Inter, sans-serif', size=13),
                margin=dict(l=40, r=40, t=60, b=40),
            )
            return fig

        # --- Case 3: Single scalar (float/int) from stats ---
        # e.g. 0.8477 (correlation coefficient) — no chart
        return None

    except Exception as e:
        logger.warning("[chart] Could not build chart: %s", e)
        return None


# ---------------------------------------------------------------------------
# Table Formatter
# ---------------------------------------------------------------------------

def format_table(raw_result: Any) -> pd.DataFrame | None:
    """
    Convert raw_result into a pandas DataFrame for Gradio's Dataframe component.

    Returns None if the result is a single scalar (nothing to tabulate).
    """
    if raw_result is None:
        return None

    try:
        # List of tuples from SQL engine
        if isinstance(raw_result, list) and len(raw_result) > 0:
            first = raw_result[0]

            # Single value — not worth a table
            if isinstance(first, tuple) and len(first) == 1:
                return None

            # Multi-column tuples → DataFrame
            if isinstance(first, tuple):
                col_count = len(first)
                columns = [f'Column {i+1}' for i in range(col_count)]
                return pd.DataFrame(raw_result, columns=columns)

            # Flat list of numbers (e.g. moving average)
            return pd.DataFrame(raw_result, columns=['Value'])

        # Dict result from stats engine
        if isinstance(raw_result, dict):
            return pd.DataFrame(
                list(raw_result.items()),
                columns=['Metric', 'Value']
            )

        # Single scalar — not tabular
        return None

    except Exception as e:
        logger.warning("[table] Could not format table: %s", e)
        return None


# ---------------------------------------------------------------------------
# Status Formatter
# ---------------------------------------------------------------------------

def format_status(result: dict, elapsed: float) -> str:
    """
    Build a one-line status string shown below the chat.

    Example outputs:
      "✅  SQL engine  |  ⚡ Cached  |  0.003s"
      "✅  Stats engine  |  🔄 Fresh query  |  4.21s"
      "❌  Error  |  3.10s"
    """
    if result.get('error'):
        return f"❌  Error  |  {elapsed:.2f}s"

    engine = '📊 SQL engine' if result.get('query_type') == 'sql' else '📈 Stats engine'
    cache_str = '⚡ Cached' if result.get('cached') else '🔄 Fresh query'
    return f"✅  {engine}  |  {cache_str}  |  {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# SQL / Code Display Formatter
# ---------------------------------------------------------------------------

def format_sql_display(result: dict) -> str:
    """
    Return the generated SQL or stats code for display in the UI.
    Shown in a code block so users can see what was run.
    """
    if result.get('sql_query'):
        return f"```sql\n{result['sql_query']}\n```"
    if result.get('stats_code'):
        return f"```python\n{result['stats_code']}\n```"
    return '_No query generated_'


# ---------------------------------------------------------------------------
# Chat Message Builder
# ---------------------------------------------------------------------------

def build_chat_message(result: dict) -> str:
    """
    Build the assistant's chat message from a pipeline result.

    This is what appears in the Gradio Chatbot component as the
    bot's reply. It combines the explanation with any scalar result.
    """
    if result.get('error'):
        error = result.get('error', 'Unknown error')
        return f"❌ I wasn't able to answer that question.\n\n**Error:** {error}"

    explanation = result.get('explanation', '').strip()
    raw = result.get('raw_result')

    # If raw result is a single scalar, include it prominently
    if isinstance(raw, (int, float)):
        return f"**{raw:,.4g}**\n\n{explanation}"

    # Single-value tuple from SQL: [(4065.0,)]
    if isinstance(raw, list) and len(raw) == 1:
        if isinstance(raw[0], tuple) and len(raw[0]) == 1:
            val = raw[0][0]
            if isinstance(val, (int, float)):
                return f"**{val:,.4g}**\n\n{explanation}"

    return explanation


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int) -> str:
    """Truncate text for chart titles."""
    return text if len(text) <= max_len else text[:max_len - 3] + '...'