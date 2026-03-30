# src/engines/stats_engine.py
"""
Module 4 — Stats Engine

Handles statistical questions that SQL cannot answer:
- Forecasting / trend prediction
- Moving averages (rolling windows)
- Correlation between columns
- Anomaly / outlier detection
- Percentage change over time

Architecture:
    User question + DataFrame
        ↓
    Build prompt with schema (column names + dtypes)
        ↓
    LLM generates Pandas/NumPy code
        ↓
    Clean LLM output (strip markdown fences)
        ↓
    execute_restricted() — RestrictedPython sandbox
        ↓
    Return standardized result dict

Return shape (always consistent, never raises):
    {
        'success': bool,
        'code':    str   — generated code (for UI transparency),
        'result':  Any   — the computed value,
        'error':   str | None
    }
"""

import logging
from langchain_core.messages import HumanMessage

from src.security.restricted_exec import execute_restricted

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------
# This is the instruction we send to the LLM.
# Key design decisions:
#   1. We inject the actual column names + dtypes so the LLM knows the schema
#   2. We tell it the DataFrame is always called 'df'
#   3. We demand the result goes into a variable called 'result'
#   4. We explicitly forbid os/sys/subprocess — belt AND suspenders
#      (RestrictedPython blocks them anyway, but the prompt reinforces it)
#   5. We list the exact allowed imports (LLM shouldn't need to import anything
#      because we pre-inject pd, np, scipy_stats into the sandbox globals)

STATS_CODE_PROMPT = """You are a Python data analyst expert.

You have a Pandas DataFrame called `df` with the following schema:
{schema}

Sample data (first 3 rows):
{sample_rows}

TASK: Write Python code to answer this question:
{question}

STRICT RULES — violating any rule causes an error:
1. Write ONLY executable Python code — no explanations, no markdown, no comments
2. No code fences (no ```python or ```)
3. The DataFrame is already loaded as `df` — do NOT read any files
4. You have access to: df, pd (pandas), np (numpy), scipy_stats, math, datetime
5. Store your FINAL answer in a variable called `result`
6. `result` must be one of: a number, a list, a dict, or a Pandas Series
7. Do NOT use: import, os, sys, subprocess, open, eval, exec, print
8. If forecasting: use np.polyfit or scipy_stats.linregress
9. For moving averages: use df[column].rolling(window).mean()
10. For correlation: use df[[col1, col2]].corr()

Example of correct output format:
result = df['total_sales'].rolling(3).mean().dropna().tolist()

PYTHON CODE:"""


# ---------------------------------------------------------------------------
# Helper: Build schema string for the prompt
# ---------------------------------------------------------------------------

def _build_schema_string(df) -> str:
    """
    Convert DataFrame dtypes into a readable schema string for the LLM prompt.
    
    Example output:
        date          object
        region        object  
        product       object
        quantity      int64
        unit_price    float64
        total_sales   float64
        salesperson   object
    """
    lines = []
    for col, dtype in df.dtypes.items():
        lines.append(f"  {col:<20} {dtype}")
    return "\n".join(lines)


def _build_sample_rows(df) -> str:
    """
    Show the LLM the first 3 rows so it understands the data format.
    Limits output to avoid bloating the prompt.
    """
    try:
        return df.head(3).to_string(index=False)
    except Exception:
        return "(could not render sample rows)"


# ---------------------------------------------------------------------------
# Helper: Clean LLM output
# ---------------------------------------------------------------------------

def _clean_code_output(raw: str) -> str:
    """
    Strip markdown fences and preamble text from LLM output.
    
    LLMs often respond with:
```python
        result = df['sales'].sum()
```
    
    We need just:
        result = df['sales'].sum()
    
    Strategy:
        1. Remove ```python and ``` fences
        2. Find the first line that looks like real Python code
        3. Return everything from that line onward
    """
    if not raw:
        return ""
    
    # Remove markdown code fences
    cleaned = raw.strip()
    cleaned = cleaned.replace("```python", "").replace("```Python", "")
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.strip()
    
    # Find the first line that starts with a Python keyword or assignment
    # This removes any preamble like "Here is the code:"
    python_starters = (
        'result', 'df', 'import', 'pd', 'np', 'scipy',
        '#', 'x ', 'y ', 'data', 'values', 'mean', 'corr',
        'rolling', 'window', 'forecast', 'trend'
    )
    
    lines = cleaned.split('\n')
    start_index = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and any(stripped.startswith(s) for s in python_starters):
            start_index = i
            break
    
    return "\n".join(lines[start_index:]).strip()


# ---------------------------------------------------------------------------
# Main Public Function
# ---------------------------------------------------------------------------

def run_stats_query(question: str, df, llm) -> dict:
    """
    Execute a statistical analysis query using LLM-generated Pandas/NumPy code.
    
    Args:
        question: User's natural language question (e.g. "What's the 3-month moving average?")
        df:       Pandas DataFrame — the user's uploaded financial data
        llm:      LangChain LLM instance from get_llm('stats')
    
    Returns:
        dict with keys:
            success  (bool)
            code     (str)   — the generated Python code (shown in UI for transparency)
            result   (Any)   — the computed answer
            error    (str | None)
    
    Never raises — all exceptions are caught and returned in the error key.
    """
    
    # --- Guard: validate inputs ---
    if not question or not question.strip():
        return _error_result("", "Question cannot be empty")
    
    if df is None or len(df) == 0:
        return _error_result("", "DataFrame is empty — no data to analyze")

    # --- Step 1: Build schema + sample for the prompt ---
    schema_str = _build_schema_string(df)
    sample_str = _build_sample_rows(df)
    
    # --- Step 2: Build the full prompt ---
    prompt = STATS_CODE_PROMPT.format(
        schema=schema_str,
        sample_rows=sample_str,
        question=question
    )
    
    logger.info("Stats engine: generating code for question: %s", question[:80])
    
    # --- Step 3: Call the LLM ---
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_code = response.content
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return _error_result("", f"LLM error: {str(e)}")
    
    # --- Step 4: Clean the LLM output ---
    code = _clean_code_output(raw_code)
    
    if not code:
        return _error_result(raw_code, "LLM returned empty code")
    
    logger.info("Stats engine: generated code:\n%s", code)
    
    # --- Step 5: Execute in RestrictedPython sandbox ---
    exec_result = execute_restricted(code=code, df=df)
    
    if not exec_result['success']:
        return _error_result(code, exec_result['error'])
    
    # --- Step 6: Normalize the result for downstream use ---
    # Convert numpy types to plain Python types for JSON serialization
    raw_result = exec_result['result']
    normalized_result = _normalize_result(raw_result)
    
    logger.info("Stats engine: success — result type: %s", type(normalized_result).__name__)
    
    return {
        'success': True,
        'code': code,
        'result': normalized_result,
        'error': None
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_result(result):
    """
    Convert NumPy/Pandas types to plain Python types.
    
    Why: numpy.float64 and pandas.Series are not JSON-serializable.
    LangGraph state must contain only serializable types.
    
    Examples:
        np.float64(42.5)     → 42.5  (Python float)
        np.int64(7)          → 7     (Python int)
        pd.Series([1,2,3])   → [1, 2, 3]  (Python list)
        np.array([1,2,3])    → [1, 2, 3]  (Python list)
    """
    import numpy as np
    
    try:
        import pandas as pd
        if isinstance(result, pd.Series):
            return result.tolist()
        if isinstance(result, pd.DataFrame):
            return result.to_dict(orient='records')
    except Exception:
        pass
    
    if isinstance(result, np.ndarray):
        return result.tolist()
    if isinstance(result, (np.integer,)):
        return int(result)
    if isinstance(result, (np.floating,)):
        return float(result)
    if isinstance(result, np.bool_):
        return bool(result)
    
    return result


def _error_result(code: str, error_message: str) -> dict:
    """
    Standardized error response. Always the same shape as a success response.
    Callers can always safely access ['success'], ['code'], ['result'], ['error'].
    """
    logger.error("Stats engine error: %s", error_message)
    return {
        'success': False,
        'code': code,
        'result': None,
        'error': error_message
    }