# src/graph/nodes.py
"""
Module 6 — LangGraph Node Functions

Each function here is a LangGraph node. Nodes are the workers on the
assembly line. Each one:
  1. Receives the full FinanceGPTState dict
  2. Does one specific job
  3. Returns a PARTIAL state update (only the fields it changed)
  4. Never raises — all exceptions caught and written to state["error"]

Node inventory:
  sql_node     — calls M3 SQL engine, writes sql_query + raw_result
  stats_node   — calls M4 Stats engine, writes stats_code + raw_result
  explain_node — calls Groq Mixtral, writes explanation

The router_node lives in router.py (M5) — imported by workflow.py.

Design rule: Nodes are thin wrappers.
  The actual logic lives in the engines (sql_engine.py, stats_engine.py).
  Nodes just call those engines, handle errors, and shape the state update.
  This keeps nodes easy to test in isolation.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Explanation Prompt
# ---------------------------------------------------------------------------
# Used by explain_node to turn raw results into plain English.
# Mixtral-8x7b is specifically chosen here — it's excellent at summarization
# and explanation, better than Llama3 for this task.

EXPLANATION_PROMPT = """You are a friendly financial data analyst explaining results to a business user.

The user asked: {question}

The analysis returned this result:
{result}

{engine_context}

Write a clear, concise plain-English explanation of what this result means.
Focus on the business insight, not the technical details.
Keep it to 2-4 sentences maximum.
Do not mention SQL, DataFrames, Python, or technical terms.
If the result is a list or table, highlight the most important finding."""


# ---------------------------------------------------------------------------
# sql_node
# ---------------------------------------------------------------------------

def sql_node(state: dict) -> dict:
    """
    LangGraph node — runs the M3 SQL engine.
    
    Reads from state:  query, db_path
    Writes to state:   sql_query, sql_valid, raw_result, error
    
    Thin wrapper around run_sql_query() from M3.
    All SQL generation, validation, and execution happens there.
    """
    from src.engines.sql_engine import run_sql_query
    from src.utils.llm_router import get_llm

    question = state.get("query", "")
    db_path = state.get("db_path", "")

    # Guard: check we have what we need
    if not question:
        logger.error("sql_node: no query in state")
        return {"error": "sql_node: no question provided", "sql_valid": False}

    if not db_path:
        logger.error("sql_node: no db_path in state")
        return {"error": "sql_node: no database path provided", "sql_valid": False}

    logger.info("sql_node: processing question: '%s...'", question[:60])

    # Get LLM fresh — never stored in state
    try:
        llm = get_llm('sql')
    except Exception as e:
        logger.error("sql_node: LLM init failed: %s", e)
        return {
            "error": f"Could not initialize LLM: {str(e)}",
            "sql_valid": False,
            "sql_query": None,
            "raw_result": None,
        }

    # Call M3 SQL engine
    result = run_sql_query(
        question=question,
        db_path=db_path,
        llm=llm
    )

    # Shape the state update
    if result["success"]:
        logger.info("sql_node: success — sql: %s", result.get("sql", "")[:80])
        return {
            "sql_query": result.get("sql"),
            "sql_valid": True,
            "raw_result": result.get("result"),
            "error": None,
        }
    else:
        logger.error("sql_node: failed — %s", result.get("error"))
        return {
            "sql_query": result.get("sql"),
            "sql_valid": False,
            "raw_result": None,
            "error": result.get("error"),
        }


# ---------------------------------------------------------------------------
# stats_node
# ---------------------------------------------------------------------------

def stats_node(state: dict) -> dict:
    """
    LangGraph node — runs the M4 Stats engine.
    
    Reads from state:  query, file_path
    Writes to state:   stats_code, raw_result, error
    
    Note: Stats engine needs the Pandas DataFrame, not the SQLite db.
    We reload the DataFrame from file_path inside this node.
    We do NOT store DataFrames in state (they're not serializable).
    """
    from src.engines.stats_engine import run_stats_query
    from src.utils.llm_router import get_llm

    question = state.get("query", "")
    file_path = state.get("file_path", "")

    # Guard: check inputs
    if not question:
        logger.error("stats_node: no query in state")
        return {"error": "stats_node: no question provided", "stats_code": None, "raw_result": None}

    if not file_path:
        logger.error("stats_node: no file_path in state")
        return {"error": "stats_node: no file path provided", "stats_code": None, "raw_result": None}

    logger.info("stats_node: processing question: '%s...'", question[:60])

    # Reload DataFrame from file — never store df in state
    try:
        if file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
        logger.info("stats_node: loaded %d rows from %s", len(df), file_path)
    except Exception as e:
        logger.error("stats_node: could not load file: %s", e)
        return {
            "error": f"Could not load data file: {str(e)}",
            "stats_code": None,
            "raw_result": None,
        }

    # Get LLM fresh
    try:
        llm = get_llm('stats')
    except Exception as e:
        logger.error("stats_node: LLM init failed: %s", e)
        return {
            "error": f"Could not initialize LLM: {str(e)}",
            "stats_code": None,
            "raw_result": None,
        }

    # Call M4 Stats engine
    result = run_stats_query(
        question=question,
        df=df,
        llm=llm
    )

    # Shape the state update
    if result["success"]:
        logger.info("stats_node: success")
        return {
            "stats_code": result.get("code"),
            "raw_result": result.get("result"),
            "error": None,
        }
    else:
        logger.error("stats_node: failed — %s", result.get("error"))
        return {
            "stats_code": result.get("code"),
            "raw_result": None,
            "error": result.get("error"),
        }


# ---------------------------------------------------------------------------
# explain_node
# ---------------------------------------------------------------------------

def explain_node(state: dict) -> dict:
    """
    LangGraph node — generates a plain English explanation of the result.
    
    Reads from state:  query, raw_result, query_type, sql_query, stats_code, error
    Writes to state:   explanation
    
    Uses Groq Mixtral-8x7b — specifically chosen for its excellent
    summarization and explanation capabilities.
    
    If there's already an error in state (SQL or stats node failed),
    this node writes a user-friendly error message as the explanation
    instead of calling the LLM — no point explaining a failed result.
    """
    from src.utils.llm_router import get_llm
    from langchain_core.messages import HumanMessage

    question = state.get("query", "")
    raw_result = state.get("raw_result")
    query_type = state.get("query_type", "sql")
    error = state.get("error")

    # If a previous node failed, write a friendly error explanation
    if error:
        logger.warning("explain_node: upstream error detected — writing error explanation")
        return {
            "explanation": (
                f"I wasn't able to answer your question due to an error: {error}. "
                "Please try rephrasing your question or check that your data file "
                "contains the columns you're asking about."
            )
        }

    # If no result was produced
    if raw_result is None:
        return {
            "explanation": (
                "The query ran successfully but returned no results. "
                "This might mean no data matches your criteria."
            )
        }

    # Build context string to help the LLM understand what engine was used
    if query_type == "sql":
        sql_query = state.get("sql_query", "")
        engine_context = f"This was computed using the SQL query: {sql_query}" if sql_query else ""
    else:
        engine_context = "This was computed using statistical analysis on the data."

    # Format raw_result as a clean string for the prompt
    result_str = _format_result_for_prompt(raw_result)

    # Build the explanation prompt
    prompt = EXPLANATION_PROMPT.format(
        question=question,
        result=result_str,
        engine_context=engine_context
    )

    logger.info("explain_node: generating explanation for result: '%s...'", result_str[:80])

    # Call Groq Mixtral — best for explanations
    try:
        llm = get_llm('explanation')
        response = llm.invoke([HumanMessage(content=prompt)])
        explanation = response.content.strip()
        logger.info("explain_node: success")
        return {"explanation": explanation}
    except Exception as e:
        logger.error("explain_node: LLM failed: %s", e)
        # Fallback — return the raw result as the explanation
        return {
            "explanation": f"Result: {result_str}"
        }


# ---------------------------------------------------------------------------
# Helper — format result for explanation prompt
# ---------------------------------------------------------------------------

def _format_result_for_prompt(raw_result) -> str:
    """
    Convert raw_result to a clean, readable string for the LLM prompt.
    
    raw_result can be many types:
      - str (from SQL engine — e.g. "[('North', 955.0), ('South', 1315.0)]")
      - list (from stats engine — e.g. [102.5, 115.3, 98.7])
      - dict (from stats engine — correlation matrix etc.)
      - float/int (single computed value)
      - None (handled before calling this)
    """
    if raw_result is None:
        return "No result"

    if isinstance(raw_result, str):
        # SQL results come back as strings — limit length for prompt
        return raw_result[:500] if len(raw_result) > 500 else raw_result

    if isinstance(raw_result, (int, float)):
        return str(raw_result)

    if isinstance(raw_result, list):
        # Show first 20 items max to avoid bloating the prompt
        preview = raw_result[:20]
        result_str = str(preview)
        if len(raw_result) > 20:
            result_str += f" ... (and {len(raw_result) - 20} more values)"
        return result_str

    if isinstance(raw_result, dict):
        # Dict — show as formatted string, limit size
        result_str = str(raw_result)
        return result_str[:500] if len(result_str) > 500 else result_str

    # Fallback for any other type
    return str(raw_result)[:500]