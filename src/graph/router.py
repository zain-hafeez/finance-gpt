# src/graph/router.py
"""
Module 5 — Query Router

Classifies a user's natural language question as either:
  "sql"   → exact aggregation (total, filter, group, sort, count, top-N)
  "stats" → statistical analysis (forecast, trend, correlation, anomaly, moving avg)

Architecture:
    User question
        ↓
    Groq llama3-70b (fast, ~200ms)
        ↓
    Parse response → "sql" or "stats"
        ↓
    Write to state["query_type"]
        ↓
    LangGraph conditional edge routes to M3 or M4

Why LLM classification instead of keyword matching?
    Keyword matching is brittle:
      "What is the trend in sales?" → has "trend" → might match stats ✅
      "What is the total trend over regions?" → has "trend" → wrong match ❌
    LLM classification understands intent, not just words.
    With a clear prompt + few-shot examples, accuracy is ~99%.

Design decisions:
    - temperature=0: deterministic, no randomness in classification
    - Response parsing: strip + lowercase + check startswith("sql"/"stats")
    - Fallback: if LLM returns something unexpected → default to "sql"
      (SQL is safer — it either works or gives a clear error)
    - Error handling: LLM failure → set state["error"], query_type="sql"
      (allows pipeline to continue and fail gracefully downstream)
"""

import logging
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification Prompt
# ---------------------------------------------------------------------------
# Design principles:
#   1. Give the LLM exactly 2 choices — "sql" or "stats". No wiggle room.
#   2. Define each category with concrete examples, not abstract descriptions.
#   3. Few-shot examples teach the LLM the boundary cases.
#   4. Repeat the output format instruction at the end — LLMs follow recency.
#   5. Capital "ONLY" enforces single-word response.

CLASSIFICATION_PROMPT = """You are a query classifier for a financial data analysis system.

Classify the user's question into EXACTLY ONE of these two categories:

sql — Use for questions that need exact calculations on the data:
  - Totals, sums, counts, averages of columns
  - Filtering rows (e.g. "show me sales > 100")
  - Grouping and aggregating (e.g. "sales by region")
  - Sorting and ranking (e.g. "top 5 products")
  - Finding min/max values
  - Counting distinct values

stats — Use for questions that need statistical analysis or prediction:
  - Forecasting or predicting future values
  - Trend analysis over time
  - Moving averages or rolling windows
  - Correlation between two columns
  - Anomaly or outlier detection
  - Percentage change or growth rates
  - Standard deviation or variance

Examples:
  "What is the total sales amount?" → sql
  "Show me sales by region" → sql
  "Which product had the highest revenue?" → sql
  "Filter rows where quantity > 10" → sql
  "What is the 3-month moving average of sales?" → stats
  "Is there a correlation between price and quantity?" → stats
  "Forecast next month's sales" → stats
  "Show me outliers in the revenue column" → stats
  "What is the trend in total sales over time?" → stats
  "How many orders were placed in January?" → sql

USER QUESTION: {question}

Respond with ONLY the single word: sql or stats"""


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------

def classify_query(question: str, llm) -> str:
    """
    Classify a user question as "sql" or "stats" using the LLM.
    
    This is the core M5 function. It's called by the LangGraph router node
    in workflow.py (M6) to determine which engine processes the query.
    
    Args:
        question: The user's natural language question
        llm:      LangChain LLM instance from get_llm('classification')
                  Should be llama3-70b at temperature=0 for determinism
    
    Returns:
        "sql" or "stats" — always one of these two strings, never raises.
        Defaults to "sql" on any ambiguity or LLM failure.
    
    Design: This function NEVER raises an exception. All errors are caught
    and logged. The caller can always safely use the return value.
    """
    # Guard: empty question
    if not question or not question.strip():
        logger.warning("classify_query called with empty question — defaulting to sql")
        return "sql"

    # Build the prompt
    prompt = CLASSIFICATION_PROMPT.format(question=question.strip())

    # Call the LLM
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_response = response.content
    except Exception as e:
        logger.error("LLM classification failed: %s — defaulting to sql", e)
        return "sql"  # Safe default

    # Parse the response
    query_type = _parse_classification(raw_response, question)
    
    logger.info(
        "classify_query: '%s...' → '%s' (raw LLM response: '%s')",
        question[:50],
        query_type,
        raw_response.strip()[:30]
    )
    
    return query_type


def _parse_classification(raw_response: str, original_question: str) -> str:
    """
    Parse the LLM's response into "sql" or "stats".
    
    The LLM should return just "sql" or "stats", but LLMs sometimes add
    punctuation, extra words, or explanation. This function handles all cases.
    
    Parsing strategy:
        1. Strip whitespace + lowercase
        2. Check if it starts with "sql" or "stats"
        3. If neither → log a warning + default to "sql"
    
    Args:
        raw_response:      Raw string from LLM .content
        original_question: Used only for warning log context
    
    Returns:
        "sql" or "stats"
    """
    if not raw_response:
        logger.warning("Empty LLM response for question: '%s...' — defaulting to sql",
                       original_question[:40])
        return "sql"

    cleaned = raw_response.strip().lower()

    # Check for "stats" first because "stats" contains "s" which could false-match
    if cleaned.startswith("stats"):
        return "stats"
    
    if cleaned.startswith("sql"):
        return "sql"

    # Neither matched — LLM gave an unexpected response
    # Log the full response for debugging, then default safely
    logger.warning(
        "Unexpected classification response: '%s' for question: '%s...' — defaulting to sql",
        raw_response.strip()[:50],
        original_question[:40]
    )
    return "sql"


# ---------------------------------------------------------------------------
# LangGraph Node Function
# ---------------------------------------------------------------------------
# This is the function that LangGraph calls directly as a node.
# It takes the full state dict, does its work, and returns a PARTIAL state
# update — only the fields it changed. LangGraph merges it automatically.

def router_node(state: dict) -> dict:
    """
    LangGraph node function for M5 query classification.
    
    LangGraph calls this with the full current state dict.
    This function reads what it needs, does its work, and returns
    only the fields it wants to update.
    
    Called by: LangGraph StateGraph in workflow.py (M6)
    Reads from state: query
    Writes to state:  query_type, error (on failure)
    
    Args:
        state: The full FinanceGPTState dict
    
    Returns:
        Partial state update dict — LangGraph merges this into the full state
    """
    from src.utils.llm_router import get_llm

    question = state.get("query", "")
    
    if not question:
        logger.error("router_node: no query in state")
        return {
            "query_type": "sql",  # Safe default
            "error": "No question provided"
        }

    # Get LLM — always fresh, never stored in state
    try:
        llm = get_llm('classification')
    except Exception as e:
        logger.error("router_node: could not get LLM: %s", e)
        return {
            "query_type": "sql",
            "error": f"LLM initialization failed: {str(e)}"
        }

    # Classify the query
    query_type = classify_query(question, llm)

    return {
        "query_type": query_type,
        "error": None  # Clear any previous error
    }


# ---------------------------------------------------------------------------
# Conditional Edge Function (used by LangGraph to route between nodes)
# ---------------------------------------------------------------------------
# In LangGraph, a "conditional edge" is a function that takes state and
# returns a string. That string is the name of the next node to execute.
# This is how the "if sql → go to sql_node, else → go to stats_node" works.

def route_to_engine(state: dict) -> str:
    """
    LangGraph conditional edge function.
    
    Called AFTER router_node completes. Reads query_type from state
    and returns the name of the next node to execute.
    
    This is NOT a node — it doesn't process data.
    It's purely a routing decision function used in workflow.py like:
    
        graph.add_conditional_edges(
            "router",
            route_to_engine,           ← this function
            {
                "sql":   "sql_node",   ← if returns "sql" → go to sql_node
                "stats": "stats_node", ← if returns "stats" → go to stats_node
            }
        )
    
    Args:
        state: Full FinanceGPTState dict (after router_node has run)
    
    Returns:
        "sql" or "stats" — must match the keys in add_conditional_edges map
    """
    query_type = state.get("query_type", "sql")
    
    # If there's already an error in state, still route somewhere
    # The downstream node will handle the error gracefully
    if query_type not in ("sql", "stats"):
        logger.warning("route_to_engine: invalid query_type '%s' — routing to sql", query_type)
        return "sql"
    
    return query_type