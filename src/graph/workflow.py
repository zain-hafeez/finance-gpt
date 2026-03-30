# src/graph/workflow.py
"""
Module 6 — LangGraph StateGraph Workflow

This file assembles the complete FinanceGPT pipeline.

Graph structure:
    START
      ↓
    router_node     (M5) — classifies "sql" or "stats"
      ↓ (conditional)
    ┌──────────────────┐
    sql_node (M3)    stats_node (M4)
    └──────────────────┘
      ↓ (both paths merge here)
    explain_node    (M6) — plain English explanation via Groq Mixtral
      ↓
    END

Public API (the only function you need from outside):
    run_query(question, file_path, db_path, session_id) -> dict

That's it. One function in, complete answer out.
"""

import logging
from langgraph.graph import StateGraph, START, END

from src.graph.state import FinanceGPTState
from src.graph.router import router_node, route_to_engine
from src.graph.nodes import sql_node, stats_node, explain_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build the Graph (compiled once at module load)
# ---------------------------------------------------------------------------

def _build_graph():
    """
    Construct and compile the LangGraph StateGraph.
    
    Called once when this module is first imported. The compiled graph
    is stored as a module-level variable and reused for every query.
    Compiling is slightly expensive — we don't want to do it per-request.
    
    Returns:
        A compiled LangGraph CompiledGraph ready to invoke.
    """
    # Step 1: Create the StateGraph, telling it what state schema to use
    # FinanceGPTState is our TypedDict from state.py (M5)
    graph = StateGraph(FinanceGPTState)

    # Step 2: Add all nodes
    # Each call registers a node with a name and the function to call
    graph.add_node("router",      router_node)   # M5 — classifies sql/stats
    graph.add_node("sql_node",    sql_node)      # M3 — SQL engine
    graph.add_node("stats_node",  stats_node)    # M4 — Stats engine
    graph.add_node("explain",     explain_node)  # M6 — plain English

    # Step 3: Add edges — define the flow

    # START → router (always — every query starts at the router)
    graph.add_edge(START, "router")

    # router → sql_node OR stats_node (conditional — depends on classification)
    graph.add_conditional_edges(
        "router",          # After this node runs...
        route_to_engine,   # ...call this function to decide the next node...
        {
            "sql":   "sql_node",    # ...if "sql" → go to sql_node
            "stats": "stats_node",  # ...if "stats" → go to stats_node
        }
    )

    # sql_node → explain (always — after SQL runs, explain the result)
    graph.add_edge("sql_node", "explain")

    # stats_node → explain (always — after stats runs, explain the result)
    graph.add_edge("stats_node", "explain")

    # explain → END (always — explanation is the last step)
    graph.add_edge("explain", END)

    # Step 4: Compile the graph
    # This validates the graph structure and prepares it for execution
    compiled = graph.compile()
    
    logger.info("FinanceGPT workflow graph compiled successfully")
    return compiled


# Compile the graph once at module load time
# All calls to run_query() reuse this compiled graph
_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_query(
    question: str,
    file_path: str,
    db_path: str,
    session_id: str = "default"
) -> dict:
    """
    Run a complete FinanceGPT query through the full LangGraph pipeline.
    
    This is the single public function of the entire M6 module.
    Gradio UI (M8) will call this function directly.
    
    What happens internally:
      1. Builds the initial FinanceGPTState with your inputs
      2. Runs the compiled LangGraph pipeline:
           router_node → sql_node or stats_node → explain_node
      3. Returns the final state as a clean result dict
    
    Args:
        question:   User's natural language question
                    e.g. "What is the total sales by region?"
        file_path:  Absolute path to the uploaded CSV or Excel file
                    e.g. "G:/Projects/finance_gpt/data/sample_sales.csv"
        db_path:    Absolute path to the SQLite database
                    e.g. "G:/Projects/finance_gpt/finance.db"
        session_id: Unique session identifier (default "default")
                    Used by LangGraph checkpointer in M9
    
    Returns:
        dict with keys:
            success      (bool)  — True if pipeline completed without error
            explanation  (str)   — Plain English answer for the user
            query_type   (str)   — "sql" or "stats"
            sql_query    (str|None) — Generated SQL (sql path only)
            stats_code   (str|None) — Generated Python code (stats path only)
            raw_result   (Any)   — The actual computed data
            error        (str|None) — Error message if something failed
    
    Never raises — all exceptions caught and returned in the result dict.
    """
    # Guard: validate inputs
    if not question or not question.strip():
        return _error_response("No question provided")

    if not file_path:
        return _error_response("No file path provided")

    if not db_path:
        return _error_response("No database path provided")

    # Build the initial state
    # Only set the fields we know at the start — LangGraph nodes fill the rest
    initial_state: FinanceGPTState = {
        # Input fields (provided by caller)
        "file_path":   file_path,
        "db_path":     db_path,
        "query":       question.strip(),
        "session_id":  session_id,

        # Routing (filled by router_node)
        "query_type":  "",
        "cached":      False,

        # SQL path (filled by sql_node if routed there)
        "sql_query":   None,
        "sql_valid":   False,

        # Stats path (filled by stats_node if routed there)
        "stats_code":  None,

        # Results (filled by engine nodes + explain_node)
        "raw_result":  None,
        "explanation": "",

        # Meta
        "error":       None,
    }

    logger.info("run_query: starting pipeline for question: '%s...'", question[:60])

    # Run the compiled graph
    try:
        final_state = _graph.invoke(initial_state)
    except Exception as e:
        logger.error("run_query: pipeline failed with exception: %s", e)
        return _error_response(f"Pipeline error: {str(e)}")

    # Shape the final response dict
    error = final_state.get("error")
    
    return {
        "success":     error is None,
        "explanation": final_state.get("explanation", "No explanation generated"),
        "query_type":  final_state.get("query_type", ""),
        "sql_query":   final_state.get("sql_query"),
        "stats_code":  final_state.get("stats_code"),
        "raw_result":  final_state.get("raw_result"),
        "error":       error,
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _error_response(error_message: str) -> dict:
    """
    Standardized error response. Same shape as a successful run_query() result.
    Callers can always safely access all keys without KeyError.
    """
    return {
        "success":     False,
        "explanation": f"Error: {error_message}",
        "query_type":  "",
        "sql_query":   None,
        "stats_code":  None,
        "raw_result":  None,
        "error":       error_message,
    }