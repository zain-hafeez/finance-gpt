# src/graph/workflow.py
"""
LangGraph StateGraph — Module 6 + M7 (cache nodes) + M9 (checkpointer)

Assembles the full pipeline:
  START → cache_node → router_node → [sql_node | stats_node]
        → explain_node → cache_write_node → END

M9 adds: SqliteSaver checkpointer (persists state per thread_id/session_id)
"""

import logging
from langgraph.graph import StateGraph, END

from src.graph.state import FinanceGPTState
from src.graph.nodes import (
    cache_node,
    sql_node,
    stats_node,
    explain_node,
    cache_write_node,
)
from src.graph.router import router_node, route_to_engine  # ← router_node lives HERE, not in nodes.py
from src.graph.checkpointer import get_checkpointer

logger = logging.getLogger(__name__)

_graph = None


def _should_skip_to_end(state: dict) -> str:
    if state.get("cached", False):
        return "end"
    return "router"
# Alias for backward compatibility with test_cache.py
_route_after_cache = _should_skip_to_end

def _build_graph():
    graph = StateGraph(FinanceGPTState)

    graph.add_node("cache_node", cache_node)
    graph.add_node("router_node", router_node)
    graph.add_node("sql_node", sql_node)
    graph.add_node("stats_node", stats_node)
    graph.add_node("explain_node", explain_node)
    graph.add_node("cache_write_node", cache_write_node)

    graph.set_entry_point("cache_node")

    graph.add_conditional_edges(
        "cache_node",
        _should_skip_to_end,
        {
            "end": END,
            "router": "router_node",
        },
    )

    graph.add_conditional_edges(
        "router_node",
        route_to_engine,
        {
            "sql": "sql_node",
            "stats": "stats_node",
        },
    )

    graph.add_edge("sql_node", "explain_node")
    graph.add_edge("stats_node", "explain_node")
    graph.add_edge("explain_node", "cache_write_node")
    graph.add_edge("cache_write_node", END)

    checkpointer = get_checkpointer()
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("[workflow] Graph compiled with checkpointer: %s", type(checkpointer).__name__)
    return compiled


def get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


def run_query(
    question: str,
    file_path: str,
    db_path: str,
    session_id: str = "default"
) -> dict:
    graph = get_graph()

    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "file_path": file_path,
        "db_path": db_path,
        "query": question,
        "session_id": session_id,
        "query_type": "",
        "cached": False,
        "sql_query": None,
        "sql_valid": False,
        "stats_code": None,
        "raw_result": None,
        "explanation": "",
        "error": None,
    }

    try:
        final_state = graph.invoke(initial_state, config=config)
        logger.info(
            "[workflow] Query complete | session=%s | type=%s | cached=%s",
            session_id,
            final_state.get("query_type", "?"),
            final_state.get("cached", False),
        )
        return {
            "query_type":  final_state.get("query_type", ""),
            "sql_query":   final_state.get("sql_query"),
            "stats_code":  final_state.get("stats_code"),
            "raw_result":  final_state.get("raw_result"),
            "explanation": final_state.get("explanation", ""),
            "error":       final_state.get("error"),
            "cached":      final_state.get("cached", False),
        }

    except Exception as e:
        logger.error("[workflow] Graph execution failed: %s", e, exc_info=True)
        return {
            "query_type":  "",
            "sql_query":   None,
            "stats_code":  None,
            "raw_result":  None,
            "explanation": f"An error occurred: {str(e)}",
            "error":       str(e),
            "cached":      False,
        }