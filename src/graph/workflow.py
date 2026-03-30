# src/graph/workflow.py
"""
LangGraph StateGraph — Full Pipeline Assembly
=============================================
Wires all nodes into a compiled graph with:
  - App-level cache check at the front (skip entire pipeline on hit)
  - LLM-powered router (sql vs stats)
  - SQL or stats engine execution
  - Plain-English explanation generation
  - Cache write at the end (store result for future hits)

Public API: run_query(question, file_path, db_path, session_id) -> dict
"""

import logging
from langgraph.graph import StateGraph, START, END

from src.graph.state import FinanceGPTState
from src.graph.router import router_node, route_to_engine
from src.graph.nodes import (
    sql_node,
    stats_node,
    explain_node,
    cache_node,
    cache_write_node,
)
from src.utils.cache_setup import setup_cache

logger = logging.getLogger(__name__)

# Enable LangChain InMemoryCache globally — runs once when module is imported
setup_cache()


def _route_after_cache(state: dict) -> str:
    """
    Conditional edge function called after cache_node.

    If the cache had a hit (cached=True), we go straight to END.
    If cache miss (cached=False), we continue to the router.

    This is the key routing decision that makes the cache work:
    a cache hit completely skips all LLM calls and SQL execution.
    """
    if state.get('cached', False):
        logger.info("[workflow] Cache hit — routing to END")
        return 'end'
    return 'router'


def _build_graph():
    """
    Build and compile the LangGraph StateGraph.

    Node execution order for a cache MISS:
      cache_node → router_node → [sql_node OR stats_node] → explain_node → cache_write_node

    Node execution order for a cache HIT:
      cache_node → END  (everything else is skipped)
    """
    graph = StateGraph(FinanceGPTState)

    # --- Register all nodes ---
    graph.add_node('cache_check', cache_node)
    graph.add_node('router', router_node)
    graph.add_node('sql_node', sql_node)
    graph.add_node('stats_node', stats_node)
    graph.add_node('explain', explain_node)
    graph.add_node('cache_write', cache_write_node)

    # --- Wire the edges ---

    # Entry point → cache check first
    graph.add_edge(START, 'cache_check')

    # After cache check: hit → END, miss → router
    graph.add_conditional_edges(
        'cache_check',
        _route_after_cache,
        {'end': END, 'router': 'router'}
    )

    # Router → sql_node or stats_node (existing M5 conditional edge)
    graph.add_conditional_edges(
        'router',
        route_to_engine,
        {'sql': 'sql_node', 'stats': 'stats_node'}
    )

    # Both engine paths converge at explain_node
    graph.add_edge('sql_node', 'explain')
    graph.add_edge('stats_node', 'explain')

    # After explanation → write to cache → done
    graph.add_edge('explain', 'cache_write')
    graph.add_edge('cache_write', END)

    return graph.compile()


# Compile once at module load — reused for all queries
_graph = _build_graph()


def run_query(
    question: str,
    file_path: str,
    db_path: str,
    session_id: str = 'default'
) -> dict:
    """
    Single public API for the entire FinanceGPT pipeline.

    Args:
        question:   User's plain English question
        file_path:  Path to the uploaded CSV/Excel file
        db_path:    Path to the SQLite database file
        session_id: Session identifier (used by M9 checkpointer)

    Returns dict with keys:
        success      bool   — True if no error occurred
        explanation  str    — Plain English answer for the user
        query_type   str    — 'sql' or 'stats'
        sql_query    str|None — Generated SQL (SQL path only)
        stats_code   str|None — Generated Python code (stats path only)
        raw_result   Any    — Raw data from the engine
        cached       bool   — True if result came from cache
        error        str|None — Error message if something failed
    """
    initial_state = {
        'file_path':   file_path,
        'db_path':     db_path,
        'query':       question.strip(),
        'session_id':  session_id,
        'query_type':  '',
        'cached':      False,
        'sql_query':   None,
        'sql_valid':   False,
        'stats_code':  None,
        'raw_result':  None,
        'explanation': '',
        'error':       None,
    }

    final_state = _graph.invoke(initial_state)

    return {
        'success':     final_state.get('error') is None,
        'explanation': final_state.get('explanation', ''),
        'query_type':  final_state.get('query_type', ''),
        'sql_query':   final_state.get('sql_query'),
        'stats_code':  final_state.get('stats_code'),
        'raw_result':  final_state.get('raw_result'),
        'cached':      final_state.get('cached', False),
        'error':       final_state.get('error'),
    }