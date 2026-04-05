# src/graph/checkpointer.py
"""
LangGraph Checkpointer Setup — Module 9

Uses SqliteSaver if langgraph-checkpoint-sqlite is installed.
Falls back to MemorySaver otherwise.

IMPORTANT: SqliteSaver.from_conn_string() in newer langgraph versions
returns a context manager. We use it as one and return the inner saver.
For MemorySaver we just instantiate directly.
"""

import logging
from pathlib import Path
from src.utils.config import CHECKPOINTS_DB

logger = logging.getLogger(__name__)


def get_checkpointer():
    """
    Returns a ready-to-use LangGraph checkpointer instance.

    Tries SqliteSaver first (persistent). Falls back to MemorySaver.
    Handles both old API (returns saver directly) and new API
    (returns context manager) of langgraph-checkpoint-sqlite.
    """
    db_path = Path(CHECKPOINTS_DB)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3

        # Use a persistent sqlite3 connection directly.
        # This avoids the context manager issue in newer langgraph versions.
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        logger.info("[checkpointer] SQLite checkpointer initialized at: %s", db_path)
        return checkpointer

    except (ImportError, Exception) as e:
        from langgraph.checkpoint.memory import MemorySaver
        logger.warning(
            "[checkpointer] SQLite checkpointer unavailable (%s). "
            "Using MemorySaver (state resets on restart).", e
        )
        return MemorySaver()


def get_checkpointer_type() -> str:
    """Returns 'sqlite' or 'memory' depending on what's available."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401
        return "sqlite"
    except ImportError:
        return "memory"