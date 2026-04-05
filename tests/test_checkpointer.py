# tests/test_checkpointer.py
"""
Module 9 — State Persistence Tests
Tests for: checkpointer setup, session isolation, run_query session_id integration
"""

import pytest
from unittest.mock import patch, MagicMock


# ── Group 1: get_checkpointer_type() ─────────────────────────────────────────

class TestGetCheckpointerType:
    """Tests for the helper that tells us which backend is active."""

    def test_returns_sqlite_when_package_available(self):
        """
        When langgraph-checkpoint-sqlite is installed,
        get_checkpointer_type() should return 'sqlite'.
        """
        # We mock the import to simulate the package being available
        mock_saver = MagicMock()
        with patch.dict("sys.modules", {"langgraph.checkpoint.sqlite": mock_saver}):
            from importlib import reload
            import src.graph.checkpointer as cp_module
            reload(cp_module)
            result = cp_module.get_checkpointer_type()
        assert result == "sqlite"

    def test_returns_memory_when_package_missing(self):
        """
        When langgraph-checkpoint-sqlite is NOT installed,
        get_checkpointer_type() should return 'memory'.
        """
        with patch.dict("sys.modules", {"langgraph.checkpoint.sqlite": None}):
            from importlib import reload
            import src.graph.checkpointer as cp_module
            reload(cp_module)
            result = cp_module.get_checkpointer_type()
        assert result == "memory"

    def test_returns_string_type(self):
        """Return value must always be a string."""
        from src.graph.checkpointer import get_checkpointer_type
        result = get_checkpointer_type()
        assert isinstance(result, str)

    def test_returns_valid_value(self):
        """Return value must be one of the two valid options."""
        from src.graph.checkpointer import get_checkpointer_type
        result = get_checkpointer_type()
        assert result in ("sqlite", "memory")


# ── Group 2: get_checkpointer() ──────────────────────────────────────────────

class TestGetCheckpointer:
    """Tests for the checkpointer factory function."""

    def test_returns_memory_saver_as_fallback(self, tmp_path):
        """
        When SqliteSaver import fails (package missing),
        get_checkpointer() should fall back to MemorySaver.
        """
        mock_memory_saver = MagicMock()
        mock_memory_instance = MagicMock()
        mock_memory_saver.return_value = mock_memory_instance

        with patch.dict("sys.modules", {"langgraph.checkpoint.sqlite": None}):
            with patch("langgraph.checkpoint.memory.MemorySaver", mock_memory_saver):
                from importlib import reload
                import src.graph.checkpointer as cp_module
                reload(cp_module)
                # Override CHECKPOINTS_DB to use tmp_path
                cp_module.CHECKPOINTS_DB = str(tmp_path / "test_checkpoints.db")
                result = cp_module.get_checkpointer()

        # Should have returned a MemorySaver instance
        assert result is not None

    def test_checkpointer_db_directory_created(self, tmp_path):
        """
        get_checkpointer() must create the parent directory if it doesn't exist.
        This is critical for Docker deployments where paths may not exist.
        """
        nested_path = tmp_path / "nested" / "dir" / "checkpoints.db"
        assert not nested_path.parent.exists()

        # Patch CHECKPOINTS_DB to our nested path
        with patch("src.graph.checkpointer.CHECKPOINTS_DB", str(nested_path)):
            try:
                from src.graph.checkpointer import get_checkpointer
                get_checkpointer()
            except Exception:
                pass  # We don't care if the checkpointer itself fails

        # The directory should have been created
        assert nested_path.parent.exists()


# ── Group 3: run_query() session_id integration ───────────────────────────────

class TestRunQuerySessionIntegration:
    """
    Tests that run_query() correctly passes session_id as LangGraph thread_id.
    We mock the graph so we don't need real LLM calls or files.
    """

    def _make_mock_graph(self, return_state: dict):
        """Helper: creates a mock compiled graph that returns a fixed state."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = return_state
        return mock_graph

    def test_run_query_passes_session_id_as_thread_id(self):
        """
        When run_query() is called with session_id='test-session-123',
        graph.invoke() should receive config={"configurable": {"thread_id": "test-session-123"}}.
        """
        fake_state = {
            "query_type": "sql",
            "sql_query": "SELECT 1",
            "stats_code": None,
            "raw_result": "[(1,)]",
            "explanation": "The answer is 1.",
            "error": None,
            "cached": False,
        }
        mock_graph = self._make_mock_graph(fake_state)

        with patch("src.graph.workflow.get_graph", return_value=mock_graph):
            from src.graph.workflow import run_query
            run_query(
                question="test question",
                file_path="test.csv",
                db_path="test.db",
                session_id="test-session-123",
            )

        # Verify graph.invoke was called with the correct thread_id
        call_args = mock_graph.invoke.call_args
        config_passed = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("config", {})
        # Either positional or keyword arg
        if not config_passed:
            config_passed = call_args[1].get("config", call_args[0][1] if len(call_args[0]) > 1 else {})
        assert config_passed.get("configurable", {}).get("thread_id") == "test-session-123"

    def test_run_query_default_session_id(self):
        """
        When session_id is not provided, run_query() should use 'default'.
        No KeyError or crash should occur.
        """
        fake_state = {
            "query_type": "sql", "sql_query": None, "stats_code": None,
            "raw_result": None, "explanation": "ok", "error": None, "cached": False,
        }
        mock_graph = self._make_mock_graph(fake_state)

        with patch("src.graph.workflow.get_graph", return_value=mock_graph):
            from src.graph.workflow import run_query
            result = run_query("question", "file.csv", "db.db")  # No session_id

        assert result is not None

    def test_run_query_different_sessions_independent(self):
        """
        Two run_query() calls with different session_ids should pass
        different thread_ids to the graph — sessions are isolated.
        """
        fake_state = {
            "query_type": "sql", "sql_query": "SELECT 1", "stats_code": None,
            "raw_result": "1", "explanation": "ok", "error": None, "cached": False,
        }
        mock_graph = self._make_mock_graph(fake_state)
        recorded_configs = []

        def capture_invoke(state, config=None):
            recorded_configs.append(config)
            return fake_state

        mock_graph.invoke.side_effect = capture_invoke

        with patch("src.graph.workflow.get_graph", return_value=mock_graph):
            from src.graph.workflow import run_query
            run_query("q1", "f.csv", "d.db", session_id="session-A")
            run_query("q2", "f.csv", "d.db", session_id="session-B")

        thread_ids = [
            c.get("configurable", {}).get("thread_id")
            for c in recorded_configs
            if c
        ]
        assert "session-A" in thread_ids
        assert "session-B" in thread_ids
        assert thread_ids[0] != thread_ids[1]

    def test_run_query_returns_dict_with_required_keys(self):
        """run_query() must always return a dict with the standard 7 keys."""
        fake_state = {
            "query_type": "stats", "sql_query": None, "stats_code": "result=1",
            "raw_result": 42, "explanation": "The answer is 42.", "error": None, "cached": False,
        }
        mock_graph = self._make_mock_graph(fake_state)

        with patch("src.graph.workflow.get_graph", return_value=mock_graph):
            from src.graph.workflow import run_query
            result = run_query("question", "file.csv", "db.db", session_id="s1")

        required_keys = {"query_type", "sql_query", "stats_code", "raw_result", "explanation", "error", "cached"}
        assert required_keys.issubset(result.keys())

    def test_run_query_handles_graph_exception_gracefully(self):
        """
        If the graph raises an unexpected exception, run_query() must
        catch it and return a proper error dict — never propagate the exception.
        """
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = RuntimeError("Graph exploded")

        with patch("src.graph.workflow.get_graph", return_value=mock_graph):
            from src.graph.workflow import run_query
            result = run_query("question", "file.csv", "db.db", session_id="s1")

        assert result["error"] is not None
        assert "exploded" in result["error"].lower() or result["explanation"] != ""
        assert result["cached"] is False

    def test_run_query_cached_result_flag(self):
        """
        When the graph returns cached=True (cache hit),
        run_query() must pass that through in the result dict.
        """
        fake_state = {
            "query_type": "sql", "sql_query": "SELECT 1", "stats_code": None,
            "raw_result": "cached data", "explanation": "Cached answer.", "error": None, "cached": True,
        }
        mock_graph = self._make_mock_graph(fake_state)

        with patch("src.graph.workflow.get_graph", return_value=mock_graph):
            from src.graph.workflow import run_query
            result = run_query("question", "file.csv", "db.db", session_id="s1")

        assert result["cached"] is True