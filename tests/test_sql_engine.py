# tests/test_sql_engine.py
# Module 3 — Tests for sql_engine.py
# LangChain 1.x compatible — mocks SQLDatabase and direct LLM call

from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from src.engines.sql_engine import run_sql_query


class TestRunSqlQuery:

    def _make_mock_llm(self, sql_response: str):
        """Helper: returns a mock LLM that responds with the given SQL string."""
        mock_llm = MagicMock()
        # llm.invoke() returns an AIMessage object — we mock that too
        mock_llm.invoke.return_value = AIMessage(content=sql_response)
        return mock_llm

    def test_successful_query_returns_correct_shape(self):
        """
        When LLM returns valid SQL and db.run() succeeds,
        result must have success=True and all expected keys.
        """
        mock_llm = self._make_mock_llm("SELECT SUM(sales) FROM data")

        with patch("src.engines.sql_engine.SQLDatabase") as mock_db_cls:
            mock_db = MagicMock()
            mock_db_cls.from_uri.return_value = mock_db
            mock_db.get_table_info.return_value = "Table: data\nColumns: sales REAL"
            mock_db.run.return_value = "[(45200.0,)]"

            result = run_sql_query(
                question="What is total sales?",
                db_path="/fake/finance.db",
                llm=mock_llm,
            )

        assert result["success"] is True
        assert result["sql"] == "SELECT SUM(sales) FROM data"
        assert result["result"] == "[(45200.0,)]"
        assert result["error"] is None
        assert "row_count" in result

    def test_dangerous_sql_blocked_before_execution(self):
        """
        If LLM generates DROP TABLE, validate_sql() must block it
        and db.run() must NEVER be called.
        """
        mock_llm = self._make_mock_llm("DROP TABLE data")

        with patch("src.engines.sql_engine.SQLDatabase") as mock_db_cls:
            mock_db = MagicMock()
            mock_db_cls.from_uri.return_value = mock_db
            mock_db.get_table_info.return_value = "Table: data"

            result = run_sql_query(
                question="Delete everything",
                db_path="/fake/finance.db",
                llm=mock_llm,
            )

        assert result["success"] is False
        assert result["error"] is not None
        mock_db.run.assert_not_called()

    def test_llm_failure_returns_clean_error_dict(self):
        """
        If the LLM call raises an exception, run_sql_query must return
        a clean error dict — never raise or crash.
        """
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Simulated Groq failure")

        with patch("src.engines.sql_engine.SQLDatabase") as mock_db_cls:
            mock_db = MagicMock()
            mock_db_cls.from_uri.return_value = mock_db
            mock_db.get_table_info.return_value = "Table: data"

            result = run_sql_query(
                question="Total sales?",
                db_path="/fake/finance.db",
                llm=mock_llm,
            )

        assert result["success"] is False
        assert result["error"] is not None
        assert result["result"] is None

    def test_markdown_fenced_sql_is_cleaned_correctly(self):
        """
        LLMs often wrap SQL in markdown fences like ```sql ... ```
        The _clean_sql_output function must strip these before validation.
        """
        fenced_sql = "```sql\nSELECT COUNT(*) FROM data\n```"
        mock_llm = self._make_mock_llm(fenced_sql)

        with patch("src.engines.sql_engine.SQLDatabase") as mock_db_cls:
            mock_db = MagicMock()
            mock_db_cls.from_uri.return_value = mock_db
            mock_db.get_table_info.return_value = "Table: data"
            mock_db.run.return_value = "[(15,)]"

            result = run_sql_query(
                question="How many rows?",
                db_path="/fake/finance.db",
                llm=mock_llm,
            )

        assert result["success"] is True
        assert "```" not in result["sql"]
        assert result["sql"].upper().startswith("SELECT")

    def test_result_always_has_all_expected_keys(self):
        """Success and failure dicts must have identical shapes — no KeyError ever."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("any error")
        expected_keys = {"success", "sql", "result", "row_count", "error"}

        with patch("src.engines.sql_engine.SQLDatabase") as mock_db_cls:
            mock_db = MagicMock()
            mock_db_cls.from_uri.return_value = mock_db
            mock_db.get_table_info.return_value = "Table: data"

            result = run_sql_query("anything", "/fake/db.db", mock_llm)

        assert set(result.keys()) == expected_keys
