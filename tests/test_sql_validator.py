# tests/test_sql_validator.py
# Module 3 — Tests for sql_validator.py
# No mocking needed — pure logic, no external dependencies

from src.engines.sql_validator import validate_sql, get_validation_error


class TestValidateSql:

    # ── Valid queries — should return True ─────────────────────────────────

    def test_simple_select_is_valid(self):
        assert validate_sql("SELECT * FROM data") is True

    def test_select_with_where_is_valid(self):
        assert validate_sql("SELECT region, SUM(sales) FROM data WHERE region='North' GROUP BY region") is True

    def test_select_with_aggregation_is_valid(self):
        assert validate_sql("SELECT COUNT(*) FROM data") is True

    def test_select_with_order_by_is_valid(self):
        assert validate_sql("SELECT product, sales FROM data ORDER BY sales DESC LIMIT 5") is True

    def test_lowercase_select_is_valid(self):
        # Validator should be case-insensitive
        assert validate_sql("select * from data") is True

    # ── Dangerous queries — should return False ─────────────────────────────

    def test_drop_table_is_blocked(self):
        assert validate_sql("DROP TABLE data") is False

    def test_delete_is_blocked(self):
        assert validate_sql("DELETE FROM data WHERE region='North'") is False

    def test_insert_is_blocked(self):
        assert validate_sql("INSERT INTO data VALUES (1, 2, 3)") is False

    def test_update_is_blocked(self):
        assert validate_sql("UPDATE data SET sales=0") is False

    def test_sql_injection_chaining_is_blocked(self):
        # The semicolon allows chaining: SELECT ... ; DROP TABLE data
        assert validate_sql("SELECT * FROM data; DROP TABLE data") is False

    def test_empty_query_is_blocked(self):
        assert validate_sql("") is False

    def test_non_select_start_is_blocked(self):
        # Even if it doesn't contain forbidden keywords, must start with SELECT
        assert validate_sql("EXPLAIN SELECT * FROM data") is False


class TestGetValidationError:

    def test_returns_string_for_empty_query(self):
        error = get_validation_error("")
        assert isinstance(error, str)
        assert len(error) > 0

    def test_returns_string_for_forbidden_keyword(self):
        error = get_validation_error("DROP TABLE data")
        assert "DROP" in error or "forbidden" in error.lower()

    def test_returns_string_for_non_select(self):
        error = get_validation_error("EXEC something")
        assert isinstance(error, str)