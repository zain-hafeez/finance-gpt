# tests/test_loader.py
# ------------------------------------------------------------
# PURPOSE: Tests for the data loader in src/data/loader.py
#
# These tests verify that:
# - CSV files are loaded into DataFrames correctly
# - Column names are cleaned properly
# - SQLite database is created with the right data
# - The returned dictionary has all expected keys
#
# RUN: pytest tests/test_loader.py -v
# ------------------------------------------------------------

import os
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import _clean_column_name, load_data


# ============================================================
# FIXTURES
# A pytest "fixture" is a function that creates test data
# and is automatically called for tests that request it.
# ============================================================

@pytest.fixture
def valid_csv_file():
    """Creates a temporary valid CSV file for testing. Cleans up after."""
    with tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w"
    ) as f:
        f.write("Date,Region,Total Sales ($),Unit Price\n")
        f.write("2024-01-01,North,250.00,25.00\n")
        f.write("2024-01-02,South,200.00,40.00\n")
        f.write("2024-01-03,East,200.00,25.00\n")
        temp_path = f.name

    yield temp_path  # Provide the path to the test

    # After the test finishes, clean up the temp file
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_db_path():
    """Creates a temporary path for the SQLite database. Cleans up after."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    # Delete the empty file so SQLite can create it fresh
    os.unlink(db_path)

    yield db_path

    # After the test, remove the database file
    if os.path.exists(db_path):
        os.unlink(db_path)


# ============================================================
# TESTS: Column name cleaning
# ============================================================

class TestColumnCleaning:
    """Tests for the _clean_column_name helper function."""

    def test_lowercase_conversion(self):
        """Column names should be converted to lowercase."""
        assert _clean_column_name("Region") == "region"
        assert _clean_column_name("TOTAL") == "total"

    def test_spaces_replaced_with_underscores(self):
        """Spaces in column names should become underscores."""
        assert _clean_column_name("First Name") == "first_name"
        assert _clean_column_name("Total Sales") == "total_sales"

    def test_special_characters_removed(self):
        """Special characters like $, (, ) should be replaced with underscores."""
        result = _clean_column_name("Total Sales ($)")
        assert result == "total_sales"

    def test_leading_trailing_whitespace_stripped(self):
        """Leading and trailing spaces should be removed."""
        assert _clean_column_name("  region  ") == "region"

    def test_digit_prefix_handled(self):
        """Column names starting with a digit should get an underscore prefix."""
        assert _clean_column_name("123revenue").startswith("_")

    def test_already_clean_name_unchanged(self):
        """A name that's already clean should pass through unchanged."""
        assert _clean_column_name("total_sales") == "total_sales"


# ============================================================
# TESTS: load_data function
# ============================================================

class TestLoadData:
    """Tests for the main load_data function."""

    def test_returns_dict_with_required_keys(self, valid_csv_file, temp_db_path):
        """load_data should return a dict with all expected keys."""
        result = load_data(valid_csv_file, db_path=temp_db_path)

        required_keys = {"dataframe", "db_path", "table_name", "columns", "row_count", "file_name"}
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_returns_correct_row_count(self, valid_csv_file, temp_db_path):
        """row_count should match the actual number of data rows."""
        result = load_data(valid_csv_file, db_path=temp_db_path)
        assert result["row_count"] == 3  # Our fixture has 3 data rows

    def test_dataframe_is_pandas_dataframe(self, valid_csv_file, temp_db_path):
        """The 'dataframe' key should contain a real Pandas DataFrame."""
        result = load_data(valid_csv_file, db_path=temp_db_path)
        assert isinstance(result["dataframe"], pd.DataFrame)

    def test_column_names_are_cleaned(self, valid_csv_file, temp_db_path):
        """Column names in the returned DataFrame should be cleaned."""
        result = load_data(valid_csv_file, db_path=temp_db_path)
        columns = result["columns"]

        # Our fixture has "Total Sales ($)" which should become "total_sales"
        assert "total_sales" in columns, f"Expected 'total_sales' in {columns}"
        # "Unit Price" should become "unit_price"
        assert "unit_price" in columns, f"Expected 'unit_price' in {columns}"

    def test_sqlite_database_created(self, valid_csv_file, temp_db_path):
        """A SQLite database file should be created at the specified path."""
        result = load_data(valid_csv_file, db_path=temp_db_path)

        # The database file should now exist
        assert os.path.exists(result["db_path"]), "SQLite database was not created"

    def test_sqlite_contains_data(self, valid_csv_file, temp_db_path):
        """The SQLite database should contain the same data as the CSV."""
        result = load_data(valid_csv_file, db_path=temp_db_path)

        # Connect to the database and count rows
        conn = sqlite3.connect(result["db_path"])
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM data")
        row_count = cursor.fetchone()[0]
        conn.close()

        assert row_count == 3, f"Expected 3 rows in SQLite, found {row_count}"

    def test_invalid_file_raises_value_error(self, temp_db_path):
        """load_data should raise ValueError for an invalid file."""
        with pytest.raises(ValueError):
            load_data("nonexistent_file.csv", db_path=temp_db_path)

    def test_table_name_is_always_data(self, valid_csv_file, temp_db_path):
        """The SQLite table name should always be 'data'."""
        result = load_data(valid_csv_file, db_path=temp_db_path)
        assert result["table_name"] == "data"
