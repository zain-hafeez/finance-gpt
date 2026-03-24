# tests/test_validator.py
# ------------------------------------------------------------
# PURPOSE: Tests for the file validator in src/data/validator.py
#
# HOW TESTS WORK:
# - Each function starting with "test_" is a test case
# - pytest finds them automatically when you run "pytest"
# - Inside each test, we call our function and use "assert"
#   to check the result is what we expect
# - Tests that expect an error use "pytest.raises()"
#
# RUN THESE TESTS: pytest tests/test_validator.py -v
# The -v flag means "verbose" — shows each test name as it runs
# ------------------------------------------------------------

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Import the function we're testing
from src.data.validator import validate_file


# ============================================================
# HELPER: A path to our valid test CSV
# ============================================================
FIXTURES_DIR = Path(__file__).parent / "fixtures"
VALID_CSV = str(FIXTURES_DIR / "valid_sales.csv")


# ============================================================
# TESTS: File existence and extension
# ============================================================

class TestFileExistence:
    """Tests that check whether files exist."""

    def test_valid_csv_passes(self):
        """A real, valid CSV file should pass without raising any error."""
        # If validate_file raises, this test fails.
        # If it returns None (no error), the test passes.
        validate_file(VALID_CSV)  # Should not raise

    def test_nonexistent_file_raises(self):
        """A path to a file that doesn't exist should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_file("C:/fake/path/nonexistent.csv")

        # Check the error message contains helpful text
        assert "not found" in str(exc_info.value).lower()


class TestFileExtension:
    """Tests for file extension validation."""

    def test_csv_extension_passes(self):
        """A .csv file should be accepted."""
        validate_file(VALID_CSV)  # Should not raise

    def test_txt_extension_rejected(self):
        """A .txt file should be rejected even if it contains CSV data."""
        # tempfile.NamedTemporaryFile creates a real temporary file
        # so we can test with a file that actually exists on disk
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("col1,col2\nval1,val2\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                validate_file(temp_path)
            assert "unsupported file type" in str(exc_info.value).lower()
        finally:
            # Always clean up temp files, even if the test fails
            os.unlink(temp_path)

    def test_pdf_extension_rejected(self):
        """A .pdf file should be rejected."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"fake pdf content")
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                validate_file(temp_path)
            assert "unsupported" in str(exc_info.value).lower()
        finally:
            os.unlink(temp_path)

    def test_uppercase_csv_extension_passes(self):
        """A .CSV file (uppercase) should still be accepted."""
        with tempfile.NamedTemporaryFile(suffix=".CSV", delete=False, mode="w") as f:
            f.write("col1,col2\nval1,val2\n")
            temp_path = f.name

        try:
            validate_file(temp_path)  # Should not raise
        finally:
            os.unlink(temp_path)


# ============================================================
# TESTS: File size
# ============================================================

class TestFileSize:
    """Tests for file size validation."""

    def test_empty_file_rejected(self):
        """A file with 0 bytes should be rejected."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name
            # Write nothing — the file is 0 bytes

        try:
            with pytest.raises(ValueError) as exc_info:
                validate_file(temp_path)
            error_msg = str(exc_info.value).lower()
            assert "empty" in error_msg
        finally:
            os.unlink(temp_path)

    def test_oversized_file_rejected(self):
        """A file larger than MAX_FILE_SIZE_MB should be rejected."""
        """A file larger than MAX_FILE_SIZE_MB should be rejected."""
        # We patch the value WHERE IT IS USED (inside validator.py's namespace),
        # not where it was defined (config.py). This is the correct way to patch
        # imported variables in Python.
        import src.data.validator as validator_module

        # Save the original value so we can restore it after the test
        original_max_bytes = validator_module.MAX_FILE_SIZE_BYTES

        # Override the value INSIDE validator.py's namespace
        validator_module.MAX_FILE_SIZE_BYTES = 10  # Only 10 bytes allowed

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("col1,col2\n" + "a,b\n" * 100)  # Much more than 10 bytes
            temp_path = f.name

        
        try:
            with pytest.raises(ValueError) as exc_info:
                validate_file(temp_path)
            assert "too large" in str(exc_info.value).lower()
        finally:
            os.unlink(temp_path)
            # ALWAYS restore the original value, even if the test fails
            validator_module.MAX_FILE_SIZE_BYTES = original_max_bytes


# ============================================================
# TESTS: File content
# ============================================================

class TestFileContent:
    """Tests for file content validation."""

    def test_header_only_csv_rejected(self):
        """A CSV with only a header row and no data rows should be rejected."""
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            f.write("col1,col2,col3\n")  # Only header, no data rows
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                validate_file(temp_path)
            error_msg = str(exc_info.value).lower()
            assert "no data rows" in error_msg or "blank" in error_msg
        finally:
            os.unlink(temp_path)

    def test_valid_csv_with_data_passes(self):
        """A CSV with header + at least one data row should pass."""
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            f.write("name,value\nAlice,100\n")
            temp_path = f.name

        try:
            validate_file(temp_path)  # Should not raise
        finally:
            os.unlink(temp_path)
