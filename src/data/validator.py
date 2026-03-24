# src/data/validator.py
# ------------------------------------------------------------
# PURPOSE: Validates uploaded CSV/Excel files before any
# processing happens. This is the first line of defense —
# if a file fails validation, it never touches the database.
#
# USED BY: src/data/loader.py (calls validate_file first)
#          app.py (the Gradio UI will show validation errors to users)
#
# DESIGN DECISION: We raise ValueError with clear human-readable
# messages. The UI layer (app.py) will catch these and show them
# to the user as friendly error messages — never raw stack traces.
# ------------------------------------------------------------

import os
from pathlib import Path

import pandas as pd

from src.utils.config import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
)


def validate_file(file_path: str) -> None:
    """
    Validates an uploaded file for use in FinanceGPT.

    Performs the following checks in order:
    1. File exists on disk
    2. File extension is .csv or .xlsx
    3. File size is within the allowed limit
    4. File is not empty (0 bytes)
    5. File can be read without corruption errors
    6. File contains at least 1 row of data

    Args:
        file_path: The full path to the file as a string.
                   Example: "C:/Users/Ahmed/uploads/sales.csv"

    Returns:
        None — if validation passes, this function simply returns.

    Raises:
        ValueError: With a descriptive message if any check fails.
        This error is designed to be caught and shown to the user.
    """

    # -----------------------------------------------------------
    # CHECK 1: Does the file actually exist?
    # -----------------------------------------------------------
    # Path(file_path) converts a string path to a Path object,
    # which has useful methods like .exists(), .suffix, .stat()
    path = Path(file_path)

    if not path.exists():
        raise ValueError(
            f"File not found: '{path.name}'. "
            "Please make sure the file was uploaded correctly."
        )

    # -----------------------------------------------------------
    # CHECK 2: Is the file extension allowed?
    # -----------------------------------------------------------
    # path.suffix gives the extension including the dot: ".csv", ".xlsx"
    # We convert to lowercase so "Sales.CSV" and "sales.csv" both work.
    file_extension = path.suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: '{file_extension}'. "
            f"FinanceGPT only accepts: {', '.join(ALLOWED_EXTENSIONS)}. "
            "Please upload a CSV or Excel file."
        )

    # -----------------------------------------------------------
    # CHECK 3: Is the file within the size limit?
    # -----------------------------------------------------------
    # path.stat().st_size returns the file size in bytes.
    file_size_bytes = path.stat().st_size

    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        # Convert bytes to MB for a human-readable error message.
        actual_size_mb = file_size_bytes / (1024 * 1024)
        raise ValueError(
            f"File too large: {actual_size_mb:.1f} MB. "
            f"Maximum allowed size is {MAX_FILE_SIZE_MB} MB. "
            "Please reduce the file size and try again."
        )

    # -----------------------------------------------------------
    # CHECK 4: Is the file empty (0 bytes)?
    # -----------------------------------------------------------
    if file_size_bytes == 0:
        raise ValueError(
            f"File '{path.name}' is empty (0 bytes). "
            "Please upload a file that contains data."
        )

    # -----------------------------------------------------------
    # CHECK 5 & 6: Can the file be read? Does it have data rows?
    # -----------------------------------------------------------
    # We try to actually read the file here. This catches:
    # - Corrupted files (will throw a parsing error)
    # - Files with only a header row but no data (0 rows)
    # - Files where the encoding is completely broken
    #
    # We use nrows=5 to only read the first 5 rows — this makes
    # the validation fast even for huge files. We just need to
    # confirm the file IS readable, not read the whole thing.
    try:
        if file_extension == ".csv":
            df_sample = pd.read_csv(file_path, nrows=5)
        else:  # .xlsx
            df_sample = pd.read_excel(file_path, nrows=5)

    except pd.errors.EmptyDataError:
        # This specific error means the file is completely blank
        raise ValueError(
            f"File '{path.name}' appears to be blank. "
            "Please upload a file with a header row and at least one data row."
        )
    except Exception as e:
        # Catch all other reading errors (corrupted file, bad encoding, etc.)
        raise ValueError(
            f"Could not read file '{path.name}'. "
            "The file may be corrupted or in an unsupported format. "
            f"Technical detail: {str(e)}"
        )

    # Check if there are any rows of actual data (not counting the header)
    if len(df_sample) == 0:
        raise ValueError(
            f"File '{path.name}' has a header row but no data rows. "
            "Please upload a file with at least one row of data."
        )

    # -----------------------------------------------------------
    # All checks passed — file is valid
    # -----------------------------------------------------------
    # We intentionally return None (no return value needed).
    # The caller (loader.py) simply proceeds if no exception was raised.
    return None