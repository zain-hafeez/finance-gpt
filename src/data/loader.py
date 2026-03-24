# src/data/loader.py
# ------------------------------------------------------------
# PURPOSE: Loads a validated CSV or Excel file into two formats:
#   1. A Pandas DataFrame (for the Stats Engine, Module 4)
#   2. A SQLite database file (for the SQL Engine, Module 3)
#
# USED BY: app.py (called when user uploads a file)
#          src/graph/nodes.py (the validate_and_load LangGraph node)
#
# RETURNS: A dict with 'dataframe', 'db_path', 'columns', 'row_count'
#          so the caller has everything it needs about the loaded data.
#
# DESIGN DECISIONS:
# - Column names are cleaned (lowercase, underscores) for safe SQL use
# - We use SQLAlchemy's create_engine for the SQLite connection because
#   LangChain's SQLDatabase wrapper requires it
# - The table name in SQLite is always "data" for simplicity
# ------------------------------------------------------------

import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from src.data.validator import validate_file
from src.utils.config import DB_PATH


def _clean_column_name(col: str) -> str:
    """
    Cleans a single column name for safe use in SQL.

    Examples:
        "Total Sales ($)"  →  "total_sales"
        "First Name"       →  "first_name"
        "Region "          →  "region"
        "123revenue"       →  "_123revenue"

    The rules:
    - Strip leading/trailing whitespace
    - Convert to lowercase
    - Replace spaces and special characters with underscores
    - Remove consecutive underscores
    - If column starts with a digit, prefix with underscore
      (SQL column names can't start with a digit)
    """
    # Step 1: Strip whitespace and convert to lowercase
    col = str(col).strip().lower()

    # Step 2: Replace any character that isn't a letter, digit, or underscore
    # with an underscore. re.sub(pattern, replacement, string)
    col = re.sub(r"[^a-z0-9_]", "_", col)

    # Step 3: Replace multiple consecutive underscores with just one
    col = re.sub(r"_+", "_", col)

    # Step 4: Remove leading/trailing underscores that might have been created
    col = col.strip("_")

    # Step 5: If the column name starts with a digit, prefix with underscore
    if col and col[0].isdigit():
        col = f"_{col}"

    # Step 6: Handle edge case where col is now empty
    if not col:
        col = "unnamed"

    return col


def _clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies _clean_column_name to every column in the DataFrame.
    Also handles duplicate column names after cleaning by appending
    a number (e.g., "total", "total_1", "total_2").
    """
    # Clean all column names
    new_columns = [_clean_column_name(col) for col in df.columns]

    # Handle duplicates: if two columns clean to the same name,
    # append _1, _2, etc. to make them unique
    seen: dict = {}
    unique_columns = []
    for col in new_columns:
        if col in seen:
            seen[col] += 1
            unique_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            unique_columns.append(col)

    df.columns = unique_columns
    return df


def load_data(file_path: str, db_path: str = DB_PATH) -> dict[str, Any]:
    """
    Loads a CSV or Excel file into a Pandas DataFrame and SQLite database.

    This is the main function of this module. It:
    1. Validates the file (raises ValueError if invalid)
    2. Reads the file into a DataFrame
    3. Cleans all column names for SQL safety
    4. Saves data to SQLite (table name: "data")
    5. Returns all the information the rest of the app needs

    Args:
        file_path: Full path to the uploaded CSV or Excel file.
        db_path:   Path where the SQLite .db file will be created.
                   Defaults to the DB_PATH value from config.py.

    Returns:
        A dictionary with these keys:
        {
            "dataframe":  pd.DataFrame  — the loaded data in memory,
            "db_path":    str           — path to the SQLite file,
            "table_name": str           — always "data",
            "columns":    list[str]     — list of cleaned column names,
            "row_count":  int           — number of data rows,
            "file_name":  str           — original filename,
        }

    Raises:
        ValueError: If the file fails validation (from validate_file).
        RuntimeError: If the database write fails.
    """

    # -----------------------------------------------------------
    # STEP 1: Validate the file first
    # -----------------------------------------------------------
    # validate_file() raises ValueError if anything is wrong.
    # If it raises, we don't proceed — the error propagates up.
    validate_file(file_path)

    # -----------------------------------------------------------
    # STEP 2: Read the file into a Pandas DataFrame
    # -----------------------------------------------------------
    path = Path(file_path)
    file_extension = path.suffix.lower()

    try:
        if file_extension == ".csv":
            # Read CSV file. Pandas automatically detects column names
            # from the first row.
            df = pd.read_csv(
                file_path,
                # encoding="utf-8-sig" handles files saved from Excel
                # that have a BOM (Byte Order Mark) at the start
                encoding="utf-8-sig",
            )
        else:  # .xlsx
            # Read Excel file. sheet_name=0 means "read the first sheet".
            df = pd.read_excel(
                file_path,
                sheet_name=0,
                # engine="openpyxl" is required for .xlsx files
                engine="openpyxl",
            )
    except Exception as e:
        raise ValueError(
            f"Failed to read file '{path.name}': {str(e)}. "
            "Please check the file is not open in Excel and try again."
        )

    # -----------------------------------------------------------
    # STEP 3: Clean column names for SQL safety
    # -----------------------------------------------------------
    # We do this BEFORE saving to SQLite so the database has
    # the clean names. The DataFrame and SQLite will always match.
    original_columns = list(df.columns)
    df = _clean_dataframe_columns(df)
    cleaned_columns = list(df.columns)

    # Log the column name changes so the developer can debug issues
    column_mapping = dict(zip(original_columns, cleaned_columns))
    changed = {k: v for k, v in column_mapping.items() if k != v}
    if changed:
        print(f"[loader] Column names cleaned: {changed}")

    # -----------------------------------------------------------
    # STEP 4: Save DataFrame to SQLite database
    # -----------------------------------------------------------
    # We use SQLAlchemy's create_engine() to create a connection.
    # The connection string format for SQLite is: "sqlite:///path/to/file.db"
    # Three slashes for a relative path, four for absolute.
    #
    # WHY SQLAlchemy? Because LangChain's SQLDatabase class (used in Module 3)
    # requires a SQLAlchemy engine, not a raw sqlite3 connection.

    try:
        # Build the SQLAlchemy connection string for SQLite
        # We use an absolute path to avoid confusion about working directory
        abs_db_path = str(Path(db_path).resolve())
        engine = create_engine(f"sqlite:///{abs_db_path}")

        # df.to_sql() writes the DataFrame to the database.
        # Parameters:
        #   "data"       — the table name in SQLite (always "data")
        #   engine       — the database connection
        #   if_exists    — "replace" means: if a "data" table already exists,
        #                  drop it and create a fresh one with this new data
        #   index=False  — don't write the DataFrame's index as a column
        df.to_sql(
            name="data",
            con=engine,
            if_exists="replace",
            index=False,
        )
        engine.dispose()

        print(
            f"[loader] Saved {len(df)} rows × {len(df.columns)} columns "
            f"to SQLite: {abs_db_path}"
        )

    except Exception as e:
        raise RuntimeError(
            f"Failed to save data to database: {str(e)}. "
            "Please try again or contact support."
        )

    # -----------------------------------------------------------
    # STEP 5: Return everything the rest of the app needs
    # -----------------------------------------------------------
    return {
        "dataframe": df,            # Pandas DataFrame for Stats Engine
        "db_path": abs_db_path,     # SQLite file path for SQL Engine
        "table_name": "data",       # Table name (always "data")
        "columns": cleaned_columns, # List of column names (cleaned)
        "row_count": len(df),       # Number of data rows
        "file_name": path.name,     # Original filename for display
    }


def get_schema_description(db_path: str) -> str:
    """
    Returns a human-readable description of the database schema.

    This will be used by the LLM in Module 3 to understand
    what columns exist so it can generate accurate SQL queries.

    Example output:
        "Table: data
         Columns:
           - date (TEXT)
           - region (TEXT)
           - product (TEXT)
           - quantity (INTEGER)
           - unit_price (REAL)
           - total_sales (REAL)"

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A string description of the table schema.
    """
    try:
        # sqlite3 is Python's built-in SQLite library.
        # We use it here (not SQLAlchemy) because we just need to
        # read metadata, not write data.
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # PRAGMA table_info returns one row per column with:
        # cid, name, type, notnull, default_value, is_primary_key
        cursor.execute("PRAGMA table_info(data)")
        columns = cursor.fetchall()
        conn.close()

        if not columns:
            return "No schema found. Please upload a file first."

        # Build a readable description string
        col_descriptions = "\n".join(
            f"  - {col[1]} ({col[2]})" for col in columns
        )
        return f"Table: data\nColumns:\n{col_descriptions}"

    except Exception as e:
        return f"Could not read schema: {str(e)}"
