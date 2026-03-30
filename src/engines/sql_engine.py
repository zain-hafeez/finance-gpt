# src/engines/sql_engine.py
# Module 3 — Text-to-SQL Engine
# LangChain 1.x compatible — create_sql_query_chain was removed in 1.x
# We build the SQL generation chain manually using SQLDatabase + direct LLM call
# This is more transparent and gives us full control over the prompt

import logging
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage
from src.engines.sql_validator import validate_sql, get_validation_error

logger = logging.getLogger(__name__)

# The prompt we send to the LLM to generate SQL.
# {schema} and {question} are filled in at runtime.
SQL_GENERATION_PROMPT = """You are a SQLite expert. Given the table schema below, write a syntactically correct SQLite SELECT query to answer the question.

SCHEMA:
{schema}

RULES:
- Write ONLY the SQL query, nothing else
- No explanations, no markdown, no code fences
- Use only column names that exist in the schema above
- Always use table name: data
- Only SELECT statements are allowed

QUESTION: {question}

SQL QUERY:"""


def run_sql_query(question: str, db_path: str, llm) -> dict:
    """
    Convert a plain-English question into SQL, validate it, execute it,
    and return the result.

    Args:
        question: The user's question in plain English.
        db_path:  Absolute path to the SQLite database file (from load_data()).
        llm:      A LangChain chat model object (from get_llm('sql')).

    Returns:
        dict with keys: success, sql, result, row_count, error
    """
    # Step 1: Connect to SQLite and read the schema
    try:
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        schema = db.get_table_info()
        logger.info("Connected to DB. Schema loaded.")
    except Exception as e:
        logger.error("Failed to connect to SQLite at %s: %s", db_path, str(e))
        return _error_result("", f"Could not connect to database: {str(e)}")

    # Step 2: Build the prompt with the actual schema and question injected
    prompt = SQL_GENERATION_PROMPT.format(
        schema=schema,
        question=question,
    )

    # Step 3: Call the LLM directly — ask it to write SQL
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_sql = response.content
        logger.info("LLM raw response: %s", raw_sql)
    except Exception as e:
        logger.error("LLM call failed: %s", str(e))
        return _error_result("", f"LLM failed to generate SQL: {str(e)}")

    # Step 4: Clean the response
    # Sometimes the LLM adds markdown fences like ```sql ... ``` or explanation text
    sql = _clean_sql_output(raw_sql)
    logger.info("Cleaned SQL: %s", sql)

    # Step 5: Validate — block DROP, DELETE, INSERT, UPDATE, etc.
    if not validate_sql(sql):
        reason = get_validation_error(sql)
        logger.warning("SQL failed validation: %s | Reason: %s", sql, reason)
        return _error_result(sql, f"Generated query failed safety check: {reason}")

    # Step 6: Execute the validated SQL
    try:
        result = db.run(sql)
        row_count = len(result.strip().split("\n")) if result.strip() else 0
        logger.info("SQL executed successfully. Rows: %s", row_count)
        return {
            "success": True,
            "sql": sql,
            "result": result,
            "row_count": row_count,
            "error": None,
        }
    except Exception as e:
        logger.error("SQL execution failed: %s | Query: %s", str(e), sql)
        return _error_result(sql, f"Query execution failed: {str(e)}")


def _clean_sql_output(raw: str) -> str:
    """
    Strip everything the LLM added around the SQL.
    Handles markdown fences, explanation text, leading/trailing whitespace.

    Examples of what this cleans:
```sql\nSELECT * FROM data\n```  →  SELECT * FROM data
        Here is the query:\nSELECT ...   →  SELECT ...
        SELECT * FROM data;              →  SELECT * FROM data
    """
    sql = raw.strip()

    # Remove markdown code fences if present
    if "```" in sql:
        # Extract content between fences
        parts = sql.split("```")
        # parts[1] is the content inside fences (may start with 'sql\n')
        if len(parts) >= 2:
            sql = parts[1].strip()
            # Remove language identifier like 'sql' at the start
            if sql.lower().startswith("sql"):
                sql = sql[3:].strip()

    # If the LLM added explanation before the SQL, find where SELECT starts
    select_pos = sql.upper().find("SELECT")
    if select_pos == -1:
        # No SELECT found at all — return as-is, validator will catch it
        return sql
    if select_pos > 0:
        # There's text before SELECT — discard it
        sql = sql[select_pos:]

    # Remove trailing semicolon (validator blocks it, cleaner without it)
    sql = sql.strip().rstrip(";").strip()

    return sql


def _error_result(sql: str, error_message: str) -> dict:
    """Standardized error response — same shape as success so callers never get KeyError."""
    return {
        "success": False,
        "sql": sql,
        "result": None,
        "row_count": 0,
        "error": error_message,
    }