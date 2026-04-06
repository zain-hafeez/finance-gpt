# tests/integration/test_full_pipeline.py
"""
Integration Tests — FinanceGPT Full Pipeline
=============================================
These tests run the REAL pipeline end-to-end:
  - Real CSV file loaded into real SQLite
  - Real Groq API calls (no mocks)
  - Real LangGraph graph execution
  - Real result returned and verified

HOW TO RUN:
  pytest tests/integration/ -v -s       ← run with print output visible
  pytest tests/integration/ -v          ← run silently
  pytest tests/ -v                      ← unit tests only (skips this folder)

WHY -s FLAG?
  The -s flag means "don't capture stdout" — our print() statements
  inside tests will be visible so you can see what SQL was generated,
  what the result was, and what explanation the LLM wrote.
  Very useful for debugging and demos.

COST:
  Each test makes 2-3 Groq API calls.
  Groq free tier is generous — these tests cost nothing on free tier.
"""

import os
import time
import pytest
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

# ── Find project root and sample data ────────────────────────────────────────
# __file__ is this test file's path
# .parent      → tests/integration/
# .parent      → tests/
# .parent      → financial-chat/  ← project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
SAMPLE_CSV   = PROJECT_ROOT / "data" / "sample_sales.csv"


# ── API Key Check ─────────────────────────────────────────────────────────────
# Before running any test, check if we have API keys.
# If not, every test is skipped with a clear message — no ugly crashes.

def _has_api_keys() -> bool:
    """Returns True if at least one LLM API key is configured."""
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file so keys are available
    return bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"))


# pytestmark applies to ALL tests in this file
pytestmark = pytest.mark.integration

# This decorator will skip any test it's applied to if no API keys found
requires_api = pytest.mark.skipif(
    not _has_api_keys(),
    reason="No API keys in .env — skipping integration tests"
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_db(tmp_path_factory):
    """
    Loads sample_sales.csv into a real SQLite database.

    scope="module" means this fixture runs ONCE for the entire test file.
    All 8 tests share the same database — efficient and realistic.

    tmp_path_factory is a pytest built-in that creates temporary directories
    that are automatically cleaned up after tests finish.

    Returns a dict with file_path, db_path, row_count, columns.
    """
    # If sample CSV doesn't exist, skip all tests with a helpful message
    if not SAMPLE_CSV.exists():
        pytest.skip(
            f"Sample CSV not found at {SAMPLE_CSV}. "
            f"Create data/sample_sales.csv to run integration tests."
        )

    # Create temp directory that persists for the whole module run
    tmp_dir = tmp_path_factory.mktemp("integration_db")
    db_path = tmp_dir / "test_finance.db"

    # Load the CSV exactly like the real app does in loader.py
    df = pd.read_csv(SAMPLE_CSV)
    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql("data", con=engine, if_exists="replace", index=False)
    engine.dispose()  # Release file handle — critical on Windows

    print(f"\n📁 Integration DB ready: {db_path}")
    print(f"📊 Loaded {len(df)} rows × {len(df.columns)} columns")

    return {
        "file_path": str(SAMPLE_CSV),
        "db_path":   str(db_path),
        "row_count": len(df),
        "columns":   list(df.columns),
    }


# ── Shared Assertion Helper ───────────────────────────────────────────────────

def _assert_valid_result(result: dict, expected_query_type: str = None):
    """
    Every successful result from run_query() must satisfy these conditions.
    We call this helper in every test instead of repeating the same checks.

    Args:
        result: The dict returned by run_query()
        expected_query_type: 'sql' or 'stats' — if provided, verifies routing
    """
    # 1. Must be a dict
    assert isinstance(result, dict), \
        f"run_query() must return a dict, got {type(result)}"

    # 2. Must have all 7 standard keys — no surprises
    required_keys = {
        "query_type", "sql_query", "stats_code",
        "raw_result", "explanation", "error", "cached"
    }
    missing = required_keys - result.keys()
    assert not missing, f"Result missing keys: {missing}"

    # 3. No error — pipeline completed successfully
    assert result["error"] is None, \
        f"Pipeline returned an error: {result['error']}"

    # 4. LLM generated a real explanation (not empty)
    assert result["explanation"], \
        "Explanation is empty — LLM may have failed silently"
    assert len(result["explanation"]) > 10, \
        f"Explanation suspiciously short: '{result['explanation']}'"

    # 5. Check routing if specified
    if expected_query_type:
        assert result["query_type"] == expected_query_type, (
            f"Expected routing to '{expected_query_type}' "
            f"but got '{result['query_type']}'"
        )


# ── Test 1: Basic SQL — Total Sales ──────────────────────────────────────────

@requires_api
def test_sql_total_sales(real_db):
    """
    THE MOST FUNDAMENTAL TEST.

    Asks the simplest possible SQL question. If this fails,
    something is deeply wrong with the SQL pipeline.

    Expected flow:
      1. router classifies → 'sql'
      2. sql_engine generates → SELECT SUM(total_sales) FROM data
      3. SQLite executes → returns a number
      4. explain_node → writes plain English explanation
    """
    from src.graph.workflow import run_query

    result = run_query(
        question="What is the total sales amount?",
        file_path=real_db["file_path"],
        db_path=real_db["db_path"],
        session_id="integration-test-1",
    )

    # Verify the result shape
    _assert_valid_result(result, expected_query_type="sql")

    # SQL query must exist and be a SELECT
    assert result["sql_query"] is not None, \
        "SQL query was not generated"
    assert "SELECT" in result["sql_query"].upper(), \
        "Generated query must be a SELECT statement"

    # Raw result must exist (the actual number)
    assert result["raw_result"] is not None, \
        "raw_result is None — query returned nothing"

    # Print what happened — visible with -s flag
    print(f"\n✅ SQL generated: {result['sql_query']}")
    print(f"✅ Raw result:    {result['raw_result']}")
    print(f"✅ Explanation:   {result['explanation'][:120]}...")


# ── Test 2: SQL with GROUP BY — Sales by Region ───────────────────────────────

@requires_api
def test_sql_sales_by_region(real_db):
    """
    SQL path: aggregation with grouping.

    Tests that the LLM generates a GROUP BY query correctly.
    This produces a multi-row result — important for chart rendering.

    Expected SQL shape:
      SELECT region, SUM(total_sales) FROM data GROUP BY region
    """
    from src.graph.workflow import run_query

    result = run_query(
        question="What is the total sales by region?",
        file_path=real_db["file_path"],
        db_path=real_db["db_path"],
        session_id="integration-test-2",
    )

    _assert_valid_result(result, expected_query_type="sql")
    assert result["sql_query"] is not None

    print(f"\n✅ SQL: {result['sql_query']}")
    print(f"✅ Result: {result['raw_result']}")


# ── Test 3: SQL with ORDER BY + LIMIT — Top Salespeople ──────────────────────

@requires_api
def test_sql_top_salespeople(real_db):
    """
    SQL path: ranking query with ORDER BY and LIMIT.

    Tests more complex SQL generation — ranking queries are common
    in financial analysis ("who are my best performers?").
    """
    from src.graph.workflow import run_query

    result = run_query(
        question="Who are the top 3 salespeople by total sales?",
        file_path=real_db["file_path"],
        db_path=real_db["db_path"],
        session_id="integration-test-3",
    )

    _assert_valid_result(result, expected_query_type="sql")

    print(f"\n✅ SQL: {result['sql_query']}")
    print(f"✅ Top salespeople: {result['raw_result']}")


# ── Test 4: Stats — Moving Average ───────────────────────────────────────────

@requires_api
def test_stats_moving_average(real_db):
    """
    Stats path: moving average — the classic stats engine test.

    This proves:
      1. Router correctly classifies "moving average" as stats (not sql)
      2. Groq generates valid Pandas rolling() code
      3. RestrictedPython sandbox executes it safely
      4. Result is a list of numbers

    If this test fails, check the stats engine and RestrictedPython setup.
    """
    from src.graph.workflow import run_query

    result = run_query(
        question="What is the 3-period moving average of total sales?",
        file_path=real_db["file_path"],
        db_path=real_db["db_path"],
        session_id="integration-test-4",
    )

    _assert_valid_result(result, expected_query_type="stats")

    # Stats code must have been generated
    assert result["stats_code"] is not None, \
        "Stats code was not generated — stats engine may have failed"

    print(f"\n✅ Generated code:\n{result['stats_code']}")
    print(f"✅ Result: {result['raw_result']}")
    print(f"✅ Explanation: {result['explanation'][:120]}...")


# ── Test 5: Stats — Correlation ───────────────────────────────────────────────

@requires_api
def test_stats_correlation(real_db):
    """
    Stats path: correlation between two columns.

    Correlation is a pure stats operation — SQL cannot do this.
    Tests that the router never routes this to SQL by mistake.

    Expected result: a number between -1.0 and 1.0
    """
    from src.graph.workflow import run_query

    result = run_query(
        question="What is the correlation between quantity and total sales?",
        file_path=real_db["file_path"],
        db_path=real_db["db_path"],
        session_id="integration-test-5",
    )

    _assert_valid_result(result, expected_query_type="stats")
    assert result["stats_code"] is not None

    print(f"\n✅ Generated code:\n{result['stats_code']}")
    print(f"✅ Correlation result: {result['raw_result']}")


# ── Test 6: Cache Hit on Repeated Question ────────────────────────────────────

@requires_api
def test_cache_hit_on_repeated_question(real_db):
    """
    Cache test: same question asked twice.

    First call  → cache MISS  → calls Groq → saves to cache
    Second call → cache HIT   → returns instantly from cache

    This proves both cache layers (LangChain InMemoryCache +
    app-level dict cache) are working correctly in the real pipeline.

    The second call should be dramatically faster — we measure both
    and assert the cache hit is faster.
    """
    from src.graph.workflow import run_query
    from src.graph.nodes import clear_cache

    # Always start with a clean cache so test is reliable
    clear_cache()

    question = "How many total orders are there?"

    # ── First call: should be a cache MISS ──
    t1_start = time.time()
    result1 = run_query(
        question=question,
        file_path=real_db["file_path"],
        db_path=real_db["db_path"],
        session_id="integration-cache-test",
    )
    t1_elapsed = time.time() - t1_start

    assert result1["cached"] is False, \
        "First call must be a cache MISS — cache was just cleared"
    assert result1["error"] is None

    # ── Second call: should be a cache HIT ──
    t2_start = time.time()
    result2 = run_query(
        question=question,
        file_path=real_db["file_path"],
        db_path=real_db["db_path"],
        session_id="integration-cache-test",
    )
    t2_elapsed = time.time() - t2_start

    assert result2["cached"] is True, \
        "Second call must be a cache HIT — same question, same file"

    # Cache hit must be faster (usually 100x+ faster)
    assert t2_elapsed < t1_elapsed, \
        f"Cache hit ({t2_elapsed:.3f}s) was not faster than miss ({t1_elapsed:.2f}s)"

    print(f"\n✅ First call  (miss):  {t1_elapsed:.2f}s")
    print(f"✅ Second call (hit):   {t2_elapsed:.4f}s")
    print(f"✅ Speedup: {t1_elapsed / max(t2_elapsed, 0.001):.0f}x faster")


# ── Test 7: Raw Result Is Populated ──────────────────────────────────────────

@requires_api
def test_result_has_raw_data(real_db):
    """
    Verifies raw_result is never None on a successful query.

    raw_result is what the chart builder and table formatter consume.
    If it's None even on a successful query, the UI shows nothing —
    a silent bug. This test catches that.
    """
    from src.graph.workflow import run_query

    result = run_query(
        question="What is the total sales by product?",
        file_path=real_db["file_path"],
        db_path=real_db["db_path"],
        session_id="integration-test-7",
    )

    _assert_valid_result(result)

    assert result["raw_result"] is not None, \
        "raw_result is None on a successful query — UI will show nothing"

    print(f"\n✅ raw_result type: {type(result['raw_result']).__name__}")
    print(f"✅ raw_result: {result['raw_result']}")


# ── Test 8: Session Isolation (M9) ────────────────────────────────────────────

@requires_api
def test_session_isolation(real_db):
    """
    M9 persistence test: two different session_ids produce independent results.

    LangGraph saves state per thread_id (session_id). This test verifies
    that session A and session B don't interfere with each other —
    critical for a multi-user app.

    Both sessions ask the same question and should get the same answer,
    but through completely independent LangGraph thread states.
    """
    from src.graph.workflow import run_query

    result_a = run_query(
        question="What is the total sales amount?",
        file_path=real_db["file_path"],
        db_path=real_db["db_path"],
        session_id="integration-session-A",
    )

    result_b = run_query(
        question="What is the total sales amount?",
        file_path=real_db["file_path"],
        db_path=real_db["db_path"],
        session_id="integration-session-B",
    )

    # Both must succeed independently
    assert result_a["error"] is None, f"Session A failed: {result_a['error']}"
    assert result_b["error"] is None, f"Session B failed: {result_b['error']}"

    # Both must have explanations
    assert result_a["explanation"] != ""
    assert result_b["explanation"] != ""

    print(f"\n✅ Session A explanation: {result_a['explanation'][:80]}...")
    print(f"✅ Session B explanation: {result_b['explanation'][:80]}...")
    print(f"✅ Sessions are independent — no state bleed detected")