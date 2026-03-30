# test_m7_manual.py
"""
M7 Smoke Test — Run this manually to verify caching works end-to-end.
Requires real API keys in .env and data/sample_sales.csv to exist.

Run with: python test_m7_manual.py
"""

import time
from src.data.loader import load_data
from src.graph.workflow import run_query
from src.graph.nodes import get_cache_size, clear_cache

# --- Setup ---
print("=" * 60)
print("M7 SMOKE TEST — Query Caching")
print("=" * 60)

FILE = 'data/sample_sales.csv'
DB   = 'finance.db'

# Load data into SQLite
load_data(FILE, DB)
clear_cache()

QUESTION = 'What is the total sales amount?'

# --- First call: cache miss ---
print(f"\n[1] FIRST CALL (cache miss expected)")
print(f"    Question: {QUESTION}")
t0 = time.time()
result1 = run_query(QUESTION, FILE, DB)
t1 = time.time()

print(f"    cached:      {result1['cached']}")
print(f"    explanation: {result1['explanation'][:80]}...")
print(f"    time:        {t1 - t0:.2f}s")
print(f"    cache size:  {get_cache_size()}")

assert result1['cached'] is False, "First call must be a cache miss"
assert result1['success'] is True

# --- Second call: cache hit ---
print(f"\n[2] SECOND CALL (cache hit expected)")
t2 = time.time()
result2 = run_query(QUESTION, FILE, DB)
t3 = time.time()

elapsed = t3 - t2
print(f"    cached:      {result2['cached']}")
print(f"    explanation: {result2['explanation'][:80]}...")
print(f"    time:        {elapsed:.3f}s")
print(f"    cache size:  {get_cache_size()}")

assert result2['cached'] is True,      "Second call must be a cache hit"
assert elapsed < 0.5,                   f"Cache hit took {elapsed:.3f}s — should be <0.5s"
assert result2['explanation'] == result1['explanation'], "Explanations must match"

# --- Different question: still a miss ---
print(f"\n[3] DIFFERENT QUESTION (cache miss expected)")
result3 = run_query('Show me sales by region', FILE, DB)
print(f"    cached:     {result3['cached']}")
assert result3['cached'] is False, "Different question must be a cache miss"

print(f"\n{'=' * 60}")
print("ALL SMOKE TESTS PASSED ✓")
print(f"Cache hit response time: {elapsed:.3f}s (target: <0.5s)")
print(f"Cache entries stored: {get_cache_size()}")
print("=" * 60)