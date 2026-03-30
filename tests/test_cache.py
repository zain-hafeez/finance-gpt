# tests/test_cache.py
"""
Module 7 — Query Caching Tests
================================
14 tests covering:
  - cache_setup.py: LangChain InMemoryCache setup
  - cache_node: hit/miss behavior
  - cache_write_node: result storage
  - clear_cache / get_cache_size utilities
  - workflow integration: cached=True skips pipeline
  - cache key isolation: different files don't share cache
  - CACHE_ENABLED=false: caching can be disabled

All LLM calls are mocked — these tests run offline, no API keys needed.
"""

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(question='What is the total sales?', file_path='/data/test.csv',
                db_path='test.db', cached=False, error=None, raw_result=None,
                explanation='', query_type='sql', sql_query=None, stats_code=None):
    """Build a minimal FinanceGPTState dict for testing."""
    return {
        'query':       question,
        'file_path':   file_path,
        'db_path':     db_path,
        'session_id':  'test',
        'query_type':  query_type,
        'cached':      cached,
        'sql_query':   sql_query,
        'sql_valid':   True,
        'stats_code':  stats_code,
        'raw_result':  raw_result,
        'explanation': explanation,
        'error':       error,
    }


# ---------------------------------------------------------------------------
# Fixture — reset cache before each test so tests don't interfere
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_cache():
    """Clear the app-level cache before every test."""
    from src.graph.nodes import clear_cache
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Group 1: cache_setup.py — LangChain InMemoryCache (3 tests)
# ---------------------------------------------------------------------------

class TestCacheSetup:

    def test_setup_cache_returns_instance_when_enabled(self):
        """setup_cache() should return an InMemoryCache object when CACHE_ENABLED=True."""
        from src.utils.cache_setup import setup_cache
        with patch('src.utils.cache_setup.CACHE_ENABLED', True):
            instance = setup_cache()
        assert instance is not None

    def test_setup_cache_returns_none_when_disabled(self):
        """setup_cache() should return None when CACHE_ENABLED=False."""
        from src.utils.cache_setup import setup_cache
        with patch('src.utils.cache_setup.CACHE_ENABLED', False):
            result = setup_cache()
        assert result is None

    def test_get_cache_instance_returns_same_object(self):
        """get_cache_instance() should return the instance set by setup_cache()."""
        from src.utils.cache_setup import setup_cache, get_cache_instance
        with patch('src.utils.cache_setup.CACHE_ENABLED', True):
            instance = setup_cache()
        assert get_cache_instance() is instance


# ---------------------------------------------------------------------------
# Group 2: _make_cache_key — key generation (2 tests)
# ---------------------------------------------------------------------------

class TestCacheKey:

    def test_same_question_same_file_produces_same_key(self):
        """Same inputs must always produce the same key (deterministic)."""
        from src.graph.nodes import _make_cache_key
        key1 = _make_cache_key('total sales', '/data/jan.csv')
        key2 = _make_cache_key('total sales', '/data/jan.csv')
        assert key1 == key2

    def test_same_question_different_file_produces_different_key(self):
        """Different files must NOT share cache keys even for identical questions."""
        from src.graph.nodes import _make_cache_key
        key1 = _make_cache_key('total sales', '/data/jan.csv')
        key2 = _make_cache_key('total sales', '/data/feb.csv')
        assert key1 != key2


# ---------------------------------------------------------------------------
# Group 3: cache_node — hit and miss behavior (3 tests)
# ---------------------------------------------------------------------------

class TestCacheNode:

    def test_cache_miss_returns_cached_false(self):
        """On a cache miss, cache_node should return {'cached': False}."""
        from src.graph.nodes import cache_node
        state = _make_state()
        result = cache_node(state)
        assert result == {'cached': False}

    def test_cache_hit_returns_stored_result_with_cached_true(self):
        """After a write, the same question should return the cached data."""
        from src.graph.nodes import cache_node, cache_write_node

        # First: run a completed state through cache_write_node to populate cache
        completed_state = _make_state(
            raw_result='[(4065.0,)]',
            explanation='Total sales is $4065.',
            query_type='sql',
            sql_query='SELECT SUM(total_sales) FROM data',
        )
        cache_write_node(completed_state)

        # Now cache_node should return a hit
        lookup_state = _make_state()
        result = cache_node(lookup_state)

        assert result['cached'] is True
        assert result['explanation'] == 'Total sales is $4065.'
        assert result['sql_query'] == 'SELECT SUM(total_sales) FROM data'

    def test_cache_hit_different_file_is_miss(self):
        """A cache hit for file A must NOT be returned for file B."""
        from src.graph.nodes import cache_node, cache_write_node

        # Write to cache for file A
        state_a = _make_state(file_path='/data/a.csv', explanation='Result A')
        cache_write_node(state_a)

        # Query with file B — must be a miss
        state_b = _make_state(file_path='/data/b.csv')
        result = cache_node(state_b)
        assert result == {'cached': False}


# ---------------------------------------------------------------------------
# Group 4: cache_write_node — storage behavior (3 tests)
# ---------------------------------------------------------------------------

class TestCacheWriteNode:

    def test_write_increases_cache_size(self):
        """After writing, cache size should increase by 1."""
        from src.graph.nodes import cache_write_node, get_cache_size
        assert get_cache_size() == 0
        state = _make_state(explanation='Test result', raw_result='some data')
        cache_write_node(state)
        assert get_cache_size() == 1

    def test_error_state_is_not_cached(self):
        """States with errors should NOT be written to cache."""
        from src.graph.nodes import cache_write_node, get_cache_size
        error_state = _make_state(error='LLM call failed')
        cache_write_node(error_state)
        assert get_cache_size() == 0

    def test_duplicate_write_does_not_grow_cache(self):
        """Writing the same question twice should keep cache size at 1."""
        from src.graph.nodes import cache_write_node, get_cache_size
        state = _make_state(explanation='Result')
        cache_write_node(state)
        cache_write_node(state)  # same key → overwrites
        assert get_cache_size() == 1


# ---------------------------------------------------------------------------
# Group 5: clear_cache + get_cache_size utilities (2 tests)
# ---------------------------------------------------------------------------

class TestCacheUtilities:

    def test_clear_cache_empties_the_cache(self):
        """clear_cache() should remove all entries."""
        from src.graph.nodes import cache_write_node, clear_cache, get_cache_size
        cache_write_node(_make_state(explanation='data'))
        assert get_cache_size() == 1
        clear_cache()
        assert get_cache_size() == 0

    def test_get_cache_size_returns_correct_count(self):
        """get_cache_size() should return the number of cached entries."""
        from src.graph.nodes import cache_write_node, get_cache_size
        cache_write_node(_make_state(question='Q1', explanation='R1'))
        cache_write_node(_make_state(question='Q2', explanation='R2'))
        assert get_cache_size() == 2


# ---------------------------------------------------------------------------
# Group 6: workflow integration — route_after_cache (1 test)
# ---------------------------------------------------------------------------

class TestRouteAfterCache:

    def test_cached_true_routes_to_end(self):
        """_route_after_cache should return 'end' when cached=True."""
        from src.graph.workflow import _route_after_cache
        state = _make_state(cached=True)
        assert _route_after_cache(state) == 'end'

    def test_cached_false_routes_to_router(self):
        """_route_after_cache should return 'router' when cached=False."""
        from src.graph.workflow import _route_after_cache
        state = _make_state(cached=False)
        assert _route_after_cache(state) == 'router'