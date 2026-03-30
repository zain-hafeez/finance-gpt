# tests/test_llm_router.py
# Module 2 — 3 automated tests for get_llm()
# All LLM calls are mocked — no real API calls, no cost, runs in < 1 second

from unittest.mock import patch, MagicMock
from src.utils.llm_router import get_llm, GROQ_REASONING, GROQ_EXPLANATION


class TestGetLlm:

    def test_sql_task_returns_groq_with_reasoning_model(self):
        """
        get_llm('sql') should build a ChatGroq using llama3-70b-8192.
        We patch ChatGroq so no real API call is made.
        """
        with patch("src.utils.llm_router.ChatGroq") as mock_groq:
            mock_instance = MagicMock()
            mock_groq.return_value = mock_instance

            result = get_llm("sql")

            # ChatGroq should have been called once
            mock_groq.assert_called_once()
            # And it should have used the reasoning model
            kwargs = mock_groq.call_args.kwargs
            assert kwargs["model"] == GROQ_REASONING
            assert kwargs["temperature"] == 0
            # The return value should be the mock ChatGroq instance
            assert result is mock_instance

    def test_explanation_task_returns_groq_with_mixtral_model(self):
        """
        get_llm('explanation') should use mixtral-8x7b-32768, not llama3.
        """
        with patch("src.utils.llm_router.ChatGroq") as mock_groq:
            mock_instance = MagicMock()
            mock_groq.return_value = mock_instance

            result = get_llm("explanation")

            kwargs = mock_groq.call_args.kwargs
            assert kwargs["model"] == GROQ_EXPLANATION
            assert result is mock_instance

    def test_falls_back_to_openai_when_groq_raises_any_error(self):
        """
        When ChatGroq raises ANY exception (rate limit, network, wrong key),
        get_llm() must silently return a ChatOpenAI instance instead.
        The caller should never see the Groq error.
        """
        with patch("src.utils.llm_router.ChatGroq") as mock_groq, \
             patch("src.utils.llm_router.ChatOpenAI") as mock_openai:

            # Force Groq to blow up — simulates rate limit or outage
            mock_groq.side_effect = Exception("Simulated: Groq rate limit exceeded")

            mock_openai_instance = MagicMock()
            mock_openai.return_value = mock_openai_instance

            result = get_llm("sql")

            # Result must be the OpenAI mock, not Groq
            assert result is mock_openai_instance
            mock_openai.assert_called_once()