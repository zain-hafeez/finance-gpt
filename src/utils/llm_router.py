# src/utils/llm_router.py
# Module 2 — LLM Router
#
# Single entry point for ALL LLM calls in this project.
# Primary provider: Groq (fast inference, free tier)
# Fallback provider: OpenAI GPT-4o-mini (when Groq rate-limits or fails)
#
# Usage from any other module:
#   from src.utils.llm_router import get_llm
#   llm = get_llm('sql')           # or 'stats', 'routing', 'explanation', 'general'
#   response = llm.invoke("your prompt here")

import logging
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from src.utils.config import GROQ_API_KEY, OPENAI_API_KEY

# Logger — uses Python's built-in logging module.
# Production apps use logging, not print(). You'll see messages in
# the terminal like: WARNING:src.utils.llm_router:Groq unavailable...
logger = logging.getLogger(__name__)


# ── Model name constants ────────────────────────────────────────────────────
# Defined once here. If Groq renames a model or you want to upgrade,
# you change it in ONE place and the entire project updates automatically.
GROQ_REASONING   = "llama-3.3-70b-versatile"     # SQL gen, routing, stats code gen
GROQ_EXPLANATION = "llama-3.3-70b-versatile"  # Plain-English result explanation
OPENAI_FALLBACK  = "gpt-4o-mini"         # Fallback for everything


def get_llm(task: str = "general") -> BaseChatModel:
    """
    Return the correct LangChain chat model for the given task.

    Args:
        task: What this LLM call is for. One of:
              'sql'         → SQL query generation
              'stats'       → Pandas/NumPy code generation
              'routing'     → Query classification (sql vs stats)
              'explanation' → Plain English result explanation
              'general'     → Default, same as sql/stats/routing

    Returns:
        A LangChain BaseChatModel object. Call .invoke() on it to get a response.
        Will be ChatGroq if Groq is available, ChatOpenAI if Groq fails.

    The fallback is invisible to the caller — they always get a working LLM back.
    """
    # Pick which Groq model to use based on the task
    groq_model = GROQ_EXPLANATION if task == "explanation" else GROQ_REASONING

    try:
        # Guard: make sure we actually have a key loaded from .env
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in .env")

        # Build and return the Groq chat model object.
        # max_retries=2: LangChain will retry twice on transient errors
        # before raising an exception that our except block catches.
        return ChatGroq(
            model=groq_model,
            api_key=GROQ_API_KEY,
            temperature=0,
            max_retries=2,
        )

    except Exception as e:
        # Log what happened so you can see it in the terminal during debugging.
        # This is NOT shown to the user — just visible in your VS Code terminal.
        logger.warning(
            "Groq unavailable for task=%s (reason: %s). Falling back to OpenAI.",
            task,
            str(e),
        )
        return _get_openai_fallback()


def _get_openai_fallback() -> BaseChatModel:
    """
    Return OpenAI GPT-4o-mini as a fallback LLM.

    The leading underscore in the name is a Python convention meaning
    'private — only call this from inside this file'. Other modules
    should always call get_llm(), never this function directly.

    Raises RuntimeError if OpenAI key is also missing — at that point
    we cannot recover and must tell the developer clearly what's wrong.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Both GROQ_API_KEY and OPENAI_API_KEY are missing from .env. "
            "At least one LLM provider must be configured."
        )

    return ChatOpenAI(
        model=OPENAI_FALLBACK,
        api_key=OPENAI_API_KEY,
        temperature=0,
        max_retries=2,
    )