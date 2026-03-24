# src/utils/config.py
# ------------------------------------------------------------
# PURPOSE: This is the central configuration file for the
# entire application. It reads settings from the .env file
# and makes them available as Python variables.
#
# WHY THIS EXISTS: Instead of calling os.getenv("MAX_FILE_SIZE_MB")
# in 10 different files, every file imports from here. If a
# setting name changes, we update it in ONE place only.
# This is the "single source of truth" pattern.
# ------------------------------------------------------------

import os
from pathlib import Path
from dotenv import load_dotenv

# load_dotenv() finds your .env file and loads all key=value
# pairs into the environment. Call this once at app startup.
load_dotenv()

# --- LLM Provider Keys ---
# These will be filled in when we work on Module 2.
# For now they'll be empty strings, which is fine for M1.
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# --- File Validation Settings ---
# Maximum file size allowed for upload, in bytes.
# We read MB from .env and convert to bytes here so the
# validator can compare directly with the file's actual size.
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024  # 10MB = 10,485,760 bytes

# Allowed file extensions. Only CSV and Excel files accepted.
ALLOWED_EXTENSIONS: tuple = (".csv", ".xlsx")

# --- Database Settings ---
# Path to the SQLite database file that stores uploaded data.
# This file is created automatically when the first CSV is uploaded.
DB_PATH: str = os.getenv("DB_PATH", "finance.db")

# Path to the SQLite file used for LangGraph conversation history.
CHECKPOINTS_DB: str = os.getenv("CHECKPOINTS_DB", "checkpoints.db")

# --- Execution Settings ---
EXECUTION_TIMEOUT_SECONDS: int = int(os.getenv("EXECUTION_TIMEOUT_SECONDS", "5"))
CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"

# --- Path Utilities ---
# ROOT_DIR points to the project root folder (financial-chat/).
# Using Path makes file paths work correctly on both Windows and Linux.
ROOT_DIR: Path = Path(__file__).parent.parent.parent
DATA_DIR: Path = ROOT_DIR / "data"