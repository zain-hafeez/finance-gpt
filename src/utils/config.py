# src/utils/config.py
# ------------------------------------------------------------
# Purpose: Loads all environment variables from the .env file
# and makes them available to the rest of the application.
# This is the single place where configuration is managed.
# ------------------------------------------------------------
# STATUS: Placeholder — full implementation in Module 2

import os
from dotenv import load_dotenv

# load_dotenv() reads the .env file and loads all the key=value
# pairs into the environment so os.getenv() can access them.
load_dotenv()

# Retrieve each setting. The second argument is the default value
# used if the variable isn't found in .env.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
EXECUTION_TIMEOUT_SECONDS = int(os.getenv("EXECUTION_TIMEOUT_SECONDS", "5"))
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
DB_PATH = os.getenv("DB_PATH", "finance.db")
CHECKPOINTS_DB = os.getenv("CHECKPOINTS_DB", "checkpoints.db")