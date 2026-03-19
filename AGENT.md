# AGENT.md — FinanceGPT Project Master Reference (v2 — Hybrid Architecture)

> **ARCHITECTURE FINALIZED — NO FURTHER CHANGES TO CORE DESIGN**
> This file is the single source of truth for the entire FinanceGPT project.
> Claude reads this at the start of every session to restore full context.
> Update ONLY the Module Status Tracker after completing each module.

---

## PROJECT OVERVIEW

**Name:** FinanceGPT
**Type:** Portfolio Project — Production-Grade AI Engineering
**Tagline:** AI-Powered Financial Data Analysis with Natural Language
**Core Feature:** Upload CSV/Excel → Ask questions in plain English → Get exact results + charts + insights
**IDE:** VS Code
**Status:** 🟡 Architecture Finalized — Development Starting

**Key Differentiator:** Hybrid query engine — LangGraph routes each question to either a Text-to-SQL engine (for exact calculations) or a Pandas/NumPy engine (for statistical analysis). Right tool for the right job.

---

## WHY THIS ARCHITECTURE (THE STORY FOR INTERVIEWS)

> *"I built a hybrid query engine. A LangGraph router classifies every question and sends it to either Text-to-SQL for exact aggregations or a Pandas/NumPy stats engine for forecasting and correlations. SQL path needs zero sandboxing — SQLite is inherently safe. Stats path uses RestrictedPython. The LLM never touches arbitrary execution on the SQL path, so math is always exact. This is how production analytics systems are actually built."*

This answer gets you hired.

---

## FINALIZED DECISIONS (ALL LOCKED IN — DO NOT REOPEN)

| # | Decision | Choice | Reason |
|---|----------|--------|--------|
| D1 | LLM Primary | Groq (Llama3-70b / Mixtral-8x7b) | Fast inference, free tier, handles SQL gen + explanation + routing |
| D2 | LLM Fallback | OpenAI GPT-4o-mini | Reliable, affordable, handles everything |
| D3 | Query Engine A | Text-to-SQL → SQLite | Exact math, deterministic, inherently safe |
| D4 | Query Engine B | Pandas + NumPy (stats node) | Forecasting, correlation, moving averages |
| D5 | Query Router | LangGraph conditional node + Groq classification | Routes SQL vs Stats path automatically |
| D6 | Code Safety | RestrictedPython (Stats node ONLY) | SQL path needs none; stats path sandboxed |
| D7 | Orchestration | LangGraph StateGraph | Modern AI workflow, state management, conditional routing |
| D8 | Caching | LangChain InMemoryCache | 3 lines, built-in, maintained |
| D9 | Visualization | Plotly | Interactive charts from query results |
| D10 | UI | Gradio 4.x | Best for ML/AI demos |
| D11 | State Persistence | LangGraph + SQLite checkpointer | Conversation history across sessions |
| D12 | Deployment | HuggingFace Spaces + Dockerfile | Free public URL for portfolio |
| D13 | Observability | Langfuse (Phase 2 only) | LLM tracing — added after core works |
| D14 | IDE | VS Code | — |
| D15 | Vector Store | ❌ NONE | Not needed — structured tabular data, direct DB access |
| D16 | Embeddings | ❌ NONE | Not needed — no RAG, no semantic search required |
| D17 | Qwen / Alibaba | ❌ REMOVED | Flaky LangChain integration, Groq handles SQL gen fine |

---

## TECHNOLOGY STACK (FINAL)

| Layer | Technology | Why |
|-------|------------|-----|
| UI | Gradio 4.x | Best for ML/AI demos, native Plotly support |
| LLM Primary | Groq (Llama3-70b + Mixtral-8x7b) | Fast, free tier, SQL gen + reasoning + routing |
| LLM Fallback | OpenAI GPT-4o-mini | Reliable fallback when Groq rate-limited |
| Orchestration | LangGraph StateGraph | Conditional routing, state management, modern standard |
| SQL Engine | LangChain `create_sql_query_chain` + SQLite | Text-to-SQL, exact math, zero security risk |
| Stats Engine | Pandas + NumPy + SciPy | Forecasting, correlation, moving averages, anomalies |
| Code Safety | RestrictedPython (stats path only) | Sandbox for stats node; SQL path needs none |
| Caching | LangChain InMemoryCache | Avoid redundant LLM calls |
| Visualization | Plotly | Interactive charts from results |
| Data Loading | Pandas + openpyxl | CSV/Excel → DataFrame → SQLite |
| State | LangGraph + SQLite checkpointer | Conversation persistence |
| Testing | pytest | Standard Python testing |
| Container | Docker | Deployment portability |
| Hosting | HuggingFace Spaces | Free public deployment |

---

## PROJECT STRUCTURE (FINAL)

```
financial-chat/
├── app.py                          # Main Gradio app entry point
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── validator.py            # File upload validation (CUSTOM)
│   │   └── loader.py               # CSV/Excel → DataFrame → SQLite (CUSTOM)
│   ├── engines/
│   │   ├── sql_engine.py           # create_sql_query_chain + SQLite execution
│   │   ├── stats_engine.py         # Pandas/NumPy stats operations
│   │   └── sql_validator.py        # SELECT-only whitelist (blocks DROP/DELETE/etc.)
│   ├── graph/
│   │   ├── workflow.py             # LangGraph StateGraph definition
│   │   ├── nodes.py                # All node functions
│   │   ├── router.py               # Query classification logic (sql vs stats)
│   │   └── state.py                # State schema (TypedDict — serializable only)
│   ├── security/
│   │   └── restricted_exec.py      # RestrictedPython wrapper (stats node only)
│   └── utils/
│       ├── config.py               # Env vars + app configuration
│       ├── llm_router.py           # Groq primary / OpenAI fallback logic
│       ├── cache_setup.py          # LangChain InMemoryCache setup
│       └── prompts.py              # All prompt templates in one place
├── tests/
│   ├── test_validator.py
│   ├── test_sql_engine.py
│   ├── test_stats_engine.py
│   ├── test_router.py
│   ├── test_workflow.py
│   └── integration/
│       └── test_full_pipeline.py
├── data/
│   ├── sample_sales.csv
│   └── sample_expenses.xlsx
├── docs/
│   ├── ARCHITECTURE.md
│   └── DEPLOYMENT.md
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── README.md
└── AGENT.md                        # This file
```

---

## COMPLETE DATA FLOW

```
User Uploads CSV/Excel
        ↓
[M1: validate_file]
  - size check (max 10MB)
  - format check (csv/xlsx only)
  - not empty, readable
        ↓
[M1: load_data]
  - Pandas read_csv / read_excel
  - df.to_sql() → SQLite (finance.db)     ← permanent query target
  - Keep df in memory too                  ← stats engine uses this
        ↓
User Asks a Question
        ↓
[M6: check_cache]
  - LangChain InMemoryCache lookup
  - Cache hit → return immediately
        ↓ (cache miss)
[M5: query_router]  ← Groq classifies in ~200ms
  - "sql"   → aggregation, filter, sort, group, count, sum, top-N
  - "stats" → forecast, trend, correlation, anomaly, moving average
        ↓               ↓
[M3: sql_node]    [M4: stats_node]
  create_sql_         Pandas/NumPy
  query_chain()       RestrictedPython
  → SQL query         → stats result
  → validate SQL      
  → execute SQLite    
  → exact result      
        ↓               ↓
        └───────┬────────┘
                ↓
[M7: visualize]
  - Plotly chart if data is chartable
  - Line, bar, pie based on result shape
                ↓
[M7: explain]
  - Groq Mixtral-8x7b
  - Plain English explanation of result
                ↓
[M8: Gradio UI display]
  - Explanation text
  - Plotly chart
  - SQL query shown (transparency)
  - Raw result table
```

---

## STATE OBJECT (LangGraph — Serializable Only)

```python
# src/graph/state.py
from typing import TypedDict, Optional, Any

class FinanceGPTState(TypedDict):
    # Input
    file_path: str                    # Path to uploaded file
    query: str                        # User's question

    # Routing
    query_type: str                   # "sql" or "stats"
    cached: bool                      # Whether result came from cache

    # SQL Path
    sql_query: Optional[str]          # Generated SQL
    sql_valid: bool                   # Passed whitelist check

    # Stats Path
    stats_code: Optional[str]         # Generated Pandas/NumPy code

    # Results
    raw_result: Optional[Any]         # Table/dict from execution
    chart_json: Optional[str]         # Plotly figure as JSON string
    explanation: str                  # Plain English answer

    # Meta
    error: Optional[str]              # Error message if any step fails
    session_id: str                   # For SQLite checkpointer
```

**Critical Rule:** No DataFrames, no SQLite connections, no Plotly objects in state. These are heavy/non-serializable. Pass file paths and JSON strings only. Reconstruct objects inside nodes.

---

## MODULE BREAKDOWN

### MODULE 0 — Project Setup ✅ COMPLETE
**Goal:** Working skeleton, nothing breaks
**Status:** 🟢 Complete
**Completed:**
- Project folder created
- Virtual environment (venv)
- Git initialized + .gitignore
- .env + .env.example
- requirements.txt skeleton
- Hello World Gradio app runs

---

### MODULE 1 — Data Ingestion & Validation
**Goal:** Upload CSV/Excel → validated → DataFrame in memory + SQLite on disk
**Status:** 🔴 Not Started
**Approach:** Custom (no pre-built alternative for app-specific validation)
**Files:**
- `src/data/validator.py`
- `src/data/loader.py`
- `src/utils/config.py`
- `tests/test_validator.py`

**validator.py responsibilities:**
- File size ≤ 10MB
- Extension is .csv or .xlsx only
- File is not empty / not corrupt
- At least 1 row of data exists
- Raises descriptive errors for each failure

**loader.py responsibilities:**
- `pd.read_csv()` or `pd.read_excel()` with error handling
- Clean column names (strip spaces, lowercase)
- `df.to_sql("data", sqlite_conn, if_exists="replace")` → finance.db
- Return both df (for stats engine) and db_path (for sql engine)
- Log schema: column names + dtypes

**Deliverable:** Upload file → validated DataFrame in memory + SQLite at `finance.db`

---

### MODULE 2 — LLM Router (Groq Primary + OpenAI Fallback)
**Goal:** All LLM calls go through a single router with automatic fallback
**Status:** 🔴 Not Started
**Approach:** LangChain + custom routing wrapper
**Files:**
- `src/utils/llm_router.py`
- `src/utils/config.py`
- `.env`

**Routing Strategy:**

| Task | Model | Provider |
|------|-------|----------|
| Query classification (sql vs stats) | llama3-70b-8192 | Groq |
| SQL query generation | llama3-70b-8192 | Groq |
| Result explanation | mixtral-8x7b-32768 | Groq |
| Any task (fallback) | gpt-4o-mini | OpenAI |

**Fallback Logic:**
- Try Groq → if RateLimitError or timeout → retry once → fallback to OpenAI
- If OpenAI also fails → return user-friendly error, do not crash

**Code Pattern:**
```python
def get_llm(task: str = "general"):
    try:
        return ChatGroq(
            model="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )
    except Exception:
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0
        )
```

**Deliverable:** Single `get_llm(task)` function used everywhere, fallback automatic

---

### MODULE 3 — SQL Engine (Text-to-SQL)
**Goal:** Natural language → SQL query → execute on SQLite → exact result
**Status:** 🔴 Not Started
**Approach:** LangChain `create_sql_query_chain` + custom SQL validator
**Files:**
- `src/engines/sql_engine.py`
- `src/engines/sql_validator.py`
- `tests/test_sql_engine.py`

**sql_engine.py responsibilities:**
```python
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

def run_sql_query(question: str, db_path: str, llm) -> dict:
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    chain = create_sql_query_chain(llm, db)
    sql = chain.invoke({"question": question})

    if not validate_sql(sql):
        return {"error": "Generated query failed safety check"}

    result = db.run(sql)
    return {"sql": sql, "result": result}
```

**sql_validator.py responsibilities:**
```python
FORBIDDEN = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER",
             "CREATE", "TRUNCATE", "EXEC", "EXECUTE"]

def validate_sql(query: str) -> bool:
    upper = query.upper()
    return not any(word in upper for word in FORBIDDEN)
```

**Why this is safe:** SQLite has no network access, no filesystem access beyond its own file, and with SELECT-only enforcement there is zero attack surface.

**Deliverable:** "Total sales by region?" → SQL generated → validated → executed → exact result returned

---

### MODULE 4 — Stats Engine (Pandas/NumPy + RestrictedPython)
**Goal:** Statistical queries → Pandas/NumPy code → sandboxed execution → result
**Status:** 🔴 Not Started
**Approach:** Groq generates Pandas/NumPy code → RestrictedPython executes it
**Files:**
- `src/engines/stats_engine.py`
- `src/security/restricted_exec.py`
- `tests/test_stats_engine.py`

**Handles these query types:**
- Forecasting / trend prediction → NumPy polyfit or linear regression
- Moving averages → `df.rolling().mean()`
- Correlation between columns → `df.corr()`
- Anomaly detection → std deviation outliers
- Percentage change over time → `df.pct_change()`
- Seasonal patterns → groupby + aggregation

**restricted_exec.py:**
```python
from RestrictedPython import compile_restricted, safe_globals, safe_builtins

ALLOWED_IMPORTS = {'pandas', 'numpy', 'scipy', 'math', 'datetime'}

def execute_stats_code(code: str, df) -> any:
    byte_code = compile_restricted(code, '<string>', 'exec')
    local_vars = {"df": df}
    safe_env = {
        "__builtins__": safe_builtins,
        "pd": __import__("pandas"),
        "np": __import__("numpy"),
    }
    exec(byte_code, safe_env, local_vars)
    return local_vars.get("result")
```

**Blocked by RestrictedPython:** `os`, `sys`, `subprocess`, `open`, `eval`, `exec`, `__import__`, file I/O, network

**Deliverable:** "Show me the sales trend for next 3 months" → Pandas code generated → sandboxed execution → result

---

### MODULE 5 — LangGraph Query Router
**Goal:** Classify every query as "sql" or "stats" and route to correct engine
**Status:** 🔴 Not Started
**Approach:** LangGraph conditional edge + Groq classification call
**Files:**
- `src/graph/router.py`
- `tests/test_router.py`

**router.py:**
```python
ROUTER_PROMPT = """
Classify this financial data question as exactly one of:
- "sql" → if it asks for: filtering, aggregation, counting, sorting,
           grouping, totals, averages, top-N, date ranges, comparisons
- "stats" → if it asks for: forecasting, prediction, trends over time,
             correlation, moving average, anomaly detection, growth rate

Question: {query}

Respond with ONLY the word "sql" or "stats". Nothing else.
"""

def classify_query(state: FinanceGPTState) -> str:
    llm = get_llm("routing")
    result = llm.invoke(ROUTER_PROMPT.format(query=state["query"]))
    return result.content.strip().lower()
```

**LangGraph conditional routing:**
```python
workflow.add_conditional_edges(
    "router_node",
    classify_query,
    {
        "sql": "sql_node",
        "stats": "stats_node"
    }
)
```

**Deliverable:** Every query correctly routed to the right engine

---

### MODULE 6 — LangGraph Workflow (Full Graph Assembly)
**Goal:** Wire all nodes into a single executable StateGraph
**Status:** 🔴 Not Started
**Approach:** LangGraph StateGraph with all nodes connected
**Files:**
- `src/graph/workflow.py`
- `src/graph/nodes.py`
- `src/graph/state.py`

**Full Graph:**
```
START
  ↓
validate_and_load      (M1 — file → df + SQLite)
  ↓
check_cache            (M7 — InMemoryCache lookup)
  ↓ (miss)
router_node            (M5 — classify: sql or stats)
  ↓           ↓
sql_node    stats_node  (M3 / M4)
  ↓           ↓
  └─────┬──────┘
        ↓
visualize_node         (Plotly chart generation)
        ↓
explain_node           (Groq plain English explanation)
        ↓
format_output_node     (combine all outputs)
        ↓
END
```

**Error handling at every node:** If a node fails → set state["error"] → route to error_node → show user-friendly message → never crash

**Deliverable:** `workflow.invoke({"file_path": ..., "query": ...})` returns complete result

---

### MODULE 7 — Query Caching
**Goal:** Identical or similar queries return instantly without LLM call
**Status:** 🔴 Not Started
**Approach:** LangChain InMemoryCache (3 lines)
**Files:**
- `src/utils/cache_setup.py`

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())
# That's it. All LLM calls now auto-cache.
```

**Deliverable:** Repeated queries return from cache, no API call made

---

### MODULE 8 — Gradio UI
**Goal:** Clean, functional UI wiring everything together
**Status:** 🔴 Not Started
**Files:**
- `app.py`

**UI Components:**
- File upload widget (CSV/Excel)
- Chat input box
- Response text area (Groq explanation)
- Plotly chart display
- SQL query shown in expandable code block (transparency feature)
- Raw results table (expandable)
- Query history (session)
- Error messages styled clearly

**Integration:**
```python
from src.graph.workflow import create_workflow

workflow = create_workflow()

def process_query(file, query, history):
    result = workflow.invoke({
        "file_path": file.name,
        "query": query,
        "session_id": get_session_id()
    })
    return (
        result["explanation"],
        result["chart_json"],    # Plotly JSON
        result["sql_query"],     # shown in code block
        result.get("error")
    )
```

**Deliverable:** Working single-tab app, all features connected

---

### MODULE 9 — State Persistence (LangGraph + SQLite Checkpointer)
**Goal:** Conversation history survives page refresh
**Status:** 🔴 Not Started
**Approach:** LangGraph built-in SQLite checkpointer

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as saver:
    workflow = create_workflow(checkpointer=saver)
```

**Deliverable:** User can refresh and continue previous session

---

### MODULE 10 — Testing Suite
**Goal:** pytest coverage for all critical paths, zero real API calls in tests
**Status:** 🔴 Not Started
**Files:**
- `tests/test_validator.py` — file validation edge cases
- `tests/test_sql_engine.py` — SQL generation + validator
- `tests/test_stats_engine.py` — RestrictedPython execution
- `tests/test_router.py` — classification accuracy (mocked LLM)
- `tests/test_workflow.py` — graph node unit tests
- `tests/integration/test_full_pipeline.py` — end-to-end with sample CSV

**All LLM calls mocked with pytest-mock. No real API usage in tests.**

**Deliverable:** `pytest` runs green, 15+ tests passing

---

### MODULE 11 — Deployment
**Goal:** Live public URL on HuggingFace Spaces
**Status:** 🔴 Not Started
**Files:**
- `Dockerfile`
- `docker-compose.yml`
- `docs/DEPLOYMENT.md`

**Deliverable:** Public URL, app fully functional in production

---

### MODULE 12 — Documentation & Portfolio Polish
**Goal:** Repo impresses AI engineering recruiters
**Status:** 🔴 Not Started
**Files:**
- `README.md` — overview, demo GIF, architecture diagram, tech decisions
- `docs/ARCHITECTURE.md` — LangGraph workflow diagram, why hybrid approach

**Key points README must explain:**
- Why Text-to-SQL instead of Pandas Agent (exact math, no sandboxing needed on SQL path)
- Why hybrid router (right tool for right job)
- Why Groq (speed + free tier) with OpenAI fallback (resilience)
- Why RestrictedPython only on stats path (defense in depth, minimal surface)
- LangGraph workflow diagram

**Deliverable:** Any recruiter reading README understands the architecture in 2 minutes

---

## MODULE STATUS TRACKER

| Module | Name | Approach | Status | Notes |
|--------|------|----------|--------|-------|
| M0 | Project Setup | Custom | 🟢 Complete | |
| M1 | Data Ingestion + SQLite | Custom | 🔴 Not Started | |
| M2 | LLM Router | LangChain + custom fallback | 🔴 Not Started | |
| M3 | SQL Engine | LangChain create_sql_query_chain | 🔴 Not Started | |
| M4 | Stats Engine | Pandas/NumPy + RestrictedPython | 🔴 Not Started | |
| M5 | Query Router | LangGraph conditional + Groq | 🔴 Not Started | |
| M6 | LangGraph Workflow | LangGraph StateGraph | 🔴 Not Started | |
| M7 | Query Caching | LangChain InMemoryCache | 🔴 Not Started | |
| M8 | Gradio UI | Custom | 🔴 Not Started | |
| M9 | State Persistence | LangGraph SQLite checkpointer | 🔴 Not Started | |
| M10 | Testing Suite | pytest + pytest-mock | 🔴 Not Started | |
| M11 | Deployment | Docker + HF Spaces | 🔴 Not Started | |
| M12 | Documentation | Custom | 🔴 Not Started | |

---

## PHASE → MODULE MAPPING

| Phase | Modules | Goal |
|-------|---------|------|
| Phase 1 — MVP | M0–M8 | Full working hybrid query app |
| Phase 2 — Intelligence | M9, + Langfuse, streaming | Persistence, observability, streaming responses |
| Phase 3 — Polish | M10, M11, M12 | Tests, deploy, docs |

---

## REQUIREMENTS.TXT (FINAL)

```
# UI
gradio>=4.0

# LLM & Orchestration
langchain
langchain-groq
langchain-openai
langchain-community
langgraph

# SQL Engine
sqlalchemy              # SQLDatabase requires this

# Data Processing
pandas
numpy
scipy
openpyxl                # Excel support

# Visualization
plotly

# Code Execution Safety (stats node only)
RestrictedPython

# Caching + Config
python-dotenv
```

---

## API KEYS (.env file)

```bash
# LLM Providers
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Application Settings
MAX_FILE_SIZE_MB=10
EXECUTION_TIMEOUT_SECONDS=5
CACHE_ENABLED=true
DB_PATH=finance.db
CHECKPOINTS_DB=checkpoints.db
```

---

## SECURITY MODEL

| Path | Security Mechanism | Risk Level |
|------|--------------------|------------|
| SQL Path | SELECT-only whitelist validator | Minimal — SQLite is isolated |
| Stats Path | RestrictedPython sandbox | Low — blocks all dangerous operations |
| File Upload | Size + extension + corruption check | Minimal |
| API Keys | Environment variables only, never logged | Standard |

**SQL Path needs no sandboxing because:**
- SQLite cannot make network calls
- SQLite cannot access filesystem beyond its own file
- SELECT-only enforcement blocks all data mutation
- LLM generates query, humans never write SQL

**Stats Path needs RestrictedPython because:**
- Groq generates arbitrary Python code
- Without sandboxing, generated code could theoretically access `os`, `sys`, `subprocess`
- RestrictedPython blocks all dangerous builtins and imports

---

## SECURITY RULES — RESTRICTEDPYTHON (STATS NODE)

```
Allowed modules:  pandas, numpy, scipy, math, datetime
Allowed builtins: safe_builtins (RestrictedPython default)

Blocked: os, sys, subprocess, eval, exec, open, file,
         __import__, compile, input, print to stdout
```

---

## PERFORMANCE TARGETS

| Step | Target |
|------|--------|
| File upload + SQLite load | < 5 seconds |
| Query classification (router) | < 1 second |
| SQL query generation + execution | < 4 seconds |
| Stats code generation + execution | < 6 seconds |
| Total query response (SQL path) | < 8 seconds |
| Total query response (stats path) | < 10 seconds |
| Max file size | 10 MB |
| Cache hit response | < 0.5 seconds |

---

## LLM ROUTING STRATEGY (FINAL)

| Task | Model | Provider | Why |
|------|-------|----------|-----|
| Query classification | llama3-70b-8192 | Groq | Fast, accurate binary classification |
| SQL generation | llama3-70b-8192 | Groq | Strong reasoning for SQL |
| Stats code generation | llama3-70b-8192 | Groq | Strong at Pandas/NumPy |
| Result explanation | mixtral-8x7b-32768 | Groq | Excellent at summarization |
| Any task (fallback) | gpt-4o-mini | OpenAI | Reliable, handles everything |

**Fallback trigger:** Groq RateLimitError, timeout > 10s, or any API error → automatic switch to OpenAI

---

## AI ENGINEERING PRINCIPLES APPLIED

### 1. Right Tool for the Right Job
- Exact calculations → SQL (not LLM-generated Python, which can hallucinate math)
- Statistical analysis → Pandas/NumPy (purpose-built for this)
- Routing decision → LLM (classification is what LLMs do best)

### 2. Minimal Attack Surface
- SQL path has zero code execution risk (SELECT-only, SQLite-isolated)
- RestrictedPython only where actually needed (stats path)
- No vector store = no embedding pipeline to go wrong

### 3. Don't Reinvent the Wheel
- `create_sql_query_chain` → 5 lines vs 100 lines custom
- LangGraph conditional routing → built-in, not custom dispatcher
- LangChain InMemoryCache → 3 lines vs custom cache dict

### 4. Custom Only Where Necessary
- File validation (app-specific business rules)
- Gradio UI (your UX design)
- SQL whitelist validator (simple but critical)
- Prompt templates (your domain knowledge)

### 5. Fail Gracefully
- Every LangGraph node has error handling
- LLM fallback chain (Groq → OpenAI)
- User sees friendly messages, never raw stack traces

---

## INTERVIEW QUESTIONS — PREPARED ANSWERS

**"Why Text-to-SQL instead of a Pandas agent?"**
> SQL is deterministic — it always returns exact results across all rows. A Pandas agent generates Python code which can hallucinate intermediate steps and silently return wrong aggregations. For financial data, exactness is non-negotiable.

**"Why do you have two query engines instead of one?"**
> SQL can't do forecasting or correlation — those need numerical computation. Rather than shoehorn everything into SQL or everything into Python, I built a router that picks the right engine per query type. That's how production analytics systems actually work.

**"How do you ensure the stats code execution is safe?"**
> RestrictedPython compiles the generated code into a restricted bytecode that blocks all dangerous builtins — no os, no subprocess, no file I/O, no network. The SQL path doesn't need this because SQLite execution is inherently isolated.

**"Why did you drop ChromaDB/FAISS?"**
> RAG is the wrong architecture for structured tabular data. Vector search retrieves the top-K most similar rows — but if I need to SUM 2000 rows, FAISS gives me 5 and I get a confidently wrong answer. Direct SQL access processes every row exactly.

**"Why LangGraph instead of sequential function calls?"**
> LangGraph gives me conditional routing, built-in state management, automatic checkpointing, and clean error handling — all things I'd have to build manually with sequential functions. It also makes the architecture visually explainable, which matters for documentation.

---

## HOW TO USE THIS FILE

**At the start of every new chat session:**
> "Here is my AGENT.md: [paste contents]. We are working on MODULE X. Continue from where we left off."

**After completing a module:**
> Update status: 🔴 Not Started → 🟡 In Progress → 🟢 Complete
> Add any notes on decisions or issues in the Notes column of the tracker

**Status legend:**
- 🔴 Not Started
- 🟡 In Progress
- 🟢 Complete
- ⏸️ Blocked

**DO NOT reopen finalized decisions.** If a new issue arises, add a note in the relevant module and handle it there.

---

*FinanceGPT · AGENT.md v2 — Hybrid Architecture (Text-to-SQL + Stats Engine) · Architecture Finalized*
