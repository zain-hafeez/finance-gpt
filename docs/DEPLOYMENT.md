# FinanceGPT — Deployment Guide

## Live Demo
🌐 **[Live App → HuggingFace Spaces](https://huggingface.co/spaces/zain-hafeez/FinanceGPT)**

---

## Run Locally (Standard Python)

### Prerequisites
- Python 3.11+
- Groq API key — free at console.groq.com
- OpenAI API key — optional (used as fallback only)

### Setup
```bash
git clone https://github.com/zain-hafeez/finance-gpt.git
cd finance-gpt
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
python app.py
# Open http://localhost:7860
```

---

## Run Locally with Docker

### Prerequisites
- Docker Desktop installed and running

### Steps
```bash
# Add API keys to .env first, then:
docker-compose up --build
# Open http://localhost:7860
```

---

## Deploy to HuggingFace Spaces

1. Create account at huggingface.co
2. New Space → SDK: Docker → Visibility: Public
3. Settings → Repository Secrets → add GROQ_API_KEY + OPENAI_API_KEY
4. Push:
```bash
git remote add huggingface https://huggingface.co/spaces/zain-hafeez/FinanceGPT
git push huggingface main
```
5. Watch Logs tab → wait ~5 minutes → status: Running ✅

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| GROQ_API_KEY | ✅ Yes | — | Primary LLM |
| OPENAI_API_KEY | No | — | Fallback LLM |
| DB_PATH | No | finance.db | SQLite path |
| CHECKPOINTS_DB | No | checkpoints.db | LangGraph state |
| MAX_FILE_SIZE_MB | No | 10 | Upload limit |
| CACHE_ENABLED | No | true | Result caching |