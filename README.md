# Chat Support Agent

A support chatbot that answers **only** from the support documentation you give
it. It uses Retrieval-Augmented Generation (RAG): questions are matched against
your docs, and if nothing relevant is found the agent politely refuses instead
of guessing or using outside knowledge.

## Features

- **Docs-only answers** — strict guardrails refuse out-of-scope questions.
- **RAG retrieval** — TF-IDF vector search over your docs (no API key required).
- **Provider-agnostic LLM** — run with no LLM (`none`), or plug in OpenAI,
  Anthropic, or a local Ollama model.
- **REST API + web UI** — FastAPI backend with a simple chat page.
- **CLI** — chat from the terminal.
- **Cited sources** — every grounded answer lists which docs it came from.

## Project structure

```
.
├── app/
│   ├── config.py        # settings from .env
│   ├── llm.py           # provider-agnostic LLM layer (none/openai/anthropic/ollama)
│   ├── vectorstore.py   # doc ingestion + TF-IDF retrieval (RAG)
│   ├── chat_engine.py   # retrieval + LLM + guardrails
│   └── main.py          # FastAPI app + endpoints
├── static/index.html    # web chat UI
├── docs/                # <-- put your support docs here (.md / .txt)
├── cli.py               # terminal chat
├── test_agent.py        # smoke tests
├── requirements.txt
└── .env.example
```

## Setup

```powershell
# 1. (optional) create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. install dependencies
pip install -r requirements.txt

# 3. configure (optional — defaults work with no keys)
Copy-Item .env.example .env
```

## Add your support docs

Drop your `.md` or `.txt` files into the `docs/` folder. The sample docs
(`getting-started.md`, `billing.md`, `troubleshooting.md`) are there as an
example — replace them with your own.

## Run

### Web UI + API
```powershell
uvicorn app.main:app --reload
```
Then open http://localhost:8000

### CLI
```powershell
python cli.py
```

### Tests
```powershell
python test_agent.py
```

## API

| Method | Path           | Description                                   |
|--------|----------------|-----------------------------------------------|
| GET    | `/api/health`  | Provider + index status                       |
| POST   | `/api/chat`    | `{ "message": "..." }` -> answer + sources    |
| POST   | `/api/reindex` | Rebuild the index after changing docs         |

Example:
```powershell
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{\"message\": \"How do I get a refund?\"}'
```

## Choosing an LLM provider

Set `LLM_PROVIDER` in `.env`:

| Value       | Requires                              |
|-------------|---------------------------------------|
| `none`      | nothing (extractive answers from docs)|
| `openai`    | `OPENAI_API_KEY`, `pip install openai`|
| `anthropic` | `ANTHROPIC_API_KEY`, `pip install anthropic` |
| `ollama`    | a running Ollama server               |

## How the "docs-only" guarantee works

1. The retriever finds the most relevant doc chunks for the question.
2. If no chunk clears `MIN_RELEVANCE`, the agent **refuses** (out of scope).
3. Only the retrieved chunks are sent to the LLM as context.
4. The system prompt forbids outside knowledge; if the LLM can't answer from the
   context it returns a sentinel and the agent refuses.

Tune `MIN_RELEVANCE`, `TOP_K`, `CHUNK_SIZE`, and `CHUNK_OVERLAP` in `.env`.
