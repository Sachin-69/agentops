# AgentOpsAI — Agent Observability & Evaluation Platform

Trace, debug, evaluate, and measure AI-agent executions across LLM calls,
tool calls, latency, token usage, errors, and cost.

Ask your agent a question — and instead of only seeing the answer, see
exactly how the agent arrived there.

## What it does

- **Agents** — create agents with a name, model, description, and versioned
  system prompt.
- **Traced runs** — every execution runs an LLM → tools → LLM pipeline where
  each step is recorded as a span: model, prompt, response, input/output
  tokens, latency, and cost.
- **Dashboard** — total runs, success rate, average latency, average cost,
  total tokens, latency charts, cost by agent, and a filterable run table.
- **Run detail** — execution waterfall timeline. Click any LLM or tool span to
  inspect its full input, output, tokens, latency, and cost.
- **Evaluations** — one-click scoring per run: accuracy, relevance,
  faithfulness, and instruction following (heuristic by default, LLM-judge
  refinement when a real provider is configured).
- **BYOK** — visitors can optionally use their own OpenAI/Anthropic key,
  stored only in their browser and used in-memory per request. Keys are never
  saved to the database, never written into traces (redaction guard), and
  never logged.

## Quickstart

```powershell
# 1. (optional) create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. install dependencies
pip install -r requirements.txt

# 3. run the server
uvicorn app.main:app --reload
```

Open http://localhost:8000 — cinematic hero up top, dashboard below:

1. Pick the seeded **Research Assistant** (or create your own agent).
2. Ask something like *“How do I get a refund?”* and hit **Execute & trace**.
3. Click the run → open the waterfall → click an LLM/tool span to inspect it.
4. Hit **Evaluate** for accuracy / relevance / faithfulness /
   instruction-following scores.

No API key needed: the built-in keyless demo answers extractively from the
docs in `docs/`. Add your own key via the **API key** button in the nav bar
for real LLM answers.

## API

| Method | Path                      | Description                                    |
|--------|---------------------------|------------------------------------------------|
| GET    | `/api/health`             | Provider + index status                        |
| POST   | `/api/agents`             | Create an agent                                |
| GET    | `/api/agents`             | List agents                                    |
| GET    | `/api/agents/:id`         | Get one agent                                  |
| POST   | `/api/runs`               | Execute an agent (`{agent_id, input}`)         |
| GET    | `/api/runs`               | List runs (`?agent_id=&limit=`)                |
| GET    | `/api/runs/:id`           | Get one run (+ latest evaluation)              |
| GET    | `/api/runs/:id/traces`    | Spans + LLM calls + tool calls + evaluation    |
| POST   | `/api/runs/:id/evaluate`  | Score the run                                  |
| GET    | `/api/analytics/overview` | Totals, rates, averages, per-agent costs       |
| POST   | `/api/chat`               | Quick grounded answer (utility, untraced)      |

Optional BYOK headers on `/api/chat`, `/api/runs`, `/api/runs/:id/evaluate`:

- `X-LLM-Provider: openai | anthropic` (auto-detected from the key if omitted)
- `X-LLM-Key: <key>` — used in-memory for that request only

## Project structure

```
.
├── app/
│   ├── main.py           # FastAPI app + all API routes
│   ├── agent_runner.py   # traced LLM → tools → LLM execution
│   ├── agentops_store.py # SQLite: agents, runs, spans, calls, evals, prompts
│   ├── evaluator.py      # accuracy / relevance / faithfulness / instruction
│   ├── redaction.py      # API-key redaction guard for all stored text
│   ├── chat_engine.py    # docs-grounded answering + guardrails
│   ├── vectorstore.py    # TF-IDF retrieval over docs/ (no key needed)
│   ├── llm.py            # provider layer: none / openai / anthropic / ollama
│   └── config.py         # settings from .env
├── static/index.html     # dashboard UI (single file, no build step)
├── docs/                 # support docs the demo agent searches (.md / .txt)
├── cli.py                # terminal chat
├── test_agent.py         # smoke tests
├── requirements.txt
└── .env.example
```

## Configuration

Copy `.env.example` to `.env`. Server-side defaults work with no keys
(`LLM_PROVIDER=none`). To use a real model server-side instead of per-request
BYOK, set `LLM_PROVIDER=openai|anthropic|ollama` plus the matching key and
`pip install openai` / `pip install anthropic`.

Tune retrieval with `TOP_K`, `MIN_RELEVANCE`, `CHUNK_SIZE`, `CHUNK_OVERLAP`.
Rebuild the index after changing docs via `POST /api/reindex`.

## Tests

```powershell
python test_agent.py
```
