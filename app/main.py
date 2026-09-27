"""FastAPI app: docs-grounded chat agent + AgentOps observability platform."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent_runner import AgentRunner
from .agentops_store import Store
from .chat_engine import ChatEngine
from .config import get_settings
from .evaluator import evaluate_run
from .llm import get_provider, infer_provider_from_key
from .vectorstore import Retriever

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

state: dict = {}


def _build_engine() -> tuple[ChatEngine, Retriever]:
    settings = get_settings()
    docs_dir = os.path.join(BASE_DIR, settings.docs_dir)
    retriever = Retriever()
    retriever.build(docs_dir, settings.chunk_size, settings.chunk_overlap)
    provider = get_provider(settings)
    return ChatEngine(settings, retriever, provider), retriever


def _store() -> Store:
    return state["store"]


def _runner() -> AgentRunner:
    return AgentRunner(state["store"], engine=state.get("engine"),
                       provider=state.get("provider"))


def _byok_provider(
    x_llm_provider: Optional[str] = None,
    x_llm_key: Optional[str] = None,
):
    """Build an ephemeral per-request provider from BYOK headers.

    The key is used in-memory only for this request: it is never stored in
    the database (see the redaction guard in agentops_store), never logged,
    and never echoed back in errors. Returns None when no key was supplied,
    in which case callers fall back to the server-configured provider.
    """
    key = (x_llm_key or "").strip()
    if not key:
        return None
    provider_name = (x_llm_provider or "").strip().lower() or infer_provider_from_key(key)
    if provider_name not in ("openai", "anthropic"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported X-LLM-Provider '{x_llm_provider}'. Use 'openai' or 'anthropic'.",
        )
    try:
        return get_provider(get_settings(), provider_override=provider_name,
                            api_key_override=key)
    except RuntimeError as exc:
        # Never include the key in the error message.
        raise HTTPException(status_code=400, detail=f"Could not use provided key: {exc}")


def _seed_default_agent(store: Store) -> None:
    if store.list_agents():
        return
    store.create_agent(
        name="Research Assistant",
        description="Researches technical topics using web/search tools.",
        model="gpt-4o-mini",
        system_prompt=(
            "You are a research assistant. Research technical topics using the "
            "available search tools, then synthesize a concise, accurate answer "
            "with sources."
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine, retriever = _build_engine()
    settings = get_settings()
    store = Store()
    state["engine"] = engine
    state["retriever"] = retriever
    state["provider"] = engine._provider
    state["store"] = store
    state["settings"] = settings
    _seed_default_agent(store)
    yield
    state.clear()


app = FastAPI(
    title="AgentOpsAI — Agent Observability & Evaluation",
    description="Trace, debug, evaluate, and measure AI-agent executions.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- Schemas ----------
class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's question.")


class SourceOut(BaseModel):
    doc: str
    title: str
    score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    grounded: bool
    sources: List[SourceOut]


class AgentCreate(BaseModel):
    name: str
    description: str = ""
    model: str = "gpt-4o-mini"
    system_prompt: str = ""


class RunCreate(BaseModel):
    agent_id: str
    input: str = Field(..., description="User question / task for the agent.")


def _run_out(r: dict) -> dict:
    return {
        "id": r["id"],
        "agent_id": r["agent_id"],
        "input": r.get("input", ""),
        "output": r.get("output", ""),
        "status": r.get("status", ""),
        "started_at": r.get("started_at"),
        "completed_at": r.get("completed_at"),
        "latency_ms": r.get("latency_ms", 0),
        "latency_s": round(float(r.get("latency_ms") or 0) / 1000.0, 3),
        "total_tokens": r.get("total_tokens", 0),
        "estimated_cost": r.get("estimated_cost", 0),
        "prompt_version": r.get("prompt_version", 1),
    }


# ---------- Legacy chat endpoints (kept) ----------
@app.get("/api/health")
def health():
    retriever: Retriever = state["retriever"]
    settings = get_settings()
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "docs_indexed": retriever.num_docs,
        "chunks_indexed": retriever.num_chunks,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest,
         x_llm_provider: Optional[str] = Header(None, alias="X-LLM-Provider"),
         x_llm_key: Optional[str] = Header(None, alias="X-LLM-Key")):
    retriever: Retriever = state["retriever"]
    if retriever.num_chunks == 0:
        raise HTTPException(status_code=503,
                            detail="No support documents indexed. Add files to docs/ and restart.")
    override = _byok_provider(x_llm_provider, x_llm_key)
    if override is not None:
        engine = ChatEngine(get_settings(), retriever, override)
    else:
        engine = state["engine"]
    try:
        result = engine.ask(req.message)
    except Exception as exc:
        # Live-site safety: a bad/expired user key must yield a clean error,
        # never a 500 traceback and never with key material echoed back.
        from .redaction import redact
        raise HTTPException(
            status_code=502,
            detail=f"LLM request failed: {redact(str(exc))[:300]}",
        )
    return ChatResponse(answer=result.answer, grounded=result.grounded,
                        sources=[SourceOut(**s.__dict__) for s in result.sources])


@app.post("/api/reindex")
def reindex():
    engine, retriever = _build_engine()
    state["engine"] = engine
    state["retriever"] = retriever
    state["provider"] = engine._provider
    return {"status": "reindexed", "chunks_indexed": retriever.num_chunks}


# ---------- AgentOps: agents ----------
@app.post("/api/agents")
def create_agent(body: AgentCreate):
    if not body.name.strip():
        raise HTTPException(status_code=422, detail="Agent name is required.")
    return _store().create_agent(body.name.strip(), body.description,
                                 body.model or "gpt-4o-mini", body.system_prompt)


@app.get("/api/agents")
def list_agents():
    return _store().list_agents()


@app.get("/api/agents/{agent_id}")
def get_agent(agent_id: str):
    a = _store().get_agent(agent_id)
    if not a:
        raise HTTPException(status_code=404, detail="Agent not found.")
    return a


# Back-compat without /api prefix
@app.get("/agents")
def list_agents_plain():
    return _store().list_agents()


@app.get("/agents/{agent_id}")
def get_agent_plain(agent_id: str):
    a = _store().get_agent(agent_id)
    if not a:
        raise HTTPException(status_code=404, detail="Agent not found.")
    return a


@app.post("/agents")
def create_agent_plain(body: AgentCreate):
    return create_agent(body)


# ---------- AgentOps: runs ----------
@app.post("/api/runs")
def create_run(body: RunCreate,
               x_llm_provider: Optional[str] = Header(None, alias="X-LLM-Provider"),
               x_llm_key: Optional[str] = Header(None, alias="X-LLM-Key")):
    store = _store()
    agent = store.get_agent(body.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found.")
    if not body.input.strip():
        raise HTTPException(status_code=422, detail="Input is required.")
    override = _byok_provider(x_llm_provider, x_llm_key)
    runner = AgentRunner(store, engine=state.get("engine"),
                         provider=override or state.get("provider"))
    rec = store.create_run(body.agent_id, body.input.strip())
    try:
        finished = runner.run(agent, body.input.strip(), run_id=rec["id"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Run failed: {exc}")
    out = _run_out(finished)
    out["message"] = "Run completed. Use GET /api/runs/{id}/traces for the full execution trace."
    return out


@app.get("/api/runs")
def list_runs(agent_id: str = "", limit: int = 100):
    return [_run_out(r) for r in _store().list_runs(agent_id, min(limit, 500))]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    r = _store().get_run(run_id)
    if not r:
        raise HTTPException(status_code=404, detail="Run not found.")
    out = _run_out(r)
    out["evaluation"] = _store().get_latest_evaluation(run_id)
    return out


@app.get("/api/runs/{run_id}/traces")
def get_traces(run_id: str):
    store = _store()
    if not store.get_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    return {
        "run_id": run_id,
        "spans": store.get_traces(run_id),
        "llm_calls": store.get_llm_calls(run_id),
        "tool_calls": store.get_tool_calls(run_id),
        "evaluation": store.get_latest_evaluation(run_id),
    }


@app.post("/api/runs/{run_id}/evaluate")
def evaluate(run_id: str,
             x_llm_provider: Optional[str] = Header(None, alias="X-LLM-Provider"),
             x_llm_key: Optional[str] = Header(None, alias="X-LLM-Key")):
    store = _store()
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    override = _byok_provider(x_llm_provider, x_llm_key)
    return evaluate_run(store, run, provider=override or state.get("provider"))


@app.get("/api/analytics/overview")
def analytics():
    return _store().analytics_overview()


# Plain-prefix mirrors
@app.post("/runs")
def create_run_plain(body: RunCreate):
    return create_run(body)


@app.get("/runs")
def list_runs_plain(agent_id: str = "", limit: int = 100):
    return list_runs(agent_id, limit)


@app.get("/runs/{run_id}")
def get_run_plain(run_id: str):
    return get_run(run_id)


@app.get("/runs/{run_id}/traces")
def get_traces_plain(run_id: str):
    return get_traces(run_id)


@app.post("/runs/{run_id}/evaluate")
def evaluate_plain(run_id: str):
    return evaluate(run_id)


@app.get("/analytics/overview")
def analytics_plain():
    return analytics()


# ---------- UI ----------
@app.get("/")
def index():
    """The AgentOps observability dashboard (runs, traces, evaluations)."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/chat")
def chat_page():
    page = os.path.join(STATIC_DIR, "chat.html")
    if os.path.exists(page):
        return FileResponse(page)
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
