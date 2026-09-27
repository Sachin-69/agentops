"""AgentOps observability store — SQLite persistence.

Tables: agents, agent_runs, trace_spans, llm_calls, tool_calls, evaluations, prompts.
Stdlib only (sqlite3). Thread-safe via a lock; check_same_thread=False.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "agentops.db"


def _now() -> float:
    return time.time()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT DEFAULT '',
  model TEXT DEFAULT 'gpt-4o-mini',
  system_prompt TEXT DEFAULT '',
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS prompts (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  system_prompt TEXT NOT NULL,
  created_at REAL NOT NULL,
  FOREIGN KEY (agent_id) REFERENCES agents(id)
);
CREATE TABLE IF NOT EXISTS agent_runs (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,
  input TEXT NOT NULL,
  output TEXT DEFAULT '',
  status TEXT DEFAULT 'running',
  started_at REAL NOT NULL,
  completed_at REAL,
  latency_ms REAL DEFAULT 0,
  total_tokens INTEGER DEFAULT 0,
  estimated_cost REAL DEFAULT 0,
  prompt_version INTEGER DEFAULT 1,
  grounded INTEGER DEFAULT 0,
  FOREIGN KEY (agent_id) REFERENCES agents(id)
);
CREATE TABLE IF NOT EXISTS trace_spans (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  parent_id TEXT,
  span_type TEXT NOT NULL,
  name TEXT NOT NULL,
  started_at REAL NOT NULL,
  ended_at REAL,
  latency_ms REAL DEFAULT 0,
  status TEXT DEFAULT 'running',
  input_json TEXT DEFAULT '',
  output_json TEXT DEFAULT '',
  metadata_json TEXT DEFAULT '{}',
  FOREIGN KEY (run_id) REFERENCES agent_runs(id)
);
CREATE TABLE IF NOT EXISTS llm_calls (
  id TEXT PRIMARY KEY,
  span_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  model TEXT NOT NULL,
  prompt TEXT DEFAULT '',
  response TEXT DEFAULT '',
  input_tokens INTEGER DEFAULT 0,
  output_tokens INTEGER DEFAULT 0,
  latency_ms REAL DEFAULT 0,
  cost REAL DEFAULT 0,
  timestamp REAL NOT NULL,
  FOREIGN KEY (run_id) REFERENCES agent_runs(id)
);
CREATE TABLE IF NOT EXISTS tool_calls (
  id TEXT PRIMARY KEY,
  span_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  tool_name TEXT NOT NULL,
  input TEXT DEFAULT '',
  output TEXT DEFAULT '',
  latency_ms REAL DEFAULT 0,
  status TEXT DEFAULT 'success',
  FOREIGN KEY (run_id) REFERENCES agent_runs(id)
);
CREATE TABLE IF NOT EXISTS evaluations (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  accuracy REAL NOT NULL,
  relevance REAL NOT NULL,
  faithfulness REAL NOT NULL,
  instruction_following REAL NOT NULL,
  overall REAL NOT NULL,
  method TEXT DEFAULT 'heuristic',
  detail_json TEXT DEFAULT '{}',
  created_at REAL NOT NULL,
  FOREIGN KEY (run_id) REFERENCES agent_runs(id)
);
"""


class Store:
    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # -- low-level helpers --
    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        # Blanket redaction: no API-key-like secret can ever be persisted,
        # no matter which write path it came through (see app/redaction.py).
        from .redaction import redact
        safe = tuple(redact(p) if isinstance(p, str) else p for p in params)
        with self._lock:
            cur = self._conn.execute(sql, safe)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def _query_one(self, sql: str, params: tuple = ()) -> Optional[dict]:
        rows = self._query(sql, params)
        return rows[0] if rows else None

    # -- agents --
    def create_agent(self, name: str, description: str = "", model: str = "gpt-4o-mini",
                     system_prompt: str = "") -> dict:
        agent_id = new_id("agent")
        ts = _now()
        self._execute(
            "INSERT INTO agents (id, name, description, model, system_prompt, created_at) VALUES (?,?,?,?,?,?)",
            (agent_id, name, description, model, system_prompt, ts),
        )
        self._execute(
            "INSERT INTO prompts (id, agent_id, version, system_prompt, created_at) VALUES (?,?,?,?,?)",
            (new_id("prompt"), agent_id, 1, system_prompt, ts),
        )
        return self.get_agent(agent_id)  # type: ignore[return-value]

    def list_agents(self) -> list[dict]:
        return self._query("SELECT * FROM agents ORDER BY created_at DESC")

    def get_agent(self, agent_id: str) -> Optional[dict]:
        return self._query_one("SELECT * FROM agents WHERE id=?", (agent_id,))

    def get_prompt_version(self, agent_id: str) -> int:
        row = self._query_one(
            "SELECT MAX(version) AS v FROM prompts WHERE agent_id=?", (agent_id,)
        )
        return int(row["v"]) if row and row["v"] else 1

    # -- runs --
    def create_run(self, agent_id: str, run_input: str) -> dict:
        run_id = new_id("run")
        ts = _now()
        pv = self.get_prompt_version(agent_id)
        self._execute(
            "INSERT INTO agent_runs (id, agent_id, input, status, started_at, prompt_version) VALUES (?,?,?,?,?,?)",
            (run_id, agent_id, run_input, "running", ts, pv),
        )
        return self.get_run(run_id)  # type: ignore[return-value]

    def finish_run(self, run_id: str, output: str, status: str, latency_ms: float,
                   total_tokens: int, cost: float, grounded: bool = False) -> None:
        self._execute(
            "UPDATE agent_runs SET output=?, status=?, completed_at=?, latency_ms=?, "
            "total_tokens=?, estimated_cost=?, grounded=? WHERE id=?",
            (output, status, _now(), latency_ms, total_tokens, cost, int(grounded), run_id),
        )

    def get_run(self, run_id: str) -> Optional[dict]:
        return self._query_one("SELECT * FROM agent_runs WHERE id=?", (run_id,))

    def list_runs(self, agent_id: str = "", limit: int = 100) -> list[dict]:
        if agent_id:
            return self._query(
                "SELECT * FROM agent_runs WHERE agent_id=? ORDER BY started_at DESC LIMIT ?",
                (agent_id, limit),
            )
        return self._query(
            "SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT ?", (limit,)
        )

    # -- spans --
    def create_span(self, run_id: str, span_type: str, name: str,
                    parent_id: str = "", input_data: str = "",
                    metadata: dict | None = None) -> dict:
        span_id = new_id("span")
        self._execute(
            "INSERT INTO trace_spans (id, run_id, parent_id, span_type, name, started_at, status, input_json, metadata_json)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (span_id, run_id, parent_id or None, span_type, name, _now(), "running",
             input_data, json.dumps(metadata or {})),
        )
        return {"id": span_id, "run_id": run_id, "name": name, "span_type": span_type,
                "started_at": _now()}

    def finish_span(self, span_id: str, status: str = "success", output_data: str = "",
                    metadata: dict | None = None) -> None:
        row = self._query_one("SELECT started_at, metadata_json FROM trace_spans WHERE id=?", (span_id,))
        started = row["started_at"] if row else _now()
        ended = _now()
        latency = (ended - started) * 1000.0
        meta = {}
        if row and row["metadata_json"]:
            try:
                meta = json.loads(row["metadata_json"])
            except Exception:
                meta = {}
        if metadata:
            meta.update(metadata)
        self._execute(
            "UPDATE trace_spans SET ended_at=?, latency_ms=?, status=?, output_json=?, metadata_json=? WHERE id=?",
            (ended, latency, status, output_data, json.dumps(meta), span_id),
        )

    def get_traces(self, run_id: str) -> list[dict]:
        return self._query(
            "SELECT * FROM trace_spans WHERE run_id=? ORDER BY started_at ASC", (run_id,)
        )

    # -- llm / tool calls --
    def add_llm_call(self, run_id: str, span_id: str, model: str, prompt: str, response: str,
                     in_tokens: int, out_tokens: int, latency_ms: float, cost: float) -> dict:
        call_id = new_id("llm")
        self._execute(
            "INSERT INTO llm_calls (id, span_id, run_id, model, prompt, response, input_tokens, output_tokens, latency_ms, cost, timestamp)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (call_id, span_id, run_id, model, prompt, response, in_tokens, out_tokens,
             latency_ms, cost, _now()),
        )
        return {"id": call_id}

    def add_tool_call(self, run_id: str, span_id: str, tool_name: str, tool_input: str,
                      tool_output: str, latency_ms: float, status: str = "success") -> dict:
        call_id = new_id("tool")
        self._execute(
            "INSERT INTO tool_calls (id, span_id, run_id, tool_name, input, output, latency_ms, status)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (call_id, span_id, run_id, tool_name, tool_input, tool_output, latency_ms, status),
        )
        return {"id": call_id}

    def get_llm_calls(self, run_id: str) -> list[dict]:
        return self._query("SELECT * FROM llm_calls WHERE run_id=? ORDER BY timestamp ASC", (run_id,))

    def get_tool_calls(self, run_id: str) -> list[dict]:
        return self._query(
            "SELECT t.* FROM tool_calls t JOIN trace_spans s ON s.id=t.span_id "
            "WHERE t.run_id=? ORDER BY s.started_at ASC", (run_id,)
        )

    # -- evaluations --
    def save_evaluation(self, run_id: str, scores: dict[str, float],
                        method: str = "heuristic", detail: dict | None = None) -> dict:
        eval_id = new_id("eval")
        overall = round(
            (scores.get("accuracy", 0) + scores.get("relevance", 0)
             + scores.get("faithfulness", 0) + scores.get("instruction_following", 0)) / 4.0, 4
        )
        self._execute(
            "INSERT INTO evaluations (id, run_id, accuracy, relevance, faithfulness, instruction_following, overall, method, detail_json, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (eval_id, run_id, scores.get("accuracy", 0), scores.get("relevance", 0),
             scores.get("faithfulness", 0), scores.get("instruction_following", 0),
             overall, method, json.dumps(detail or {}), _now()),
        )
        return self.get_evaluation(eval_id)  # type: ignore[return-value]

    def get_evaluation(self, eval_id: str) -> Optional[dict]:
        return self._query_one("SELECT * FROM evaluations WHERE id=?", (eval_id,))

    def get_latest_evaluation(self, run_id: str) -> Optional[dict]:
        return self._query_one(
            "SELECT * FROM evaluations WHERE run_id=? ORDER BY created_at DESC LIMIT 1", (run_id,)
        )

    # -- analytics --
    def analytics_overview(self) -> dict[str, Any]:
        row = self._query_one(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS ok, "
            "AVG(latency_ms) AS avg_lat, AVG(estimated_cost) AS avg_cost, "
            "SUM(total_tokens) AS toks, SUM(estimated_cost) AS cost "
            "FROM agent_runs"
        ) or {}
        total = int(row.get("total") or 0)
        ok = int(row.get("ok") or 0)
        per_agent = self._query(
            "SELECT agent_id, COUNT(*) AS runs, AVG(latency_ms) AS avg_latency_ms, "
            "AVG(estimated_cost) AS avg_cost FROM agent_runs GROUP BY agent_id"
        )
        recent = self._query(
            "SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT 10"
        )
        err = self._query_one(
            "SELECT COUNT(*) AS c FROM agent_runs WHERE status='error'"
        )
        return {
            "total_runs": total,
            "success_rate": round(ok / total, 4) if total else 0.0,
            "success_count": ok,
            "error_count": int((err or {}).get("c") or 0),
            "avg_latency_ms": round(float(row.get("avg_lat") or 0), 2),
            "avg_latency_s": round(float(row.get("avg_lat") or 0) / 1000.0, 2),
            "avg_cost": round(float(row.get("avg_cost") or 0), 6),
            "total_tokens": int(row.get("toks") or 0),
            "total_cost": round(float(row.get("cost") or 0), 6),
            "per_agent": per_agent,
            "recent_runs": recent,
        }
