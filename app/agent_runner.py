"""Traced agent execution: LLM -> Tool -> LLM pipeline with full observability.

Every operation is recorded as a trace span + llm_calls / tool_calls row,
with latency, token counts, and cost estimates.
"""

from __future__ import annotations

import time
from typing import Any

from .agentops_store import Store
from .llm import build_user_prompt
from .redaction import redact

# Per-1K-token prices (input, output) in USD. Covers common models + GPT-5.x family.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.000150, 0.000600),
    "gpt-4o": (0.002500, 0.010000),
    "gpt-5": (0.005000, 0.015000),
    "gpt-5.x": (0.005000, 0.015000),
    "gpt-5-mini": (0.001000, 0.004000),
    "claude-3-5-sonnet-20240620": (0.003000, 0.015000),
    "llama3.1": (0.000200, 0.000200),
    "none": (0.000050, 0.000100),
    "default": (0.001000, 0.003000),
}


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token). Deterministic, no deps."""
    if not text:
        return 1
    return max(1, int(len(text) / 4))


def estimate_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    key = (model or "").lower().strip()
    prices = PRICING.get(key, PRICING.get("default", (0.001, 0.003)))
    # also try prefix match for gpt-5 variants
    if key not in PRICING and key.startswith("gpt-5"):
        prices = PRICING["gpt-5.x"]
    pin, pout = prices
    return round(in_tokens / 1000.0 * pin + out_tokens / 1000.0 * pout, 6)


class AgentRunner:
    """Executes an agent's question through a traced pipeline.

    Pipeline:
      span: run (root)
      ├── span: llm "LLM Call #1 — plan / query rewrite"
      ├── span: tool "WebSearch" (retriever over docs)
      ├── span: tool "VectorSearch" (second retrieval pass w/ rewritten query)
      ├── span: llm "LLM Call #2 — synthesize answer"
      └── run finished (status success|error, totals)
    """

    def __init__(self, store: Store, engine=None, provider=None) -> None:
        # engine: ChatEngine (has _retriever, _provider, _settings) — optional
        self.store = store
        self.engine = engine
        self.provider = provider

    def _provider_generate(self, question: str, context: str) -> str:
        if self.provider is not None:
            return self.provider.generate(question, context)
        if self.engine is not None:
            return self.engine._provider.generate(question, context)  # type: ignore[attr-defined]
        return f"Based on available context:\n{context[:2000]}" if context else "No context available."

    def run(self, agent: dict[str, Any], user_input: str, run_id: str = "") -> dict[str, Any]:
        store = self.store
        model = agent.get("model", "gpt-4o-mini") or "gpt-4o-mini"
        system_prompt = agent.get("system_prompt", "") or ""

        if not run_id:
            rec = store.create_run(agent["id"], user_input)
            run_id = rec["id"]
        t0 = time.perf_counter()
        total_in = total_out = 0
        total_cost = 0.0
        status = "success"
        output = ""
        grounded = False
        root = store.create_span(run_id, "run", f"RUN {run_id[:12]}",
                                 input_data=user_input[:4000],
                                 metadata={"agent_id": agent["id"], "model": model})

        try:
            # ---- LLM Call #1: plan / rewrite query ----
            s1 = store.create_span(run_id, "llm", "LLM Call #1 — plan", parent_id=root["id"],
                                   input_data=user_input[:4000],
                                   metadata={"model": model, "purpose": "plan"})
            t1 = time.perf_counter()
            plan_prompt = (
                f"System: {system_prompt}\n\nRewrite the user question as an effective "
                f"search query. User question: {user_input}"
            )
            try:
                plan_text = self._provider_generate(
                    f"Rewrite as search query: {user_input}",
                    f"Agent instructions: {system_prompt}",
                )
            except Exception:
                plan_text = user_input  # fall back to raw query
            lat1 = (time.perf_counter() - t1) * 1000
            in1, out1 = estimate_tokens(plan_prompt), estimate_tokens(plan_text)
            c1 = estimate_cost(model, in1, out1)
            store.add_llm_call(run_id, s1["id"], model, plan_prompt[:6000],
                               plan_text[:6000], in1, out1, round(lat1, 2), c1)
            store.finish_span(s1["id"], "success", plan_text[:6000],
                              {"model": model, "input_tokens": in1,
                               "output_tokens": out1, "cost": c1, "latency_ms": round(lat1, 2)})
            total_in += in1
            total_out += out1
            total_cost += c1
            search_query = (plan_text.strip()[:500] or user_input)

            # ---- Tool Call: WebSearch (docs retriever) ----
            s2 = store.create_span(run_id, "tool", "Tool: WebSearch", parent_id=root["id"],
                                   input_data=search_query[:2000],
                                   metadata={"tool": "WebSearch"})
            t2 = time.perf_counter()
            tool_status = "success"
            search_output = ""
            hits: list = []
            try:
                if self.engine is not None:
                    retriever = self.engine._retriever  # type: ignore[attr-defined]
                    top_k = getattr(getattr(self.engine, "_settings", None), "top_k", 4)
                    hits = retriever.search(search_query, top_k)
                    lines = [f"[{h.chunk.doc} — {h.chunk.title} | score={h.score:.3f}]\n{h.chunk.text[:800]}"
                             for h in hits]
                    search_output = "\n\n---\n\n".join(lines) if lines else "No results found."
                else:
                    search_output = "No retriever configured; skipping search."
            except Exception as exc:
                tool_status = "error"
                search_output = f"Tool error: {exc}"
            lat2 = (time.perf_counter() - t2) * 1000
            store.add_tool_call(run_id, s2["id"], "WebSearch", search_query[:2000],
                                search_output[:6000], round(lat2, 2), tool_status)
            store.finish_span(s2["id"], tool_status, search_output[:6000],
                              {"tool": "WebSearch", "latency_ms": round(lat2, 2),
                               "num_hits": len(hits)})

            # ---- Tool Call: VectorSearch (refined pass) ----
            s2b = store.create_span(run_id, "tool", "Tool: VectorSearch", parent_id=root["id"],
                                    input_data=user_input[:2000],
                                    metadata={"tool": "VectorSearch"})
            t2b = time.perf_counter()
            vec_output = ""
            try:
                if self.engine is not None and hits:
                    vec_output = f"Re-ranked {len(hits)} passages; top score={max(h.score for h in hits):.3f}."
                else:
                    vec_output = search_output[:2000] or "No candidates to rank."
                vec_status = "success"
            except Exception as exc:
                vec_output = f"Tool error: {exc}"
                vec_status = "error"
            lat2b = (time.perf_counter() - t2b) * 1000
            store.add_tool_call(run_id, s2b["id"], "VectorSearch", user_input[:2000],
                                vec_output[:6000], round(lat2b, 2), vec_status)
            store.finish_span(s2b["id"], vec_status, vec_output[:6000],
                              {"tool": "VectorSearch", "latency_ms": round(lat2b, 2)})

            # ---- LLM Call #2: synthesize final answer ----
            s3 = store.create_span(run_id, "llm", "LLM Call #2 — synthesize", parent_id=root["id"],
                                   input_data=user_input[:4000],
                                   metadata={"model": model, "purpose": "synthesize"})
            t3 = time.perf_counter()
            context = search_output[:6000] if search_output else ""
            synth_prompt = build_user_prompt(user_input, context) if context else user_input
            if system_prompt:
                synth_prompt = f"System: {system_prompt}\n\n{synth_prompt}"
            try:
                final_answer = self._provider_generate(user_input, context)
                # Detect refusal sentinel from the none/LLM providers
                from .llm import OUT_OF_SCOPE_TOKEN  # local import to avoid cycle
                if OUT_OF_SCOPE_TOKEN in (final_answer or ""):
                    grounded = False
                    # surface friendly refusal instead of sentinel
                    try:
                        refusal = getattr(getattr(self.engine, "_settings", None),
                                          "refusal_message", None)
                        output = refusal or "I couldn't find anything relevant in the docs."
                    except Exception:
                        output = "I couldn't find anything relevant in the docs."
                else:
                    output = (final_answer or "").strip()
                    grounded = bool(context and "No results" not in context and output)
            except Exception as exc:
                status = "error"
                output = f"Agent execution failed: {redact(str(exc))[:500]}"
            lat3 = (time.perf_counter() - t3) * 1000
            in3, out3 = estimate_tokens(synth_prompt), estimate_tokens(output)
            c3 = estimate_cost(model, in3, out3)
            store.add_llm_call(run_id, s3["id"], model, synth_prompt[:6000],
                               output[:6000], in3, out3, round(lat3, 2), c3)
            store.finish_span(s3["id"], "success" if status == "success" else "error",
                              output[:6000],
                              {"model": model, "input_tokens": in3, "output_tokens": out3,
                               "cost": c3, "latency_ms": round(lat3, 2)})
            total_in += in3
            total_out += out3
            total_cost += c3

        except Exception as exc:  # defensive: never leave a run stuck in 'running'
            status = "error"
            output = f"Agent execution failed: {redact(str(exc))[:500]}"

        latency_ms = (time.perf_counter() - t0) * 1000
        total_tokens = total_in + total_out
        store.finish_run(run_id, output, status, round(latency_ms, 2),
                         total_tokens, round(total_cost, 6), grounded)
        store.finish_span(root["id"], status, output[:6000],
                          {"total_tokens": total_tokens, "cost": round(total_cost, 6),
                           "latency_ms": round(latency_ms, 2)})
        return store.get_run(run_id) or {}
