"""Simple evaluator: accuracy, relevance, faithfulness, instruction-following.

Strategy: fast deterministic heuristics by default (no API key needed).
If an LLM provider is available (not the 'none' provider), an LLM-judge
pass refines the scores; heuristic is always the fallback.
"""

from __future__ import annotations

import re
from typing import Any

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def heuristic_scores(run_input: str, run_output: str, traces: list[dict],
                     llm_calls: list[dict], tool_calls: list[dict]) -> tuple[dict[str, float], dict]:
    q, a = run_input or "", run_output or ""
    qt, at = _tokens(q), _tokens(a)

    # Relevance: question↔answer word overlap (+ retrieval hit bonus)
    relevance = _overlap(qt, at)
    if tool_calls and any("No results" not in (t.get("output") or "") for t in tool_calls):
        relevance = min(1.0, relevance + 0.25)
    relevance = round(max(0.05, min(1.0, relevance if at else 0.0)), 4)

    # Faithfulness: answer grounded in tool outputs?
    tool_text = " ".join(t.get("output", "") or "" for t in tool_calls)
    faithfulness = _overlap(at, _tokens(tool_text)) if tool_text.strip() else (0.5 if at else 0.0)
    if "failed" in a.lower() or "error" in a.lower():
        faithfulness = min(faithfulness, 0.3)
    faithfulness = round(max(0.0, min(1.0, faithfulness)), 4)

    # Accuracy: non-empty + grounded + reasonable length
    if not a.strip() or "failed" in a.lower()[:60]:
        accuracy = 0.1
    elif len(a.strip()) < 20:
        accuracy = 0.4
    elif faithfulness > 0.5 and relevance > 0.3:
        accuracy = 0.9
    elif faithfulness > 0.3:
        accuracy = 0.7
    else:
        accuracy = 0.5
    accuracy = round(accuracy, 4)

    # Instruction following: answered (not refused), on-topic, well-formed
    refused = any(p in a.lower() for p in
                  ["couldn't find anything", "only answer", "i'm sorry, but i can only"])
    if not a.strip():
        instruction = 0.1
    elif refused:
        instruction = 0.45  # correctly followed docs-only constraint
    elif relevance > 0.4 and len(a) > 50:
        instruction = 0.9
    else:
        instruction = 0.65
    instruction = round(instruction, 4)

    scores = {
        "accuracy": accuracy,
        "relevance": relevance,
        "faithfulness": faithfulness,
        "instruction_following": instruction,
    }
    detail = {
        "question_tokens": len(qt),
        "answer_tokens": len(at),
        "num_llm_calls": len(llm_calls),
        "num_tool_calls": len(tool_calls),
        "refused": refused,
    }
    return scores, detail


def llm_judge_scores(provider: Any, run_input: str, run_output: str) -> tuple[dict[str, float] | None, str]:
    """Ask the LLM to score 0-1 on the four axes. Returns (scores, raw) or (None, raw)."""
    try:
        judge_q = (
            "Score the following agent answer on four axes, each 0.0-1.0. "
            "Reply with exactly four lines like 'accuracy: 0.8'."
        )
        ctx = f"Question: {run_input}\nAnswer: {run_output}\nAxes: accuracy, relevance, faithfulness, instruction_following."
        raw = provider.generate(judge_q, ctx)
        scores: dict[str, float] = {}
        for line in (raw or "").splitlines():
            m = re.match(r"\s*(accuracy|relevance|faithfulness|instruction[_\s-]?following)\s*[:=]\s*([0-9]*\.?[0-9]+)", line, re.I)
            if m:
                key = "instruction_following" if "instruction" in m.group(1).lower() else m.group(1).lower()
                scores[key] = max(0.0, min(1.0, float(m.group(2))))
        if len(scores) >= 3:
            for k in ("accuracy", "relevance", "faithfulness", "instruction_following"):
                scores.setdefault(k, 0.5)
            return {k: round(float(scores[k]), 4) for k in
                    ("accuracy", "relevance", "faithfulness", "instruction_following")}, raw
        return None, raw or ""
    except Exception as exc:
        return None, f"judge error: {exc}"


def evaluate_run(store, run: dict, provider: Any = None) -> dict:
    run_id = run["id"]
    traces = store.get_traces(run_id)
    llm_calls = store.get_llm_calls(run_id)
    tool_calls = store.get_tool_calls(run_id)
    scores, detail = heuristic_scores(run.get("input", ""), run.get("output", ""),
                                      traces, llm_calls, tool_calls)
    method = "heuristic"
    # LLM judge refinement when a real provider is configured
    provider_name = ""
    try:
        provider_name = type(provider).__name__ if provider else ""
    except Exception:
        provider_name = ""
    if provider is not None and provider_name != "NoneProvider":
        judged, raw = llm_judge_scores(provider, run.get("input", ""), run.get("output", ""))
        if judged:
            # average heuristic + judge for stability
            scores = {k: round((scores[k] + judged[k]) / 2, 4) for k in scores}
            method = "llm_judge+heuristic"
            detail["judge_raw"] = (raw or "")[:2000]
        else:
            detail["judge_raw"] = (raw or "")[:2000]
    detail["provider"] = provider_name
    return store.save_evaluation(run_id, scores, method=method, detail=detail)
