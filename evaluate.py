

from __future__ import annotations

import json
import os
import time

from collections import Counter
from collections.abc import Callable

from config import BAMBOOGLE_PATH, EVAL_DIR, OFFICIAL_CHECKPOINTS, Config
from data import load_jsonl, write_jsonl
from rollout import generate_direct, generate_search, make_llm
from scoring import score_trajectory


def _gpu_name() -> str:
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def bootstrap_ci(scores: list[float], n_boot: int = 10_000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of per-question scores."""
    import numpy as np

    arr = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def run_evaluation(mode: str = "direct", name: str | None = None, model: str | None = None,
                   adapter: str | None = None, limit: int | None = None,
                   search_fn: Callable[[list[str]], list[str]] | None = None) -> dict:
    cfg = Config()
    model = OFFICIAL_CHECKPOINTS.get(model, model) if model else cfg.model_name
    # Check the adapter before vLLM spends minutes starting up on a paid GPU.
    if adapter and not os.path.exists(os.path.join(adapter, "adapter_config.json")):
        raise FileNotFoundError(
            f"No LoRA adapter at {adapter} (adapter_config.json is missing). Check it with: "
            f"modal volume ls search-r1 {adapter.removeprefix('/vol/')}")
    name = name or mode
    questions = load_jsonl(BAMBOOGLE_PATH)
    if limit:
        questions = questions[:limit]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = make_llm(model, cfg, cfg.eval_vllm_gpu_mem, enable_lora=adapter is not None)
    lora_request = None
    if adapter:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("eval_adapter", 1, adapter)

    if mode == "direct":
        trajs = generate_direct(llm, tokenizer, questions, cfg, lora_request)
        fallback_to_text = True
    elif mode == "search":
        if search_fn is None:
            raise ValueError("search mode needs a search_fn")
        trajs = generate_search(llm, tokenizer, questions, cfg, search_fn,
                                n_samples=1, temperature=0.0, lora_request=lora_request)
        fallback_to_text = False
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    for traj in trajs:
        score_trajectory(traj, fallback_to_text=fallback_to_text)

    rows = [{
        "id": t.qid, "question": t.question, "golden_answers": t.golden_answers,
        "prediction": t.prediction, "em": t.reward, "n_turns": t.n_turns,
        "n_searches": t.n_searches, "finish": t.finish, "response": t.response_text,
    } for t in trajs]
    scores = [r["em"] for r in rows]
    lo, hi = bootstrap_ci(scores)
    summary = {
        "name": name, "mode": mode, "model": model, "adapter": adapter,
        "dataset": "bamboogle", "n": len(rows), "correct": int(sum(scores)),
        "em": sum(scores) / len(rows), "ci95": [lo, hi],
        "mean_searches": sum(r["n_searches"] for r in rows) / len(rows),
        "mean_turns": sum(r["n_turns"] for r in rows) / len(rows),
        "answer_rate": sum(t.finish == "answer" for t in trajs) / len(rows),
        "finish_counts": dict(Counter(t.finish for t in trajs)),
        "gpu": _gpu_name(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    os.makedirs(EVAL_DIR, exist_ok=True)
    write_jsonl(f"{EVAL_DIR}/{name}.jsonl", rows)
    with open(f"{EVAL_DIR}/{name}.summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    first = rows[0]
    print(f"\n=== Full first trajectory ===\nQ: {first['question']}\nGold: {first['golden_answers']}\n"
          f"{first['response'][:4000]}\nPrediction: {first['prediction']!r}  EM: {first['em']}")
    for row in rows[1:4]:
        print(f"\nQ: {row['question']}\nGold: {row['golden_answers']}\n"
              f"Prediction: {row['prediction']!r}  searches: {row['n_searches']}  EM: {row['em']}")
    print("\n" + json.dumps(summary, indent=2))
    return summary