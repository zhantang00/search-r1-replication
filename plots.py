

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
COLORS = {"grpo": "tab:blue", "ppo": "tab:orange"}
LABELS = {"grpo": "GRPO (32 q x 5)", "ppo": "PPO (64 q x 1)"}
PAPER = {"direct": 0.024, "grpo": 0.232, "ppo": 0.264}   # Qwen2.5-3B-Instruct, Bamboogle
MAIN_STEP = 100                                         # pre-planned main checkpoint
# Checkpoints chosen by the paper's rule for the 300-step extension: the final
# checkpoint, or the last stable one if the training reward diverged.
RULE_SELECTED = {"grpo": 280, "ppo": 300}
WINDOW = 10                                             # moving-average window (steps)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_metrics(run: str) -> list[dict]:
    """One row per step; if a step was logged twice (after a restart), keep the last."""
    by_step: dict[int, dict] = {}
    with open(RESULTS / "metrics" / f"{run}_metrics.jsonl") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                by_step[row["step"]] = row
    return [by_step[s] for s in sorted(by_step)]


def load_eval(name: str) -> tuple[list[dict] | None, dict] | None:
    """(per-question rows or None, summary), or None if there is no summary."""
    rows_path = RESULTS / "eval" / f"{name}.jsonl"
    summary_path = RESULTS / "eval" / f"{name}.summary.json"
    if not summary_path.exists():
        return None
    rows = ([json.loads(line) for line in open(rows_path) if line.strip()]
            if rows_path.exists() else None)
    return rows, json.load(open(summary_path))


def checkpoint_evals(run: str) -> list[tuple[int, str]]:
    """(step, eval name) for every evaluated checkpoint of a run, step 0 = untrained."""
    found = [(0, "step0"), (MAIN_STEP, run)]
    for path in sorted((RESULTS / "eval").glob(f"{run}_step*.summary.json")):
        name = path.name.removesuffix(".summary.json")
        found.append((int(name.split("_step")[1]), name))
    return sorted(s for s in found if load_eval(s[1]) is not None)


def moving_average(values: list[float], window: int = WINDOW) -> list[float]:
    out, total = [], 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def mcnemar_exact(a: list[dict], b: list[dict]) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired per-question EM (same 125 questions).

    Returns (questions only A got right, questions only B got right, p-value).
    """
    em_a = {r["id"]: r["em"] for r in a}
    em_b = {r["id"]: r["em"] for r in b}
    ids = em_a.keys() & em_b.keys()
    only_a = sum(em_a[i] == 1 and em_b[i] == 0 for i in ids)
    only_b = sum(em_a[i] == 0 and em_b[i] == 1 for i in ids)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n
    return only_a, only_b, min(1.0, 2 * tail)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    pairs = [(r["step"], r[key]) for r in rows if r.get(key) is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _plot_runs(ax, metrics: dict, key: str, title: str, raw: bool = False) -> None:
    for run, rows in metrics.items():
        steps, values = _series(rows, key)
        if raw:
            ax.plot(steps, values, color=COLORS[run], alpha=0.2, linewidth=0.8)
        ax.plot(steps, moving_average(values), color=COLORS[run], linewidth=2, label=LABELS[run])
    ax.axvline(MAIN_STEP, color="gray", linestyle=":", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.grid(alpha=0.3)


def plot_training(metrics: dict) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _plot_runs(ax, metrics, "reward_mean", f"Training reward (EM on NQ+HotpotQA), {WINDOW}-step average",
               raw=True)
    ax.text(MAIN_STEP + 3, ax.get_ylim()[0] + 0.01, "main checkpoints (step 100)", color="gray", fontsize=8)
    ax.set_ylabel("mean EM reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_train_reward.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    _plot_runs(axes[0], metrics, "searches_mean", "Searches per question")
    _plot_runs(axes[1], metrics, "gen_tokens_mean", "Generated tokens per trajectory")
    _plot_runs(axes[2], metrics, "answer_rate", "Answer rate (finished with <answer>)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_behavior.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _plot_runs(axes[0], metrics, "kl", "KL to the reference policy (k3 estimate)")
    axes[0].legend()
    ppo = metrics.get("ppo", [])
    for key, style, label in [("explained_var", "-", "explained variance"),
                              ("value_mean", "--", "mean value (critic)"),
                              ("return_mean", ":", "mean return")]:
        steps, values = _series(ppo, key)
        axes[1].plot(steps, moving_average(values), linestyle=style, color=COLORS["ppo"],
                     linewidth=2, label=label)
    axes[1].axvline(MAIN_STEP, color="gray", linestyle=":", linewidth=1)
    axes[1].set_title("PPO critic")
    axes[1].set_xlabel("training step")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_kl_critic.png", dpi=150)
    plt.close(fig)


def plot_bamboogle() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for i, run in enumerate(["grpo", "ppo"]):
        points = checkpoint_evals(run)
        steps = [s + (i * 3 - 1.5) for s, _ in points]               # small offset: no overlap
        summaries = [load_eval(name)[1] for _, name in points]
        em = [s["em"] for s in summaries]
        err = [[s["em"] - s["ci95"][0] for s in summaries], [s["ci95"][1] - s["em"] for s in summaries]]
        ax.errorbar(steps, em, yerr=err, color=COLORS[run], marker="o", capsize=3, linewidth=2,
                    label=f"{LABELS[run]} (ours, LoRA)")
        ax.axhline(PAPER[run], color=COLORS[run], linestyle="--", linewidth=1,
                   label=f"paper {run.upper()} ({PAPER[run]:.3f})")
        chosen = [(x, e) for (step, _), x, e in zip(points, steps, em) if step == RULE_SELECTED[run]]
        for x, e in chosen:
            ax.plot(x, e, marker="*", markersize=16, color=COLORS[run], markeredgecolor="black",
                    linestyle="none", label=f"{run.upper()} checkpoint by the paper's rule")
    ax.axhline(PAPER["direct"], color="gray", linestyle=":", linewidth=1, label="Direct Inference (0.024)")
    ax.set_title("Bamboogle EM by training checkpoint (95% bootstrap CI, n = 125)")
    ax.set_xlabel("training step (0 = untrained model with search)")
    ax.set_ylabel("exact match")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))   # outside: no overlap
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_bamboogle.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _row(label: str, name: str, paper: str = "") -> str | None:
    loaded = load_eval(name)
    if loaded is None:
        return None
    s = loaded[1]
    turn_limit = s["finish_counts"].get("max_turns", 0)
    return (f"| {label} | **{s['em']:.3f}** ({s['correct']}/{s['n']}) | "
            f"{s['ci95'][0]:.3f}–{s['ci95'][1]:.3f} | {s['mean_searches']:.2f} | "
            f"{s['answer_rate']:.2f} | {turn_limit} | {paper} |")


def write_tables() -> str:
    header = ("| Method | Bamboogle EM | 95% CI | Searches / question | Answer rate | "
              "Hit turn limit | Paper |\n|---|---|---|---|---|---|---|")
    main = [
        _row("Direct Inference (no retrieval)", "direct", "0.024"),
        _row("Search-R1 template, untrained (step 0)", "step0", "—"),
        _row("**Search-R1-instruct, GRPO (LoRA, step 100)**", "grpo", "0.232 (Table 3)"),
        _row("**Search-R1-instruct, PPO (LoRA, step 100)**", "ppo", "0.264 (Table 2)"),
    ]
    reference = [
        _row("Official GRPO checkpoint (full fine-tune)", "official_grpo", "—"),
        _row("Official PPO v0.3 checkpoint (full fine-tune)", "official_ppo_v03", "—"),
    ]
    extension = []
    for run in ["grpo", "ppo"]:
        for step, name in checkpoint_evals(run):
            if step > MAIN_STEP:
                note = " (paper's rule)" if step == RULE_SELECTED[run] else ""
                extension.append(_row(f"{run.upper()} step {step}{note}", name))

    tests = [("grpo", "step0"), ("ppo", "step0"), ("grpo", "ppo"),
             ("official_grpo", "grpo"), ("official_ppo_v03", "ppo")]
    for run in ["grpo", "ppo"]:
        for step, name in checkpoint_evals(run):
            if step > MAIN_STEP:
                tests += [(name, run), (name, "step0")]
    test_rows = []
    for a, b in tests:
        la, lb = load_eval(a), load_eval(b)
        if la is None or lb is None or la[0] is None or lb[0] is None:
            continue
        only_a, only_b, p = mcnemar_exact(la[0], lb[0])
        verdict = "significant" if p < 0.05 else "not significant"
        test_rows.append(f"| {a} vs {b} | {la[1]['em']:.3f} vs {lb[1]['em']:.3f} | "
                         f"{only_a} | {only_b} | {p:.3f} | {verdict} |")

    text = "\n".join([
        "## Main results (Qwen2.5-3B-Instruct, Bamboogle, n = 125)", "", header,
        *[r for r in main if r], "",
        "## Reference: authors' released checkpoints, same evaluation code", "", header,
        *[r for r in reference if r], "",
        "## Extended training (all evaluated checkpoints; analysis, not selection)", "", header,
        *[r for r in extension if r], "",
        "## Paired significance tests (exact McNemar on the same 125 questions)", "",
        "| Comparison | EM | Only first right | Only second right | p | At 0.05 |",
        "|---|---|---|---|---|---|", *test_rows, "",
    ])
    (RESULTS / "results.md").write_text(text)
    return text


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    metrics = {run: load_metrics(run) for run in ["grpo", "ppo"]}
    plot_training(metrics)
    plot_bamboogle()
    print(write_tables())
    print(f"figures written to {FIGURES}/")


if __name__ == "__main__":
    main()
