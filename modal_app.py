
import sys
from pathlib import Path

import modal

SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))
from config import APP_DIR, HF_HOME, VOL  # noqa: E402

app = modal.App("search-r1-lora")


volume = modal.Volume.from_name("search-r1", create_if_missing=True, version=2)

image = (
    modal.Image.from_registry("nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libnuma1")
    .pip_install_from_requirements(str(SRC / "requirements.txt"))
    .run_commands(
        "python -m pip check",
        "python -c 'import torch, vllm, transformers, peft, faiss; "
        "print(torch.__version__, vllm.__version__, transformers.__version__, peft.__version__)'",
    )
    .env({"HF_HOME": HF_HOME, "PYTHONPATH": APP_DIR, "TOKENIZERS_PARALLELISM": "false"})
    .workdir(APP_DIR)
    .add_local_dir(SRC, remote_path=APP_DIR,
                   ignore=["**/__pycache__", "**/*.pyc", ".git", ".venv", "results"])
)


RETRIEVER_CPUS = 16


@app.cls(image=image, volumes={VOL: volume}, cpu=RETRIEVER_CPUS, memory=96 * 1024,
         timeout=24 * 3600, scaledown_window=20 * 60, max_containers=1)
class Retriever:
    """CPU-only search service. Loads the 65 GB index and 14 GB corpus into RAM once
    per container.
    """

    @modal.enter()
    def load(self):
        from config import Config
        from retriever import DenseRetriever, ensure_offsets

        if ensure_offsets():  # first start only: index the corpus file by byte offset
            volume.commit()
        self.retriever = DenseRetriever(topk=Config().topk, n_threads=RETRIEVER_CPUS)

    @modal.method()
    def search(self, queries: list[str]) -> list[str]:
        return self.retriever.search(queries)

    @modal.method()
    def agreement(self, queries: list[str]) -> float:
        return self.retriever.agreement(queries)


@app.function(image=image, volumes={VOL: volume}, cpu=4, memory=16384, timeout=4 * 3600)
def download_assets(skip_retrieval: bool = False) -> dict:
    """Prepare QA data, cache model weights, and fetch the corpus + index."""
    from data import download_models, download_retrieval_assets, prepare_qa_data

    stats = prepare_qa_data()
    download_models()
    volume.commit()
    print("QA data and models cached; direct evaluation can start now.")
    if not skip_retrieval:
        download_retrieval_assets()
        volume.commit()
    return stats



EVAL_GPUS = ["L4", "A10", "L40S", "A100-40GB"]


@app.function(image=image, volumes={VOL: volume}, gpu=EVAL_GPUS, cpu=4, memory=32768,
              timeout=2 * 3600)
def evaluate(mode: str = "direct", name: str = "", model: str = "", adapter: str = "",
             limit: int = 0) -> dict:
    """Evaluate on Bamboogle. Empty strings / 0 mean 'use the default'."""
    from evaluate import run_evaluation  # the evaluate.py module, not this function

    search_fn = Retriever().search.remote if mode == "search" else None
    summary = run_evaluation(mode=mode, name=name or None, model=model or None,
                             adapter=adapter or None, limit=limit or None,
                             search_fn=search_fn)
    volume.commit()
    return summary


def _check_retriever() -> int:
    """Run the paper's case-study queries; return how many gave the expected top-1 title."""
    from retriever import CASE_STUDY_QUERIES

    queries = list(CASE_STUDY_QUERIES)
    blocks = Retriever().search.remote(queries)
    hits = 0
    for query, block in zip(queries, blocks):
        first_line = block.split("\n")[0]
        ok = CASE_STUDY_QUERIES[query] in first_line
        hits += ok
        print(f"[{'OK  ' if ok else 'MISS'}] {query!r}\n       {first_line[:150]}")
    print(f"{hits}/{len(queries)} case-study queries returned the paper's top-1 title")
    return hits


@app.local_entrypoint()
def retriever_check():
    from data import read_bamboogle_csv
    from retriever import CASE_STUDY_QUERIES

    _check_retriever()
    print("\nFull result for the first query:\n" + Retriever().search.remote(
        ["Curious fragrance information"])[0])
    queries = list(CASE_STUDY_QUERIES) + [r["question"] for r in read_bamboogle_csv(
        str(SRC / "Bamboogle_Prerelease_-_Sheet1.csv"))]
    share = Retriever().agreement.remote(queries)
    print(f"\nOur search and faiss return the same top-3 for {share:.1%} of {len(queries)} queries")


@app.local_entrypoint()
def search_evals(names: str = "step0,official_grpo,official_ppo_v03", limit: int = 0):
    """Search-mode evaluations in parallel, sharing one warm retriever."""
    from config import OFFICIAL_CHECKPOINTS

    _check_retriever()
    calls = []
    for name in names.split(","):
        if name != "step0" and name not in OFFICIAL_CHECKPOINTS:
            raise ValueError(f"unknown eval name {name!r}")
        model = "" if name == "step0" else name
        calls.append(evaluate.spawn(mode="search", name=name, model=model, limit=limit))
    for call in calls:
        _print_summary(call.get())


def _print_summary(s: dict) -> None:
    print(f"{s['name']:>18}: EM {s['em']:.3f} ({s['correct']}/{s['n']}), "
          f"CI {s['ci95'][0]:.3f}-{s['ci95'][1]:.3f}, searches {s['mean_searches']:.2f}, "
          f"answer rate {s['answer_rate']:.2f}, finish {s['finish_counts']}")


@app.local_entrypoint()
def eval_checkpoints(runs: str = "grpo,ppo", steps: str = "200,300", limit: int = 0):
    """Evaluate saved training checkpoints in parallel, sharing one warm retriever."""
    from config import RUNS_DIR

    _check_retriever()
    calls = []
    for run in runs.split(","):
        for step in (int(x) for x in steps.split(",")):
            calls.append(evaluate.spawn(
                mode="search", name=f"{run}_step{step}", limit=limit,
                adapter=f"{RUNS_DIR}/{run}/ckpt/step_{step:04d}/policy"))
    for call in calls:
        _print_summary(call.get())


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
TRAIN_GPUS = ["H100", "A100-80GB"]


def _run_training(preset: str, run_name: str, steps: int) -> None:
    from config import get_config
    from train import train as run_train  # the train.py module, not the Modal function

    cfg = get_config(preset, steps=steps or None)
    run_name = run_name or (f"{preset}_smoke{steps}" if steps else preset)
    if cfg.use_fake_retriever:
        from retriever import FakeRetriever
        search_fn = FakeRetriever().search
    else:
        search_fn = Retriever().search.remote
    run_train(cfg, run_name, search_fn, commit=volume.commit)


@app.function(image=image, volumes={VOL: volume}, gpu=TRAIN_GPUS, cpu=8, memory=64 * 1024,
              timeout=24 * 3600, retries=modal.Retries(max_retries=2, initial_delay=30.0))
def train(preset: str = "grpo", run_name: str = "", steps: int = 0) -> None:
    """Train on an H100 or A100-80GB. After a crash or preemption, Modal retries
    and the run resumes from runs/<run_name>/latest."""
    _run_training(preset, run_name, steps)


@app.function(image=image, volumes={VOL: volume}, gpu=EVAL_GPUS, cpu=4, memory=32 * 1024,
              timeout=2 * 3600)
def train_debug(preset: str = "debug_grpo", run_name: str = "", steps: int = 0) -> None:
    """Plumbing test on a small GPU: Qwen2.5-0.5B, fake retriever, 5 steps."""
    _run_training(preset, run_name, steps)