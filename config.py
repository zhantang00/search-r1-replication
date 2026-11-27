
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Paths on the Modal Volume (mounted at /vol in every container)
# --------------------------------------------------------------------------- #
VOL = "/vol"
ASSETS_DIR = f"{VOL}/assets"   # corpus, index, raw downloads
DATA_DIR = f"{VOL}/data"       # our processed JSONL files
RUNS_DIR = f"{VOL}/runs"       # training runs (batch 3+)
EVAL_DIR = f"{VOL}/eval"       # evaluation outputs
HF_HOME = f"{VOL}/hf_cache"    # Hugging Face model cache (set as env var in the image)

TRAIN_PATH = f"{DATA_DIR}/train.jsonl"
BAMBOOGLE_PATH = f"{DATA_DIR}/bamboogle.jsonl"
SEARCH_TEMPLATE_PATH = f"{DATA_DIR}/search_r1_template.txt"
INDEX_PATH = f"{ASSETS_DIR}/e5_Flat.index"
CORPUS_PATH = f"{ASSETS_DIR}/wiki-18.jsonl"
# Byte offset of each passage line (blank lines skipped). "v2": the first
# version counted a blank line as a passage; the new name forces a rebuild.
CORPUS_OFFSETS_PATH = f"{ASSETS_DIR}/wiki-18.offsets.v2.npy"

# Path inside the container where modal_app.py places the repo (incl. the CSV).
APP_DIR = "/app"
BAMBOOGLE_CSV = f"{APP_DIR}/Bamboogle_Prerelease_-_Sheet1.csv"

# --------------------------------------------------------------------------- #
# External resources
# --------------------------------------------------------------------------- #
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEBUG_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
RETRIEVER_MODEL = "intfloat/e5-base-v2"
TRAIN_DATA_REPO = "PeterJinGo/nq_hotpotqa_train"   # official processed NQ+HotpotQA
INDEX_REPO = "PeterJinGo/wiki-18-e5-index"         # files: part_aa, part_ab
INDEX_PARTS = ("part_aa", "part_ab")
CORPUS_REPO = "PeterJinGo/wiki-18-corpus"          # file: wiki-18.jsonl.gz
CORPUS_FILE = "wiki-18.jsonl.gz"

# Authors' released checkpoints, used in batch 2 to validate the eval harness.
OFFICIAL_CHECKPOINTS = {
    "official_grpo": "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo",
    "official_ppo_v03": "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo-v0.3",
}


# --------------------------------------------------------------------------- #
# Run settings
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    name: str = "grpo"
    algo: str = "grpo"                  # "grpo" or "ppo"
    model_name: str = BASE_MODEL
    max_model_len: int = 4096           # paper App. B: max sequence length 4,096

    # Rollout (paper App. B)
    max_turns: int = 4                  # action budget B
    topk: int = 3                       # passages per search
    max_gen_tokens: int = 500           # tokens per generation call
    max_info_tokens: int = 500          # tokens per <information> block
    train_temperature: float = 1.0      # paper: temperature 1.0, top-p 1.0
    direct_max_tokens: int = 256        # direct-inference baseline only
    use_fake_retriever: bool = False    # debug runs only

    # LoRA (policy and critic)
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0           # 0 keeps train-time and rollout-time logprobs identical
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )

    # Optimization
    questions_per_step: int = 64
    samples_per_question: int = 5       # GRPO group size (paper: 5); PPO uses 1
    steps: int = 150
    mini_batches: int = 2               # optimizer steps per rollout batch (paper: 512 / 256)
    micro_batch_size: int = 4           # sequences per forward/backward pass
    policy_lr: float = 1e-5             # paper 1e-6 for full fine-tuning; ~10x for LoRA
    critic_lr: float = 1e-4             # paper 1e-5 (10x the policy's); same 10x for LoRA
    policy_warmup_frac: float = 0.285   # paper App. B
    critic_warmup_frac: float = 0.015   # paper App. B
    kl_beta: float = 0.001
    clip_eps: float = 0.2
    value_clip_eps: float = 0.5         # PPO critic value clipping (verl default)
    gae_gamma: float = 1.0              # paper App. B: gamma = lambda = 1
    gae_lambda: float = 1.0
    max_grad_norm: float = 1.0

    # Bookkeeping
    ckpt_every: int = 10                # a crash or preemption loses at most 10 steps
    sample_log_every: int = 5
    seed: int = 0
    train_vllm_gpu_mem: float = 0.30    # vLLM share of the GPU during training
    eval_vllm_gpu_mem: float = 0.85     # vLLM share of the GPU during evaluation


PRESETS: dict[str, dict] = {
    # GRPO sized from the H100 smoke test: 64 x 5 took ~275 s/step, so
    # 32 x 5 for 100 steps keeps the run to roughly 4-5 hours.
    "grpo": dict(algo="grpo", questions_per_step=32, samples_per_question=5, steps=100),
    # PPO: one sample per question (paper). 64 questions per step keeps the
    # critic's batch reasonable while costing far less than a GRPO step.
    "ppo": dict(algo="ppo", questions_per_step=64, samples_per_question=1, steps=100),
    # fallback
    "grpo_min": dict(algo="grpo", questions_per_step=32, samples_per_question=5, steps=100),
    "ppo_min": dict(algo="ppo", questions_per_step=32, samples_per_question=1, steps=100),
    # Plumbing tests on a small GPU with a fake retriever
    "debug_grpo": dict(algo="grpo", model_name=DEBUG_MODEL, use_fake_retriever=True,
                       questions_per_step=4, samples_per_question=2, steps=5,
                       ckpt_every=2, sample_log_every=1),
    "debug_ppo": dict(algo="ppo", model_name=DEBUG_MODEL, use_fake_retriever=True,
                      questions_per_step=4, samples_per_question=1, steps=5,
                      ckpt_every=2, sample_log_every=1),
}


def get_config(name: str = "grpo", **overrides) -> Config:
    """Build a Config from a preset name plus keyword overrides (None values are ignored)."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}")
    values = {**PRESETS[name], **{k: v for k, v in overrides.items() if v is not None}}
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    return Config(name=name, **values)


def config_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)