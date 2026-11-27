from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from collections.abc import Callable

from config import RUNS_DIR, TRAIN_PATH, Config, config_to_dict
from data import load_jsonl
from rollout import Trajectory, generate_search, make_llm
from scoring import score_trajectory

LORA_TMP = "/tmp/lora"  # adapters handed to vLLM each step (local disk, not the Volume)


# --------------------------------------------------------------------------- #
# Tensors from trajectories
# --------------------------------------------------------------------------- #
def _pad(rows, width: int, value: float = 0.0):
    """Stack variable-length rows (lists or 1-D tensors) into a [n, width] float tensor."""
    import torch
    out = torch.full((len(rows), width), value, dtype=torch.float32)
    for i, row in enumerate(rows):
        out[i, :len(row)] = torch.as_tensor(row, dtype=torch.float32)
    return out


def _pack(trajs: list[Trajectory], pad_id: int, device):
    """Right-padded prompt+response ids, attention mask, and for every response
    token the index of the position that predicts it.

    Logits (or hidden states) at position p predict token p + 1, so response
    token j, at absolute position P + j, is predicted at P + j - 1.
    Returns ids [n, width], attn [n, width], pos [n, R_max].
    """
    import torch
    seqs = [t.prompt_ids + t.response_ids for t in trajs]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        attn[i, :len(s)] = 1
    r_max = max(len(t.response_ids) for t in trajs)
    starts = torch.tensor([len(t.prompt_ids) - 1 for t in trajs])
    pos = (starts[:, None] + torch.arange(r_max)[None]).clamp(max=width - 2)
    return ids.to(device), attn.to(device), pos.to(device)


def _response_logprobs(model, trajs: list[Trajectory], pad_id: int):
    """Log-prob of every response token under `model`: [n, R_max]."""
    from algorithms import token_logprobs

    ids, attn, pos = _pack(trajs, pad_id, next(model.parameters()).device)
    logits = model(input_ids=ids, attention_mask=attn).logits
    full = token_logprobs(logits[:, :-1], ids[:, 1:])           # [n, width - 1]
    return full.gather(1, pos)


def _response_values(critic, trajs: list[Trajectory], pad_id: int):
    """Critic value of the state just before each response token: [n, R_max]."""
    ids, attn, pos = _pack(trajs, pad_id, next(critic.parameters()).device)
    return critic(ids, attn).gather(1, pos)


def _micro_batches(indices: list[int], trajs: list[Trajectory], size: int) -> list[list[int]]:
    """Sort by length (less padding), longest first, then chunk.

    Longest first: the biggest GPU allocations happen first and shorter batches
    reuse those cached blocks. Shortest first made every batch need a slightly
    larger block, so the allocator's cache grew until the GPU was full.
    """
    ordered = sorted(indices, key=lambda i: len(trajs[i].prompt_ids) + len(trajs[i].response_ids),
                     reverse=True)
    return [ordered[k:k + size] for k in range(0, len(ordered), size)]


def _mini_batches(n: int, count: int, rng: random.Random) -> list[list[int]]:
    order = list(range(n))
    rng.shuffle(order)
    return [order[k::count] for k in range(count)]


def _for_all(fn, trajs: list[Trajectory], size: int) -> list:
    """Run fn (micro-batch of trajectories -> [n, R_max]) over every trajectory
    without gradients. Returns per-trajectory CPU tensors of response length."""
    import torch

    out: list = [None] * len(trajs)
    with torch.no_grad():
        for mb in _micro_batches(list(range(len(trajs))), trajs, size):
            values = fn([trajs[i] for i in mb]).float().cpu()
            for row, i in enumerate(mb):
                out[i] = values[row, :len(trajs[i].response_ids)]
    return out


# --------------------------------------------------------------------------- #
# Critic (PPO)
# --------------------------------------------------------------------------- #
def _build_critic(cfg: Config, resume_dir: str | None):
    """A LoRA-adapted copy of the base model (no LM head) plus a scalar value head.

    The head starts at zero, so initial values are 0 and the first advantages
    equal the returns. The critic reads the same tokens as the policy, retrieved
    text included; only generated positions are used for its loss.
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModel

    class Critic(torch.nn.Module):
        def __init__(self, backbone, hidden: int):
            super().__init__()
            self.backbone = backbone
            self.head = torch.nn.Linear(hidden, 1)
            torch.nn.init.zeros_(self.head.weight)
            torch.nn.init.zeros_(self.head.bias)

        def forward(self, input_ids, attention_mask):
            hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            return self.head(hidden.float()).squeeze(-1)           # [n, width]

    backbone = AutoModel.from_pretrained(cfg.model_name, dtype=torch.bfloat16,
                                         attn_implementation="sdpa").to("cuda")
    backbone.config.use_cache = False
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if resume_dir:
        backbone = PeftModel.from_pretrained(backbone, f"{resume_dir}/critic", is_trainable=True)
    else:
        backbone = get_peft_model(backbone, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules), task_type="FEATURE_EXTRACTION",
            bias="none"))
    critic = Critic(backbone, backbone.config.hidden_size).to("cuda")
    if resume_dir:
        critic.head.load_state_dict(torch.load(f"{resume_dir}/value_head.pt"))
    critic.eval()
    return critic


def _ppo_advantages(trajs, rewards, old_lps, ref_lps, values, cfg: Config):
    """KL-in-reward, GAE over generated tokens, whitened advantages.
    Returns per-trajectory advantage rows, return rows, and stats."""
    from algorithms import explained_variance, gae, masked_mean, masked_whiten, ppo_token_rewards

    width = max(len(t.response_ids) for t in trajs)
    mask = _pad([t.loss_mask for t in trajs], width)
    old, ref, vals = _pad(old_lps, width), _pad(ref_lps, width), _pad(values, width)
    token_rewards = ppo_token_rewards(rewards.float(), old, ref, mask, cfg.kl_beta)
    adv, ret = gae(token_rewards, vals, mask, cfg.gae_gamma, cfg.gae_lambda)
    stats = {"explained_var": explained_variance(vals, ret, mask),
             "value_mean": masked_mean(vals, mask).item(),
             "return_mean": masked_mean(ret, mask).item()}
    adv = masked_whiten(adv, mask)
    lengths = [len(t.response_ids) for t in trajs]
    return ([adv[i, :n] for i, n in enumerate(lengths)],
            [ret[i, :n] for i, n in enumerate(lengths)], stats)


def _critic_update(critic, optimizer, trajs, values_old, returns, cfg: Config,
                   pad_id: int, rng: random.Random) -> dict:
    import torch
    from algorithms import masked_mean, value_loss

    device = next(critic.parameters()).device
    params = [p for p in critic.parameters() if p.requires_grad]
    minis = _mini_batches(len(trajs), cfg.mini_batches, rng)
    sums = {"value_loss": 0.0, "critic_grad_norm": 0.0}
    critic.train()
    for mini in minis:
        optimizer.zero_grad(set_to_none=True)
        for mb in _micro_batches(mini, trajs, cfg.micro_batch_size):
            batch = [trajs[i] for i in mb]
            width = max(len(t.response_ids) for t in batch)
            mask = _pad([t.loss_mask for t in batch], width).to(device)
            v_old = _pad([values_old[i] for i in mb], width).to(device)
            ret = _pad([returns[i] for i in mb], width).to(device)
            v = _response_values(critic, batch, pad_id)
            per_traj = masked_mean(value_loss(v, v_old, ret, cfg.value_clip_eps), mask, dim=1)
            (per_traj.sum() / len(mini)).backward()
            sums["value_loss"] += per_traj.sum().item() / len(trajs)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        sums["critic_grad_norm"] += float(grad_norm) / len(minis)
        optimizer.step()
    critic.eval()
    return sums


# --------------------------------------------------------------------------- #
# Policy update (GRPO and PPO)
# --------------------------------------------------------------------------- #
def _policy_update(policy, optimizer, trajs, adv_rows, old_lps, ref_lps, cfg: Config,
                   pad_id: int, rng: random.Random, kl_in_loss: bool) -> dict:
    """Clipped surrogate averaged over generated tokens within each trajectory,
    then over trajectories. GRPO adds beta * KL to the loss;
    for PPO the KL is already in the reward, so here it is only measured."""
    import torch
    from algorithms import kl_k3, masked_mean, policy_loss

    device = next(policy.parameters()).device
    params = [p for p in policy.parameters() if p.requires_grad]
    minis = _mini_batches(len(trajs), cfg.mini_batches, rng)
    sums = {"policy_loss": 0.0, "kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0}

    policy.train()
    for mini in minis:
        optimizer.zero_grad(set_to_none=True)
        for mb in _micro_batches(mini, trajs, cfg.micro_batch_size):
            batch = [trajs[i] for i in mb]
            width = max(len(t.response_ids) for t in batch)
            mask = _pad([t.loss_mask for t in batch], width).to(device)
            old = _pad([old_lps[i] for i in mb], width).to(device)
            ref = _pad([ref_lps[i] for i in mb], width).to(device)
            adv = _pad([adv_rows[i] for i in mb], width).to(device)

            logp = _response_logprobs(policy, batch, pad_id)
            pg, clipped = policy_loss(logp, old, adv, cfg.clip_eps)
            kl = kl_k3(logp, ref)
            per_token = pg + cfg.kl_beta * kl if kl_in_loss else pg
            per_traj = masked_mean(per_token, mask, dim=1)
            (per_traj.sum() / len(mini)).backward()

            share = len(mb) / len(trajs)
            sums["policy_loss"] += masked_mean(pg, mask, dim=1).mean().item() * share
            sums["kl"] += masked_mean(kl, mask, dim=1).mean().item() * share
            sums["clip_frac"] += masked_mean(clipped, mask, dim=1).mean().item() * share
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        sums["grad_norm"] += float(grad_norm) / len(minis)
        optimizer.step()
    policy.eval()
    return sums


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def _save_latest(run_dir: str, step: int, policy, optimizer, scheduler,
                 critic=None, critic_opt=None, critic_sched=None) -> None:
    import torch

    tmp, latest, old = f"{run_dir}/latest.tmp", f"{run_dir}/latest", f"{run_dir}/latest.old"
    shutil.rmtree(tmp, ignore_errors=True)
    policy.save_pretrained(f"{tmp}/policy")
    torch.save(optimizer.state_dict(), f"{tmp}/optimizer.pt")
    torch.save(scheduler.state_dict(), f"{tmp}/scheduler.pt")
    if critic is not None:
        critic.backbone.save_pretrained(f"{tmp}/critic")
        torch.save(critic.head.state_dict(), f"{tmp}/value_head.pt")
        torch.save(critic_opt.state_dict(), f"{tmp}/critic_optimizer.pt")
        torch.save(critic_sched.state_dict(), f"{tmp}/critic_scheduler.pt")
    with open(f"{tmp}/state.json", "w") as f:
        json.dump({"step": step}, f)
    if os.path.exists(latest):
        os.replace(latest, old)
    os.replace(tmp, latest)
    shutil.rmtree(old, ignore_errors=True)


def _append_jsonl(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _warmup_scheduler(optimizer, warmup_frac: float, steps: int):
    import torch
    warmup = max(1, round(warmup_frac * steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup))


# --------------------------------------------------------------------------- #
def train(cfg: Config, run_name: str, search_fn: Callable[[list[str]], list[str]],
          commit: Callable[[], None] = lambda: None) -> None:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm.lora.request import LoRARequest

    from algorithms import _self_test, grpo_advantages

    if cfg.algo not in ("grpo", "ppo"):
        raise ValueError(f"unknown algo {cfg.algo!r}")
    if cfg.algo == "ppo" and cfg.samples_per_question != 1:
        raise ValueError("PPO uses one sample per question (samples_per_question=1)")
    is_ppo = cfg.algo == "ppo"

    run_dir = f"{RUNS_DIR}/{run_name}"
    os.makedirs(f"{run_dir}/ckpt", exist_ok=True)
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(config_to_dict(cfg), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # Create vLLM before any PyTorch work in this process: vLLM forks its engine process,
    # and forking after PyTorch has started its worker threads can deadlock the child.
    llm = make_llm(cfg.model_name, cfg, cfg.train_vllm_gpu_mem, enable_lora=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    _self_test()  # fail fast if the RL math is broken in this environment
    torch.manual_seed(cfg.seed)

    base = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to("cuda")
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    latest = f"{run_dir}/latest"
    resuming = os.path.exists(f"{latest}/state.json")
    if resuming:
        policy = PeftModel.from_pretrained(base, f"{latest}/policy", is_trainable=True)
        start_step = json.load(open(f"{latest}/state.json"))["step"]
    else:
        policy = get_peft_model(base, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules), task_type="CAUSAL_LM", bias="none"))
        start_step = 0
    policy.eval()
    policy.print_trainable_parameters()

    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                                  lr=cfg.policy_lr, weight_decay=0.0)
    scheduler = _warmup_scheduler(optimizer, cfg.policy_warmup_frac, cfg.steps)
    critic = critic_opt = critic_sched = None
    if is_ppo:
        critic = _build_critic(cfg, latest if resuming else None)
        critic_opt = torch.optim.AdamW([p for p in critic.parameters() if p.requires_grad],
                                       lr=cfg.critic_lr, weight_decay=0.0)
        critic_sched = _warmup_scheduler(critic_opt, cfg.critic_warmup_frac, cfg.steps)
    if resuming:
        optimizer.load_state_dict(torch.load(f"{latest}/optimizer.pt"))
        scheduler.load_state_dict(torch.load(f"{latest}/scheduler.pt"))
        if is_ppo:
            critic_opt.load_state_dict(torch.load(f"{latest}/critic_optimizer.pt"))
            critic_sched.load_state_dict(torch.load(f"{latest}/critic_scheduler.pt"))
        print(f"resumed {run_name} at step {start_step}")

    questions = load_jsonl(TRAIN_PATH)
    order = list(range(len(questions)))
    random.Random(cfg.seed).shuffle(order)  # fixed order, so a resumed run sees the same data
    q_per_step, group = cfg.questions_per_step, cfg.samples_per_question

    for step in range(start_step, cfg.steps):
        t_start = time.time()
        rng = random.Random(cfg.seed * 100_003 + step)

        # 1-2. Rollouts from the current adapter.
        adapter_dir = f"{LORA_TMP}/step_{step}"
        policy.save_pretrained(adapter_dir)
        shutil.rmtree(f"{LORA_TMP}/step_{step - 1}", ignore_errors=True)
        lora = LoRARequest(f"policy_step_{step}", step + 1, adapter_dir)
        picked = [questions[order[(step * q_per_step + k) % len(order)]] for k in range(q_per_step)]
        search_seconds = [0.0]

        def timed_search(queries: list[str]) -> list[str]:
            t0 = time.time()
            results = search_fn(queries)
            search_seconds[0] += time.time() - t0
            return results

        trajs = generate_search(llm, tokenizer, picked, cfg, timed_search, n_samples=group,
                                temperature=cfg.train_temperature, lora_request=lora,
                                return_logprobs=True)
        t_rollout = time.time()

        # 3. Rewards.
        rewards = torch.tensor([score_trajectory(t) for t in trajs])

        # 4. Old and reference logprobs (+ values for PPO), all before any update.
        size = cfg.micro_batch_size
        old_lps = _for_all(lambda b: _response_logprobs(policy, b, pad_id), trajs, size)
        with policy.disable_adapter():
            ref_lps = _for_all(lambda b: _response_logprobs(policy, b, pad_id), trajs, size)
        values = _for_all(lambda b: _response_values(critic, b, pad_id), trajs, size) if is_ppo else None
        t_logprobs = time.time()

        # 5-6. Advantages and updates.
        extra: dict = {}
        if is_ppo:
            adv_rows, returns, extra = _ppo_advantages(trajs, rewards, old_lps, ref_lps, values, cfg)
            extra.update(_critic_update(critic, critic_opt, trajs, values, returns, cfg, pad_id, rng))
            extra["critic_lr"] = critic_sched.get_last_lr()[0]
            critic_sched.step()
        else:
            advantages = grpo_advantages(rewards, group).tolist()
            adv_rows = [torch.full((len(t.response_ids),), a) for t, a in zip(trajs, advantages)]
        stats = _policy_update(policy, optimizer, trajs, adv_rows, old_lps, ref_lps, cfg, pad_id,
                               rng, kl_in_loss=not is_ppo)
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        t_update = time.time()

        # 7. Metrics.
        gen_tokens = [sum(t.loss_mask) for t in trajs]
        diffs = [abs(o - v) for t, lp in zip(trajs, old_lps)
                 for o, v, m in zip(lp.tolist(), t.rollout_logprobs, t.loss_mask)
                 if m and not math.isnan(v)]
        grouped = rewards.view(-1, group)
        row = {
            "step": step + 1,
            "reward_mean": rewards.mean().item(),
            "answer_rate": sum(t.finish == "answer" for t in trajs) / len(trajs),
            "searches_mean": sum(t.n_searches for t in trajs) / len(trajs),
            "turns_mean": sum(t.n_turns for t in trajs) / len(trajs),
            "gen_tokens_mean": sum(gen_tokens) / len(trajs),
            "seq_len_max": max(len(t.prompt_ids) + len(t.response_ids) for t in trajs),
            "zero_std_frac": (grouped.std(dim=1) == 0).float().mean().item() if group > 1 else None,
            **stats,
            "lr": lr,
            **extra,
            "logprob_diff_vs_vllm": sum(diffs) / len(diffs) if diffs else None,
            "time_rollout": t_rollout - t_start,
            "time_search": search_seconds[0],   # part of time_rollout spent waiting on the retriever
            "time_logprobs": t_logprobs - t_rollout,
            "time_update": t_update - t_logprobs,
            "time_step": time.time() - t_start,
        }
        _append_jsonl(f"{run_dir}/metrics.jsonl", row)
        print(json.dumps({k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in row.items()}))
        if step % cfg.sample_log_every == 0:
            for t in trajs[:2]:
                _append_jsonl(f"{run_dir}/samples.jsonl", {
                    "step": step + 1, "question": t.question, "golden_answers": t.golden_answers,
                    "prediction": t.prediction, "reward": t.reward, "response": t.response_text})

        # Checkpoints.
        done = step + 1
        if done % cfg.ckpt_every == 0 or done == cfg.steps:
            policy.save_pretrained(f"{run_dir}/ckpt/step_{done:04d}/policy")
            _save_latest(run_dir, done, policy, optimizer, scheduler,
                         critic, critic_opt, critic_sched)
        commit()

    print(f"finished {run_name}: {cfg.steps} steps")