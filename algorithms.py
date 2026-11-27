
from __future__ import annotations

import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp(min=1)
    return (x * mask).sum(dim) / mask.sum(dim).clamp(min=1)


def masked_whiten(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standardize over masked entries (PPO advantage normalization, as in verl)."""
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask)
    return (x - mean) * torch.rsqrt(var + eps) * mask


def explained_variance(values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor) -> float:
    """1 - Var(returns - values) / Var(returns) over masked entries; 1.0 = perfect critic."""
    def var(t):
        return masked_mean((t - masked_mean(t, mask)) ** 2, mask)
    return float(1.0 - var(returns - values) / var(returns).clamp(min=1e-8))


def token_logprobs(logits: torch.Tensor, targets: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """log p(target) per position, computed in float32 chunks to bound memory.

    logits: [N, L, V] (any float dtype), targets: [N, L] -> [N, L]
    """
    out = []
    for start in range(0, logits.size(1), chunk):
        block = logits[:, start:start + chunk].float()
        picked = block.gather(-1, targets[:, start:start + chunk, None]).squeeze(-1)
        out.append(picked - torch.logsumexp(block, dim=-1))
    return torch.cat(out, dim=1)


# --------------------------------------------------------------------------- #
# GRPO
# --------------------------------------------------------------------------- #
def grpo_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    """(r - group mean) / (group std + eps); trajectories must be grouped by question."""
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True) if group_size > 1 else torch.zeros_like(mean)
    return ((grouped - mean) / (std + eps)).view(-1)


# --------------------------------------------------------------------------- #
# Shared by GRPO and PPO
# --------------------------------------------------------------------------- #
def policy_loss(logp: torch.Tensor, logp_old: torch.Tensor, advantages: torch.Tensor,
                clip_eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token clipped surrogate loss (to minimize) and a 0/1 'was clipped' tensor.

    advantages: [N, T] (PPO) or [N, 1] (GRPO, broadcast over tokens).
    """
    ratio = torch.exp(logp - logp_old)
    loss_unclipped = -advantages * ratio
    loss_clipped = -advantages * ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    return torch.maximum(loss_unclipped, loss_clipped), (loss_clipped > loss_unclipped).float()


def kl_k3(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Per-token KL(pi || ref) estimator 'k3', clamped as in verl's low_var_kl."""
    log_ratio = (logp_ref - logp).clamp(-20.0, 20.0)
    return (torch.exp(log_ratio) - log_ratio - 1.0).clamp(-10.0, 10.0)


# --------------------------------------------------------------------------- #
# PPO
# --------------------------------------------------------------------------- #
def ppo_token_rewards(scores: torch.Tensor, logp_old: torch.Tensor, logp_ref: torch.Tensor,
                      mask: torch.Tensor, beta: float) -> torch.Tensor:
    """KL-in-reward: -beta * (log pi_old - log pi_ref) on generated tokens, plus the EM
    score on each trajectory's last generated token."""
    rewards = -beta * (logp_old - logp_ref) * mask
    positions = torch.arange(mask.size(1), device=mask.device).expand_as(mask)
    last = torch.where(mask > 0, positions, torch.full_like(positions, -1)).max(dim=1).values
    rows = torch.arange(mask.size(0), device=mask.device)
    has_tokens = last >= 0
    rewards[rows[has_tokens], last[has_tokens]] += scores[has_tokens]
    return rewards


def gae(token_rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor,
        gamma: float = 1.0, lam: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation over generated tokens only.

    Masked positions (retrieved text, padding) are skipped: they neither get an
    advantage nor pass value along, so the recursion runs over the sequence of
    generated tokens. With gamma = lam = 1 this reduces to
    advantage_t = (sum of rewards from t on) - V_t.
    """
    next_value = torch.zeros_like(values[:, 0])
    next_adv = torch.zeros_like(values[:, 0])
    advantages = torch.zeros_like(values)
    for t in reversed(range(values.size(1))):
        live = mask[:, t] > 0
        delta = token_rewards[:, t] + gamma * next_value - values[:, t]
        adv_t = delta + gamma * lam * next_adv
        advantages[:, t] = torch.where(live, adv_t, torch.zeros_like(adv_t))
        next_value = torch.where(live, values[:, t], next_value)
        next_adv = torch.where(live, adv_t, next_adv)
    returns = (advantages + values) * mask
    return advantages, returns


def value_loss(values: torch.Tensor, values_old: torch.Tensor, returns: torch.Tensor,
               clip_eps: float) -> torch.Tensor:
    """Per-token clipped value loss (verl/PPO style)."""
    clipped = values_old + (values - values_old).clamp(-clip_eps, clip_eps)
    return 0.5 * torch.maximum((values - returns) ** 2, (clipped - returns) ** 2)


# --------------------------------------------------------------------------- #
def _self_test() -> None:
    close = lambda a, b: torch.allclose(a, torch.as_tensor(b, dtype=a.dtype), atol=1e-5)

    # masked mean ignores masked entries
    x, m = torch.tensor([[1., 2., 100.]]), torch.tensor([[1., 1., 0.]])
    assert close(masked_mean(x, m), 1.5) and close(masked_mean(x, m, dim=1), [1.5])

    # token_logprobs == log_softmax gather, across chunk boundaries
    logits, tgt = torch.randn(2, 5, 7), torch.randint(0, 7, (2, 5))
    ref = torch.log_softmax(logits, -1).gather(-1, tgt[..., None]).squeeze(-1)
    assert torch.allclose(token_logprobs(logits, tgt, chunk=2), ref, atol=1e-5)

    # GRPO: group of identical rewards -> zero advantage; mixed group -> standardized
    adv = grpo_advantages(torch.tensor([1., 1., 1., 0., 1., 0.]), group_size=3)
    assert close(adv[:3], [0., 0., 0.])
    assert adv[3] < 0 and adv[4] > 0 and close(adv[3], adv[5])

    # policy loss: ratio 1 -> loss = -A, nothing clipped; big ratio with A>0 -> clipped
    lp = torch.zeros(1, 2)
    loss, clipped = policy_loss(lp, lp, torch.tensor([[2.0]]), 0.2)
    assert close(loss, [[-2., -2.]]) and clipped.sum() == 0
    loss, clipped = policy_loss(torch.tensor([[1.0]]), torch.tensor([[0.0]]), torch.tensor([[1.0]]), 0.2)
    assert close(loss, [[-1.2]]) and clipped.sum() == 1

    # KL is zero when policy == reference, positive otherwise
    assert close(kl_k3(lp, lp), [[0., 0.]]) and kl_k3(torch.tensor([0.5]), torch.tensor([0.0])) > 0

    # PPO rewards: score lands on the last generated token, KL on all generated tokens
    mask = torch.tensor([[1., 1., 0., 0., 1., 0.]])           # gen, gen, info, info, gen, pad
    r = ppo_token_rewards(torch.tensor([1.0]), torch.zeros(1, 6), torch.zeros(1, 6), mask, beta=0.1)
    assert close(r, [[0., 0., 0., 0., 1., 0.]])

    # GAE with gamma = lam = 1: advantage = reward-to-go - V on generated tokens, 0 elsewhere
    values = torch.tensor([[0.2, 0.3, 9.0, 9.0, 0.6, 9.0]])
    rewards = torch.tensor([[0.1, 0.0, 5.0, 5.0, 1.0, 5.0]])  # rewards on masked positions are ignored
    rewards = rewards * mask
    adv, ret = gae(rewards, values, mask)
    to_go = torch.tensor([[1.1, 1.0, 0., 0., 1.0, 0.]])
    assert close(adv, (to_go - values) * mask), adv
    assert close(ret, to_go * mask)

    # value loss is zero when values already equal returns
    assert close(value_loss(ret, ret, ret, 0.2), torch.zeros(1, 6))

    # whitening: masked entries -> mean 0, std 1; unmasked stay 0
    w = masked_whiten(torch.tensor([[1., 2., 3., 99.]]), torch.tensor([[1., 1., 1., 0.]]))
    assert close(w[0, 3], 0.0) and abs(float(w[0, :3].mean())) < 1e-5
    assert abs(float((w[0, :3] ** 2).mean()) - 1.0) < 1e-3

    # explained variance: perfect critic -> 1, constant-zero critic on varied returns -> <= 0
    r_ = torch.tensor([[1., 2., 3.]]); m_ = torch.ones(1, 3)
    assert abs(explained_variance(r_, r_, m_) - 1.0) < 1e-6
    assert explained_variance(torch.zeros(1, 3), r_, m_) <= 0.0
    print("algorithms.py self-test passed")


if __name__ == "__main__":
    _self_test()