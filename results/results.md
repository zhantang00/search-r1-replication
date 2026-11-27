## Main results (Qwen2.5-3B-Instruct, Bamboogle, n = 125)

| Method | Bamboogle EM | 95% CI | Searches / question | Answer rate | Hit turn limit | Paper |
|---|---|---|---|---|---|---|
| Direct Inference (no retrieval) | **0.024** (3/125) | 0.000–0.056 | 0.00 | 1.00 | 0 | 0.024 |
| Search-R1 template, untrained (step 0) | **0.152** (19/125) | 0.088–0.216 | 0.88 | 0.99 | 1 | — |
| **Search-R1-instruct, GRPO (LoRA, step 100)** | **0.264** (33/125) | 0.192–0.344 | 1.82 | 0.99 | 1 | 0.232 (Table 3) |
| **Search-R1-instruct, PPO (LoRA, step 100)** | **0.256** (32/125) | 0.184–0.328 | 1.89 | 0.95 | 6 | 0.264 (Table 2) |

## Reference: authors' released checkpoints, same evaluation code

| Method | Bamboogle EM | 95% CI | Searches / question | Answer rate | Hit turn limit | Paper |
|---|---|---|---|---|---|---|
| Official GRPO checkpoint (full fine-tune) | **0.320** (40/125) | 0.240–0.400 | 1.90 | 1.00 | 0 | — |
| Official PPO v0.3 checkpoint (full fine-tune) | **0.272** (34/125) | 0.200–0.352 | 2.22 | 0.90 | 12 | — |

## Extended training (all evaluated checkpoints; analysis, not selection)

| Method | Bamboogle EM | 95% CI | Searches / question | Answer rate | Hit turn limit | Paper |
|---|---|---|---|---|---|---|
| GRPO step 200 | **0.296** (37/125) | 0.216–0.376 | 2.30 | 0.94 | 8 |  |
| GRPO step 280 (paper's rule) | **0.288** (36/125) | 0.208–0.368 | 2.32 | 0.94 | 8 |  |
| GRPO step 290 | **0.232** (29/125) | 0.160–0.304 | 3.59 | 0.59 | 51 |  |
| PPO step 200 | **0.128** (16/125) | 0.072–0.192 | 1.04 | 1.00 | 0 |  |
| PPO step 300 (paper's rule) | **0.104** (13/125) | 0.056–0.160 | 1.00 | 1.00 | 0 |  |

## Paired significance tests (exact McNemar on the same 125 questions)

| Comparison | EM | Only first right | Only second right | p | At 0.05 |
|---|---|---|---|---|---|
| grpo vs step0 | 0.264 vs 0.152 | 19 | 5 | 0.007 | significant |
| ppo vs step0 | 0.256 vs 0.152 | 18 | 5 | 0.011 | significant |
| grpo vs ppo | 0.264 vs 0.256 | 10 | 9 | 1.000 | not significant |
| official_grpo vs grpo | 0.320 vs 0.264 | 15 | 8 | 0.210 | not significant |
| official_ppo_v03 vs ppo | 0.272 vs 0.256 | 11 | 9 | 0.824 | not significant |
| grpo_step200 vs grpo | 0.296 vs 0.264 | 14 | 10 | 0.541 | not significant |
| grpo_step200 vs step0 | 0.296 vs 0.152 | 23 | 5 | 0.001 | significant |
| grpo_step280 vs grpo | 0.288 vs 0.264 | 12 | 9 | 0.664 | not significant |
| grpo_step280 vs step0 | 0.288 vs 0.152 | 22 | 5 | 0.002 | significant |
| grpo_step290 vs grpo | 0.232 vs 0.264 | 10 | 14 | 0.541 | not significant |
| grpo_step290 vs step0 | 0.232 vs 0.152 | 18 | 8 | 0.076 | not significant |
| ppo_step200 vs ppo | 0.128 vs 0.256 | 6 | 22 | 0.004 | significant |
| ppo_step200 vs step0 | 0.128 vs 0.152 | 12 | 15 | 0.701 | not significant |
| ppo_step300 vs ppo | 0.104 vs 0.256 | 5 | 24 | 0.001 | significant |
| ppo_step300 vs step0 | 0.104 vs 0.152 | 9 | 15 | 0.307 | not significant |
