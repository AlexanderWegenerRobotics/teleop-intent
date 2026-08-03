# Results

Held-out test split: 22 (episode, arm) pairs.

| model | test macro F1 | SE |
|---|---|---|
| HMM | 0.768 | 0.038 |
| GRU | 0.832 | 0.035 |
| Transformer | 0.833 | 0.033 |

| comparison | difference | 95% interval | verdict |
|---|---|---|---|
| GRU - HMM | +0.064 | [+0.019, +0.109] | distinguishable |
| Transformer - HMM | +0.065 | [+0.030, +0.100] | distinguishable |
| Transformer - GRU | +0.001 | [-0.017, +0.018] | tie |

| ablation | validation macro F1 |
|---|---|
| shuffled features (chance floor) | 0.244 |
| no temporal context (features only) | 0.786 |
| GRU (features + memory) | 0.862 |
| Transformer (features + attention) | 0.846 |
