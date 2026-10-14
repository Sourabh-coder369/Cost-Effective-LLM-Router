# Baseline router workflow

This file documents the stable Matrix Factorization baseline in `router/`.
For the full repository layout and current GPT-4/Llama experiment, start with
the root [README](../README.md).

## Components

```text
router/baseline/
  dataset.py          Loads preference labels and cached text embeddings.
  model.py            MatrixFactorizationRouter implementation.
  train_router.py     Baseline training entry point.

router/domain_specific/
  domain_dataset.py   General/math detector and domain-aware data loader.
  domain_model.py     Hard Domain-MF implementation.
  train_domain_router.py
```

## Input data

Each Parquet split must include:

| Column | Meaning |
| --- | --- |
| `prompt` | User prompt to route. |
| `strong_model` / `weak_model` | Names of the model pair. |
| `binary_label` | `1` if the strong model wins, `0` if the weak model wins. |

Use one embedding model and dimension consistently across train, validation, and
test. Cached embeddings are accepted by both baseline and domain loaders.

## Training

```powershell
.\.venv\Scripts\python.exe router\baseline\train_router.py `
  --data_path <train.parquet> `
  --val_data_path <val.parquet> `
  --checkpoint_dir artifacts\baseline_mf
```

`--val_data_path` is optional. If omitted, the training entry point makes an
internal validation split; supplying an explicit validation split is preferred
for reproducible comparisons.

## Evaluation

Use RouteLLM-style metrics for a saved baseline checkpoint:

```powershell
.\.venv\Scripts\python.exe evaluation\routellm_evaluation.py `
  --checkpoint artifacts\baseline_mf\best_model.pt `
  --test_data <test.parquet> `
  --test_embeddings <test_embeddings.pt> `
  --output artifacts\baseline_mf\test_results.json
```

The evaluation reports APGR, CPT, and performance at fixed strong-model call
budgets. See the root README for their definitions.
