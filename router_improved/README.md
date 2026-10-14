# Router improvement experiments

This directory is an isolated research area. It never changes the baseline or
hard-domain implementations in `router/`.

## Experiment catalog

| Script | Purpose | Output directory |
| --- | --- | --- |
| `train.py` | Original unconstrained soft-expert router. | `artifacts/gpt_llama_soft_domain_router/` |
| `train_soft_math_mf.py` | Softly blends hard general/math MF scores using `P(math | embedding)`. | `artifacts/soft_math_mf/` |
| `train_soft_math_length_mf.py` | Linear prompt-length feature experiment. | `artifacts/soft_math_length_mf/` |
| `train_soft_math_length_bucket_mf.py` | Nonlinear prompt-length bucket residual. | `artifacts/soft_math_length_bucket_mf/` |
| `train_residual_mlp_router.py` | Direct residual MLP using MPNet, math probability, and length features. | `artifacts/residual_mlp_router/` |
| `train_model_heads.py` | BGE-large shared projection with separate small GPT-4 and Llama score heads. | `artifacts/model_heads_bge_256/` |
| `sweep_model_head_sizes.py` | Compare widths of the GPT-4/Llama heads over three seeds, selecting by validation APGR. | `artifacts/model_head_size_sweep/` |
| `select_soft_length_bucket_ensemble.py` | Selects soft/bucket ensemble weights by validation APGR. | `artifacts/soft_length_bucket_ensemble/` |

Support scripts are named by function:

- `analyze_domain_embeddings.py`: semantic/domain separability analysis.
- `generate_mpnet_test_embeddings.py`: creates a missing MPNet cache on CPU.
- `evaluate_*` and `reproduce_*`: reproducible cost/quality and slice reports.
- `train_clustered_domain_mf.py` and `train_logistic_router.py`: comparison baselines.

## Reproducibility rules

1. Use the same embedding model and dimension for train, validation, and test.
2. Fit feature transforms, math detectors, and model parameters on training
   data only.
3. Select epochs, architecture, and ensemble weights using validation APGR.
4. Treat test results as a final report, not an architecture-selection signal.

## Soft Math MF flow

```text
prompt
  -> MPNet embedding
  -> logistic math detector: g = P(math | embedding)
  -> general MF logit and math MF logit
  -> (1 - g) * general_logit + g * math_logit
  -> P(GPT-4 wins) and threshold-based routing
```

## Standard command pattern

Every training script accepts a Parquet path and an embedding-cache path for
each split:

```powershell
.\.venv\Scripts\python.exe router_improved\<training-script>.py `
  --train-data <train.parquet> --train-embeddings <train.pt> `
  --val-data <val.parquet> --val-embeddings <val.pt> `
  --test-data <test.parquet> --test-embeddings <test.pt> `
  --output-dir artifacts\<experiment-name>
```

For the current GPT-4/Llama data, use 768-dimensional
`all-mpnet-base-v2` embeddings. Do not mix them with the 384- or
1024-dimensional caches.

The model-head ablation is an exception: it uses the three matching
1024-dimensional BGE-large caches under `data/original_gpt4_llama7b/bge_embeddings/`.
Run `python router_improved/train_model_heads.py` from the repository root.
Its two scores are relative latent scores trained from winner labels; use their
difference for routing, not either score as a calibrated quality estimate.
Run `python router_improved/sweep_model_head_sizes.py` to compare per-model
head widths of 8, 16, 32, 64, and 128 with seeds 41, 42, and 43. The sweep
selects the width by mean validation APGR and evaluates only its representative
median-seed checkpoint on test.

## Reading results

Each experiment writes:

- `best_model.pt`: selected checkpoint;
- `results.json`: configuration, training history, and held-out metrics;
- optional `training.log` files when launched in the background.

APGR is the primary model-selection metric. Higher means the router recovers
more of GPT-4's quality advantage across a range of GPT-4 call budgets.
