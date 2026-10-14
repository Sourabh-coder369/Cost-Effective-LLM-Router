# Cost-Effective LLM Router

Train and evaluate routers that decide whether a prompt should be sent to a
strong, more expensive model (GPT-4) or a weaker, lower-cost model
(Llama-2-7B). The project uses pairwise preference labels: `1` means the
strong model won, and `0` means the weak model won.

## Project map

```text
router/                 Baseline Matrix Factorization and hard-domain routers
router_improved/        Isolated routing experiments; does not modify router/
evaluation/             RouteLLM-style evaluation and Qwen benchmark studies
evaluation/results/     Checked-in evaluation plots and summary reports
interfaces/             Simple Python and CLI inference interfaces
data/                   Local datasets and embedding caches (git-ignored)
artifacts/              Local checkpoints, metrics, and plots (git-ignored)
docs/                   Reference papers and presentation material
scripts/                Data upload and experiment launch helpers
Data_creation/          Dataset creation notebooks and generated examples
```

## Current GPT-4 / Llama experiment

The reproducible dataset is expected at:

```text
data/original_gpt4_llama7b/gpt4_llama7b_data_unbalanced/
  router_train.parquet
  router_val.parquet
  router_test.parquet
  .embedding_cache/
```

Use matching embedding tensors for all three splits. The current experiments
use `sentence-transformers/all-mpnet-base-v2` embeddings with dimension 768.

### Main routing variants

| Variant | Entry point | Notes |
| --- | --- | --- |
| Baseline MF | `router/baseline/train_router.py` | One capability representation per model. |
| Hard Domain-MF | `router/domain_specific/train_domain_router.py` | Separate general and math capability vectors. |
| Soft Math MF | `router_improved/train_soft_math_mf.py` | Blends general/math scores using `P(math | embedding)`. |
| Length-bucket Soft Math MF | `router_improved/train_soft_math_length_bucket_mf.py` | Adds a learned nonlinear prompt-length residual. |
| Residual MLP | `router_improved/train_residual_mlp_router.py` | Direct prompt router with semantic and structured features. |

The validation-selected soft/bucket ensemble is created with
`router_improved/select_soft_length_bucket_ensemble.py`.

## Quick start

Create the local environment and install the baseline dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r router\requirements.txt
```

Run a soft Math MF experiment with cached embeddings:

```powershell
.\.venv\Scripts\python.exe router_improved\train_soft_math_mf.py `
  --train-data data\original_gpt4_llama7b\gpt4_llama7b_data_unbalanced\router_train.parquet `
  --train-embeddings data\original_gpt4_llama7b\gpt4_llama7b_data_unbalanced\.embedding_cache\router_train.parquet_6b59c6b9cae834b6.pt `
  --val-data data\original_gpt4_llama7b\gpt4_llama7b_data_unbalanced\router_val.parquet `
  --val-embeddings data\original_gpt4_llama7b\gpt4_llama7b_data_unbalanced\.embedding_cache\router_val.parquet_884f7401a8bb0b9d.pt `
  --test-data data\original_gpt4_llama7b\gpt4_llama7b_data_unbalanced\router_test.parquet `
  --test-embeddings data\original_gpt4_llama7b\gpt4_llama7b_data_unbalanced\.embedding_cache\router_test.parquet_mpnet_base_v2.pt `
  --output-dir artifacts\soft_math_mf
```

See [router_improved/README.md](router_improved/README.md) for the experiment
catalog, evaluation commands, and metric definitions.

See [scripts/README.md](scripts/README.md) for helper scripts and
[evaluation/domain_labeling/README.md](evaluation/domain_labeling/README.md) for
the Qwen and regex labeling tools.

See [docs/README_ROUTER.md](docs/README_ROUTER.md) for the router research notes
and [docs/project/project_summary_slide.md](docs/project/project_summary_slide.md)
for the project presentation outline. Qwen math/general labeling utilities are
grouped under `evaluation/domain_labeling`; generated evaluation outputs are
kept in `evaluation/results/` when versioned and under `artifacts/` when local.

## Metrics

- **ROC-AUC / PR-AUC**: quality of the strong-model-win ranking.
- **APGR**: average performance gap recovered across 10% through 100% strong
  model call budgets. Higher is better.
- **CPT(50%) / CPT(80%)**: fraction of strong-model calls needed to recover
  50% or 80% of the strong-vs-weak quality gap. Lower is better.
- **Cost savings**: approximately the percentage of calls sent to Llama rather
  than GPT-4, when weak-model cost is negligible relative to GPT-4.

## Repository conventions

- Keep reusable source code in `router/`, `router_improved/`, `evaluation/`,
  or `interfaces/`.
- Keep generated checkpoints, predictions, plots, logs, and reports in
  `artifacts/<experiment-name>/`.
- Keep raw data and embedding caches under `data/`; they are intentionally not
  committed.
- Select models and ensemble weights on validation data. Reserve an untouched
  final test split for the final claim.
