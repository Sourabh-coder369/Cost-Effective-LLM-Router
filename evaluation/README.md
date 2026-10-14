# Evaluation workspace

- `domain_labeling/` contains regex/Qwen prompt labels, Kaggle notebook assets,
  and label comparisons.
- `domain_discovery/` contains embedding-based clustering experiments.
- `results/` contains checked-in plots and benchmark summaries.
- The remaining scripts at this level cover router benchmarks, model-layer
  experiments, and analysis. Their data products belong under `artifacts/`.

Local model checkpoints, caches, and generated reports are kept outside this
source folder under `artifacts/`, which is ignored by Git.
