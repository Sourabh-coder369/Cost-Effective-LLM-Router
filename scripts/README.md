# Project scripts

- `data_upload/` contains scripts for publishing local datasets to Hugging Face.
- `experiments/` contains launchers for the current BGE DomainMF experiments.
- `legacy/` contains launchers that refer to an older project layout.

Run the 256-dimensional DomainMF launcher from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_domain_router_256.py
```

To retrain both the baseline and regex DomainMF at the default 256-D model
size, then compare them on the test split, run:

```powershell
.\scripts\experiments\retrain_regex_256.ps1
```

This saves a fresh run under `artifacts/regex_bge256_retrained/` and preserves
the existing checkpoints. The baseline and explicit-split DomainMF training
commands now default to 256-D; pass `--model_embedding_dim 64` or
`--model-embedding-dim 64` to request 64-D explicitly.

The upload scripts contact Hugging Face and upload files when run. They are
helpers only; this documentation update does not execute them.
