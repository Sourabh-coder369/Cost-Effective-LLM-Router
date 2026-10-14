# Generate Laya features for `original_gpt4_llama7b`

The Kaggle pipeline reads the local dataset in `data/original_gpt4_llama7b/gpt4_llama7b_data_unbalanced/`. It supports `router_train.parquet`, `router_val.parquet`, and `router_test.parquet` (or the matching CSV files). The input column is `prompt`; model responses and preference labels are not sent to Laya.

The output has one row per source question, preserving its split and row index. Exact duplicate prompts are run through Laya once and their 40-value vector is reused. Prompts exceeding the packed 1,024-token limit are included with `status="discarded_over_token_limit"` and `features=null`.

## Run the Kaggle notebook

1. Create a Kaggle Dataset with the three Parquet files from the folder above. The CSV splits work too, but Parquet is preferred.
2. Upload [laya_features_kaggle.ipynb](laya_features_kaggle.ipynb) as a Kaggle notebook and attach that dataset.
3. Select **GPU T4 x2**, enable Internet, and run all cells. The first cells install Laya and PyArrow, locate the split files, and run a 16-prompt-per-GPU smoke check. The full extraction follows if the smoke cell succeeds.
4. Save a notebook version after extraction completes and download `/kaggle/working/laya_features/laya_features.zip` from the notebook output files.

The notebook embeds the full extraction code. It writes a worker module to Kaggle's working directory at runtime because Python's spawned GPU workers need importable functions. The only attached data needed is the three dataset split files.

## Output contents

- `rlaif_features.jsonl`: one record per original row, with `split`, `row_index`, `query_hash`, `prompt`, `status`, and a 40-number `features` list when retained.
- `extract_metadata.json`: split counts, unique-prompt count, discard count, feature names, source-file hash, and token limit.
- `laya_features_gpu*.jsonl`: resumable feature shards, one row per unique prompt.
- `discarded_queries.jsonl`: unique prompts excluded by the 1,024-token packed-input limit.

The 1,024-token cap includes Laya's probe instructions and options. The question text itself therefore has less than 1,024 tokens available. Reduce `BATCH_SIZE` in the notebook setup cell (for example, 16 to 4) if Kaggle GPU memory is tight; this does not change which prompts fit.

Feature extraction and router training are separate steps. After downloading `laya_features.zip` into `data/laya_features/`, train the pairwise MF router locally:

```powershell
.\.venv\Scripts\python.exe kaggle_laya_mf\laya_rlaif_mf_local.py
```

This trainer uses `binary_label` to learn a preference router for the two models in this dataset: GPT-4 and Llama-2-7B-Chat. It selects the best epoch on the validation split and reports held-out test metrics. The checkpoint, training history, and metrics are saved under `kaggle_laya_mf/work/rlaif_mf/`. It does not use model response text as input.

## Run the standalone script instead

If you prefer the `.py` entry point, upload `laya_features_kaggle.py` and the split files to a Kaggle Dataset, then run:

```python
!pip install -q "laya==0.3.23" pyarrow
!python /kaggle/input/YOUR-DATASET-SLUG/laya_features_kaggle.py \
  --rlaif-data-dir /kaggle/input/YOUR-DATASET-SLUG \
  --gpu-ids 0,1 --laya-batch-size 16 --max-input-tokens 1024 --smoke-test
```

After the smoke check succeeds, run the same command without `--smoke-test` to process the full dataset.
