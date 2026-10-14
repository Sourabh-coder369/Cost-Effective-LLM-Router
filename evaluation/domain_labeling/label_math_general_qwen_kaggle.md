# Qwen3-1.7B math/general labeling on Kaggle T4 x2

This is an offline annotation script for your router datasets. It uses
`Qwen/Qwen3-1.7B` with thinking disabled, one FP16 model replica per T4,
and concurrent batches of eight rows per GPU. It does not solve the questions
or use regex labels, answers, preference labels, or dataset names as model input.
The annotation rubric includes mathematical explanations as math and treats
software implementation as general unless solving mathematics is the primary task.

## Notebook setup

1. In Kaggle notebook settings, select **GPU T4 x2** and enable **Internet**.
2. Attach your dataset containing `router_train.parquet`, `router_val.parquet`,
   and `router_test.parquet`. CSV and JSONL inputs are also supported.
3. Upload `label_math_general_qwen_kaggle.py` as a Kaggle dataset or copy its
   contents into a notebook cell beginning with
   `%%writefile /kaggle/working/label_math_general_qwen_kaggle.py`.
4. Install the following packages in a notebook cell. Keep Kaggle's existing
   CUDA-enabled PyTorch installation:

```python
%pip install -q "transformers>=4.51.0,<5" "accelerate>=0.30" pandas pyarrow huggingface_hub tqdm
```

If the script is attached as a dataset, copy it into the working directory:

```python
from pathlib import Path
import shutil

script = next(Path('/kaggle/input').rglob('label_math_general_qwen_kaggle.py'))
shutil.copy2(script, '/kaggle/working/label_math_general_qwen_kaggle.py')
```

Find your input paths:

```python
from pathlib import Path
print(*Path('/kaggle/input').rglob('router_*.parquet'), sep='\n')
```

Run a small smoke test, replacing the input path with the path printed above:

```python
!python /kaggle/working/label_math_general_qwen_kaggle.py \
    --input /kaggle/input/YOUR_DATASET/router_train.parquet \
    --text-column prompt \
    --limit 100 \
    --output-dir /kaggle/working/qwen_label_smoke
```

Inspect the labels before the full run:

```python
import pandas as pd
sample = pd.read_parquet('/kaggle/working/qwen_label_smoke/00_router_train/labeled.parquet')
display(sample[['prompt', 'qwen_domain', 'qwen_label_status', 'qwen_raw_label']].head(30))
print(sample['qwen_domain'].value_counts(dropna=False))
```

Then run all splits in a separate output directory:

```python
!python /kaggle/working/label_math_general_qwen_kaggle.py \
    --input /kaggle/input/YOUR_DATASET/router_train.parquet \
            /kaggle/input/YOUR_DATASET/router_val.parquet \
            /kaggle/input/YOUR_DATASET/router_test.parquet \
    --text-column prompt \
    --batch-size-per-gpu 8 \
    --output-dir /kaggle/working/qwen_domain_labels
```

Use `--text-column question` for a dataset with a `question` column. Each
input is exported separately, in its original row order, so cached embeddings
remain aligned. Existing input columns, including regex and routing labels,
are preserved. Output columns cannot already exist in the input.

## Output

Each numbered input directory contains:

- `labeled.parquet`: original columns plus the columns below.
- `needs_review.csv`: rows without an accepted binary label.
- `batch_*.json`: atomic progress checkpoints, written after each GPU batch.
- `summary.json`: label/status counts.

| Column | Meaning |
| --- | --- |
| `qwen_domain` | `math` or `general`; null for review rows |
| `qwen_domain_id` | Math = 1, general = 0; nullable integer |
| `qwen_label_status` | `ok`, `uncertain`, `invalid_output`, `empty_prompt`, or `too_long` |
| `qwen_raw_label` | Exact generated response for audit |
| `qwen_input_tokens` | Full annotation input token count, including instructions |
| `qwen_model`, `qwen_revision` | Model identifier and resolved model commit |

The root `manifest.json` records the rubric, input file hashes, settings, and
model commit. Labels are accepted only when the entire generated response is
`math` or `general` (ignoring case/outer whitespace). Other outputs are retained
for review. Inputs longer than `--max-input-tokens` (default 4096, including the
rubric) are **not truncated**; they go to review. Raise that limit if necessary,
keeping input plus output within the model's 32,768-token context.

The script uses greedy decoding for reproducible short labels, with thinking
disabled. This is a task-specific choice; Qwen's general non-thinking generation
recommendation uses sampling. Validate on a human-reviewed sample before relying
on the annotations. Model reference:
<https://huggingface.co/Qwen/Qwen3-1.7B>.

## Resume

Rerun the same full-run command with `--resume`. It skips completed rows and
uses the saved model commit. Input file contents, ordering, selected text column,
limits, rubric and generation settings must match; batch size may change.
An interrupted uncommitted batch is regenerated.

For a new Kaggle session, persist/download the output checkpoints first, attach
them as a Kaggle dataset, and copy the **entire output directory**, including
`manifest.json` and numbered directories, into `/kaggle/working/qwen_domain_labels`.
Then run the original command with `--resume`. Kaggle working files do not
automatically survive a discarded session. Do not run two script instances
against the same output directory.

If a batch runs out of GPU memory, the script retries smaller sub-batches. A
single-row memory failure stops the run with its row number; progress already
written remains available. Lower batch size or restart with a new output directory
and a smaller token limit if needed.

## Using the labels in this project

The current `DomainRouterDataset` still computes domain IDs from regex. This
script creates the replacement annotation files; it does not change router training.
When integrating them, read `qwen_domain_id` after resolving review rows. Do not
replace the pairwise model preference column (`binary_label`) with domain labels.
Do not drop review rows from the dataset unless you also filter embeddings by
exactly the same row indices.
