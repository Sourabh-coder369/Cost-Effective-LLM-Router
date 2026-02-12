# Router Training Workflow

This directory contains unified scripts for training LLM routers on different datasets and model pairs.

## Quick Start

### 1. Prepare Data

Convert your preference dataset to router format:

```bash
# From RLAIF-style dataset (with ranked answers)
python prepare_router_data.py \
    --input rlaif.parquet \
    --output router_data.parquet \
    --format rlaif \
    --strong_model "gpt-4" \
    --weak_model "llama-2-7b-chat" \
    --sample_size 20000

# From already-formatted dataset
python prepare_router_data.py \
    --input strong_weak_dataset_t025.parquet \
    --output router_data.parquet \
    --format standard
```

### 2. Create Train/Val/Test Splits

```bash
python make_splits.py \
    --input router_data.parquet \
    --output_dir data \
    --train_frac 0.8 \
    --val_frac 0.1
```

This creates:
- `data/router_train.parquet`
- `data/router_val.parquet`
- `data/router_test.parquet`

### 3. Train Router

```bash
python train_router.py \
    --data_path data/router_train.parquet \
    --checkpoint_dir checkpoints \
    --batch_size 128 \
    --num_epochs 50 \
    --learning_rate 0.001
```

### 4. Evaluate Router

```bash
python generate_perf_cost_table.py
```

This generates a table showing Performance Drop vs Cost Advantage, matching the research paper format.

## Training on Different Model Pairs

### Example 1: GPT-4 vs Llama-2-7B

```bash
# Prepare data
python prepare_router_data.py \
    --input rlaif.parquet \
    --output gpt4_llama7b_data.parquet \
    --format rlaif \
    --strong_model "gpt-4" \
    --weak_model "llama-2-7b-chat"

# Split
python make_splits.py \
    --input gpt4_llama7b_data.parquet \
    --output_dir gpt4_llama7b_splits

# Train
python train_router.py \
    --data_path gpt4_llama7b_splits/router_train.parquet \
    --checkpoint_dir gpt4_llama7b_checkpoints

# Evaluate (update checkpoint paths in generate_perf_cost_table.py first)
python generate_perf_cost_table.py
```

### Example 2: GPT-3.5 vs Llama-2-13B

```bash
# Prepare data
python prepare_router_data.py \
    --input rlaif.parquet \
    --output gpt35_llama13b_data.parquet \
    --format rlaif \
    --strong_model "gpt-3.5-turbo" \
    --weak_model "llama-2-13b-chat"

# Split
python make_splits.py \
    --input gpt35_llama13b_data.parquet \
    --output_dir gpt35_llama13b_splits

# Train
python train_router.py \
    --data_path gpt35_llama13b_splits/router_train.parquet \
    --checkpoint_dir gpt35_llama13b_checkpoints

# Evaluate (update checkpoint paths in generate_perf_cost_table.py first)
python generate_perf_cost_table.py
```

## File Structure

```
capstone/
├── prepare_router_data.py         # Unified data preparation
├── make_splits.py                  # Create train/val/test splits
├── train_router.py                 # Unified training script
├── generate_perf_cost_table.py    # Evaluation (research paper format)
├── router/                         # Router model implementation (core modules only)
│   ├── model.py                   # MatrixFactorizationRouter model
│   ├── dataset.py                 # Dataset and data loading utilities
│   └── inference.py               # Inference utilities
└── README_ROUTER.md               # This file
```

**Note**: All training, evaluation, and data preparation scripts are now at the root level for easy access. The `router/` folder contains only the core model implementation and utilities.

## Data Format

The router expects data in this format:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | str | User query/prompt |
| `strong_model` | str | Name of strong model |
| `weak_model` | str | Name of weak model |
| `strong_response` | str | Response from strong model (optional) |
| `weak_response` | str | Response from weak model (optional) |
| `label` | str | 'wins' (strong wins), 'winw' (weak wins), or 'tie' |
| `binary_label` | int | 1 if strong wins, 0 otherwise |

## Evaluation Metrics

The evaluation produces a table matching the research paper format showing:

- **Performance Drop**: How much quality degrades from "always strong" baseline (in %)
- **Cost Advantage**: Percentage of strong model calls avoided (cost savings in %)

The table shows these metrics at different performance drop tolerances (0.5%, 1.0%, 1.5%, etc.), allowing you to see the cost-quality tradeoff curve.

## Advanced Options

### No Class Balancing

```bash
python make_splits.py \
    --input router_data.parquet \
    --output_dir data \
    --no_balance
```
