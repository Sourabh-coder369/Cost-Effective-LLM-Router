# Experiment: 128-Dimension Router with Unbalanced Data

$ExperimentName = "gpt4_llama7b_128dim_unbalanced"
$DataPath = "gpt4_llama7b_router_data.parquet"
$SplitsDir = "data/${ExperimentName}_splits"
$CheckpointsDir = "checkpoints/${ExperimentName}"

$PythonPath = ".\.venv\Scripts\python.exe"

# 1. Create Data Splits (No Balancing)
Write-Host "Creating data splits..."
& $PythonPath data_processing/make_splits.py `
    --input $DataPath `
    --output_dir $SplitsDir `
    --no_balance

# 2. Train Router (128 Dimensions)
Write-Host "Starting training..."
& $PythonPath core_model/train_router.py `
    --data_path "${SplitsDir}/router_train.parquet" `
    --checkpoint_dir $CheckpointsDir `
    --model_embedding_dim 128 `
    --query_embedding_dim 384 `
    --batch_size 128 `
    --num_epochs 10 `
    --learning_rate 0.001

Write-Host "Experiment complete. Checkpoints saved to $CheckpointsDir"
