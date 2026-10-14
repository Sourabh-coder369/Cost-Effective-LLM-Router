# Retrain the BGE baseline and regex DomainMF at the default 256-D model size,
# then compare both on the held-out test split.

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$dataDir = Join-Path $projectRoot "data\original_gpt4_llama7b\gpt4_llama7b_data_unbalanced"
$embeddingDir = Join-Path $projectRoot "data\original_gpt4_llama7b\bge_embeddings"
$outputDir = Join-Path $projectRoot "artifacts\regex_bge256_retrained"
$baselineDir = Join-Path $outputDir "baseline"
$domainDir = Join-Path $outputDir "domain_mf"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment not found: $python"
}

& $python (Join-Path $projectRoot "router\baseline\train_router.py") `
    --data_path (Join-Path $dataDir "router_train.parquet") `
    --val_data_path (Join-Path $dataDir "router_val.parquet") `
    --train_embeddings (Join-Path $embeddingDir "router_train_bge_large_embeddings.pt") `
    --val_embeddings (Join-Path $embeddingDir "router_val_bge_large_embeddings.pt") `
    --query_embedding_dim 1024 `
    --embedding_model "BAAI/bge-large-en-v1.5" `
    --checkpoint_dir $baselineDir
if ($LASTEXITCODE -ne 0) { throw "Baseline training failed with exit code $LASTEXITCODE." }

& $python (Join-Path $projectRoot "router\domain_specific\train_explicit_splits.py") `
    --train-data (Join-Path $dataDir "router_train.parquet") `
    --train-embeddings (Join-Path $embeddingDir "router_train_bge_large_embeddings.pt") `
    --val-data (Join-Path $dataDir "router_val.parquet") `
    --val-embeddings (Join-Path $embeddingDir "router_val_bge_large_embeddings.pt") `
    --output-dir $domainDir `
    --epochs 10 `
    --batch-size 512 `
    --learning-rate 3e-4 `
    --seed 42
if ($LASTEXITCODE -ne 0) { throw "Regex DomainMF training failed with exit code $LASTEXITCODE." }

& $python (Join-Path $projectRoot "router\domain_specific\evaluation\reproduce_comparison_results.py") `
    --test-data (Join-Path $dataDir "router_test.parquet") `
    --test-embeddings (Join-Path $embeddingDir "router_test_bge_large_embeddings.pt") `
    --baseline-checkpoint (Join-Path $baselineDir "best_model.pt") `
    --domain-checkpoint (Join-Path $domainDir "best_model.pt") `
    --output (Join-Path $outputDir "comparison_results.json") `
    --embedding-model "BAAI/bge-large-en-v1.5"
if ($LASTEXITCODE -ne 0) { throw "Comparison evaluation failed with exit code $LASTEXITCODE." }

Write-Host ""
Write-Host "Completed retraining and comparison. Results: $outputDir"
