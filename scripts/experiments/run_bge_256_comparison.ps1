# Reproduce baseline-vs-domain router savings, quality-drop, and threshold tables
# for 256-dimensional model vectors with BGE-large query embeddings.

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment not found: $python"
}

& $python `
    (Join-Path $projectRoot "router\domain_specific\evaluation\reproduce_comparison_results.py") `
    --test-data (Join-Path $projectRoot "data\original_gpt4_llama7b\gpt4_llama7b_data_unbalanced\router_test.parquet") `
    --test-embeddings (Join-Path $projectRoot "data\original_gpt4_llama7b\bge_embeddings\router_test_bge_large_embeddings.pt") `
    --baseline-checkpoint (Join-Path $projectRoot "router\baseline\checkpoints_bge_modeldim_256\best_model.pt") `
    --domain-checkpoint (Join-Path $projectRoot "router\domain_specific\checkpoints_domain_mf_bge_modeldim_256\best_model.pt") `
    --output (Join-Path $projectRoot "router\domain_specific\checkpoints_domain_mf_bge_modeldim_256\comparison_results.json") `
    --embedding-model "BAAI/bge-large-en-v1.5"

if ($LASTEXITCODE -ne 0) {
    throw "Comparison evaluation failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "Saved results to: router\domain_specific\checkpoints_domain_mf_bge_modeldim_256\comparison_results.json"
