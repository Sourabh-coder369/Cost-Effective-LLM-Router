param(
    [Parameter(Mandatory = $true)] [string] $TrainEmbeddings,
    [Parameter(Mandatory = $true)] [string] $ValEmbeddings,
    [Parameter(Mandatory = $true)] [string] $TestEmbeddings,
    [string] $OutputDir = "artifacts/gpt_llama_soft_domain_router"
)

$dataDir = "data/original_gpt4_llama7b/gpt4_llama7b_data_unbalanced"

python router_improved/train.py `
  --train-data "$dataDir/router_train.parquet" --train-embeddings $TrainEmbeddings `
  --val-data "$dataDir/router_val.parquet" --val-embeddings $ValEmbeddings `
  --test-data "$dataDir/router_test.parquet" --test-embeddings $TestEmbeddings `
  --output-dir $OutputDir
