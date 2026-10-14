import os
from pathlib import Path
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
api = HfApi()

repo_id = "MSourav/capstone-datasets"
print(f"Creating/getting dataset repo {repo_id}...")
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)

files_to_upload = [
    PROJECT_ROOT / "gpt4_llama7b_data_unbalanced.zip",
    PROJECT_ROOT / "gpt4_llama7b_router_data.parquet",
    PROJECT_ROOT / "rlaif.parquet",
    PROJECT_ROOT / "router/data/router_train_balanced.parquet",
    PROJECT_ROOT / "router/data/router_train_bge_large_embeddings.parquet"
]

for file_path in files_to_upload:
    if os.path.exists(file_path):
        print(f"Uploading {file_path}...")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=Path(file_path).relative_to(PROJECT_ROOT).as_posix(),
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"Successfully uploaded {file_path}")
    else:
        print(f"File {file_path} not found.")

print("All uploads complete.")
