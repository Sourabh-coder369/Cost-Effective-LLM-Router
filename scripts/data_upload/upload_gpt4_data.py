import os
from pathlib import Path
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
api = HfApi()
repo_id = "MSourav/capstone-datasets"

print(f"Ensuring repository {repo_id} exists...")
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)

folder_to_upload = PROJECT_ROOT / "data" / "original_gpt4_llama7b" / "gpt4_llama7b_data_unbalanced"

print(f"Uploading files from {folder_to_upload}...")
for root, _, files in os.walk(folder_to_upload):
    for file in files:
        file_path = os.path.join(root, file)
        # Normalize path for Hugging Face
        path_in_repo = Path(file_path).relative_to(folder_to_upload).as_posix()
        
        print(f"Uploading {file_path} -> {path_in_repo}")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"Successfully uploaded {file_path}")

print("Upload complete!")
