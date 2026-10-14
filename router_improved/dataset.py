"""Data loading for the isolated GPT-4/Llama routing experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class PreferenceDataset(Dataset):
    """Pairs saved query embeddings with the existing strong-wins label."""

    def __init__(self, data_path: Path, embeddings_path: Path) -> None:
        self.data_path = Path(data_path)
        self.embeddings_path = Path(embeddings_path)
        self.frame = pd.read_parquet(self.data_path)
        if "binary_label" not in self.frame:
            if "label" not in self.frame:
                raise ValueError("Expected either binary_label or label in the Parquet file")
            self.frame["binary_label"] = (self.frame["label"] == "wins").astype("int64")

        self.embeddings = torch.load(self.embeddings_path, map_location="cpu", weights_only=False)
        if len(self.embeddings) != len(self.frame):
            raise ValueError(
                f"Embedding rows ({len(self.embeddings)}) do not match data rows ({len(self.frame)}): "
                f"{self.embeddings_path}"
            )
        self.embeddings = self.embeddings.float()
        self.labels = torch.tensor(self.frame["binary_label"].to_numpy(), dtype=torch.float32)

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[index], self.labels[index]
