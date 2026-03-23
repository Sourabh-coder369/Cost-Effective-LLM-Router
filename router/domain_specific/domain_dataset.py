"""
Domain-Aware Dataset for Domain-Specific MF Router
====================================================
Extends RouterDataset to also return a domain_id (0=general, 1=math)
per sample, derived from the regex-based math labelling.

Reuses pre-computed BGE Large embeddings from:
  router/data/router_train_bge_large_embeddings.pt
  router/data/router_val_bge_large_embeddings.pt
  router/data/router_test_bge_large_embeddings.pt
"""

import re
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import os
import sys

# ── Math regex patterns (same as find_math_regex.py, score ≥ 1) ─────────────
MATH_PATTERNS = {
    "explicit_math_notation": (3, re.compile(
        r"d\/d[a-z]\s*[\(\[]|∂[a-zA-Z]\/∂[a-zA-Z]|f'\s*\(|f''\s*\(|"
        r"\b\d*[a-z]\s*\^\s*\d+\s*[\+\-]\s*\d*[a-z]|"
        r"\b[a-z]\s*\^\s*\d+\s*[+\-]\s*\d+\s*=\s*\d|"
        r"\b\d+[a-z]\s*[\+\-]\s*\d+[a-z]\s*=|\b[a-z]\s*=\s*[-]?\d+[a-z]\s*[\+\-]",
        re.IGNORECASE
    )),
    "calculus": (3, re.compile(
        r"\b(derivative|differentiate|antiderivative|indefinite integral|definite integral"
        r"|taylor series|maclaurin series|fourier transform|laplace transform"
        r"|partial derivative|gradient descent|chain rule|product rule|quotient rule"
        r"|riemann sum|fundamental theorem of calculus)\b|∫\s*[a-zA-Z\d]|d\/d[a-zA-Z]\b",
        re.IGNORECASE
    )),
    "linear_algebra": (3, re.compile(
        r"\b(eigenvalue|eigenvector|determinant|matrix multiplication|dot product"
        r"|cross product|transpose|null space|column space|linear independence"
        r"|orthogonal|orthonormal|rank of (?:a |the )?matrix|singular value decomposition"
        r"|SVD|LU decomposition|row echelon)\b",
        re.IGNORECASE
    )),
    "algebraic_equation": (2, re.compile(
        r"\bsolve\b.{0,40}[a-z]\s*[\+\-\*\/\^].{0,20}=\s*[-\d]|"
        r"\bfind\s+[a-z]\b.{0,30}=\s*[-\d]|"
        r"\b(for\s+x|for\s+y|for\s+z|for\s+n)\b.{0,25}=|"
        r"solve\s+(?:the\s+)?(?:equation|quadratic|system)|quadratic\s+(?:equation|formula)|"
        r"system\s+of\s+(?:linear\s+)?equations|simultaneous\s+equations",
        re.IGNORECASE
    )),
    "geometry_precise": (2, re.compile(
        r"\b(pythagorean theorem|pythagoras|hypotenuse|law of sines|law of cosines|"
        r"area of (?:a |the )?(?:triangle|circle|rectangle|polygon|sector|ellipse)|"
        r"volume of (?:a |the )?(?:sphere|cube|cylinder|cone|pyramid)|surface area|"
        r"circumference of|angle bisector|inscribed angle|central angle|"
        r"trigonometric identity|sin\^2|cos\^2|tan\^2|"
        r"(?:sin|cos|tan|sec|csc|cot)\s*\([\d°])\b",
        re.IGNORECASE
    )),
    "probability_rigorous": (2, re.compile(
        r"\bP\s*\(\s*[A-Z]|"
        r"\b(Bayes(?:'s)? theorem|conditional probability|probability distribution"
        r"|normal distribution|binomial distribution|poisson distribution|expected value"
        r"|variance|standard deviation|hypothesis test|null hypothesis|p-value"
        r"|confidence interval|central limit theorem|random variable|probability of|"
        r"permutation|combination|n choose k|binomial coefficient)\b",
        re.IGNORECASE
    )),
    "number_theory_strict": (2, re.compile(
        r"\b(greatest common (?:divisor|factor)|least common multiple|"
        r"prime factori(?:z|s)ation|modular arithmetic|modulo|"
        r"euler(?:'s)? (?:theorem|totient)|fermat(?:'s)? (?:little theorem|last theorem)|"
        r"divisibility rule|coprime|relatively prime|"
        r"arithmetic(?:\s+modulo|\s+progression\s+with)|geometric progression with|"
        r"sum of (?:first\s+)?\d+ (?:terms|natural numbers))\b",
        re.IGNORECASE
    )),
    "math_domain_explicit": (2, re.compile(
        r"\b(math(?:ematics)?\s+(?:problem|question|homework|exercise|exam|test)|"
        r"algebra\s+(?:problem|question|homework)|calculus\s+(?:problem|question|homework)|"
        r"geometry\s+(?:problem|question|homework)|statistics\s+(?:problem|question|homework)|"
        r"(?:solve|answer)\s+this\s+(?:math|algebra|calculus|geometry)\b)",
        re.IGNORECASE
    )),
    "arithmetic_expression": (1, re.compile(
        r"(?<![.\d])\d+\s*[\+\-\*×÷]\s*\d+(?![.\d])|"
        r"\(\s*\d+\s*[\+\-\*]\s*\d+\s*\)\s*[\+\-\*\/]|\d+\s*\/\s*\d+\s*[\+\-\*]",
    )),
    "powers_roots": (1, re.compile(
        r"\b\w+\s*\^\s*\d+|√\s*\d+|\bsqrt\s*\(\s*\d+|\bsquare\s+root\s+of\s+\d+|"
        r"\bcube\s+root\s+of\s+\d+|[²³⁴⁵⁶⁷⁸⁹]\b",
        re.IGNORECASE
    )),
    "math_symbols": (1, re.compile(r"[∑∏∫∂∇∞π±≠≤≥≈∈∉⊂⊃∪∩θλμσΩ]")),
}

MIN_MATH_SCORE = 1   # score ≥ 1 → math domain


def _extract_text(raw: str) -> str:
    if "Human:" in raw:
        raw = raw.split("Human:")[-1].split("Assistant:")[0]
    return raw.strip()


def _math_score(prompt: str) -> int:
    text = _extract_text(prompt)
    return sum(w for _, (w, p) in MATH_PATTERNS.items() if p.search(text))


def _domain_id(prompt: str) -> int:
    """0 = general, 1 = math"""
    return 1 if _math_score(prompt) >= MIN_MATH_SCORE else 0


# ─────────────────────────────────────────────────────────────────────────────

class DomainRouterDataset(Dataset):
    """
    Dataset yielding (embedding, binary_label, domain_id) triples.

    domain_id: 0 = general, 1 = math  (derived via regex scoring)
    """

    def __init__(
        self,
        data_path: str,
        precomputed_embeddings_path: Optional[str] = None,
    ):
        print(f"Loading data from {data_path}...")
        self.df = pd.read_parquet(data_path)
        print(f"  rows: {len(self.df):,}")

        # Load embeddings
        if precomputed_embeddings_path and os.path.exists(precomputed_embeddings_path):
            print(f"  Loading precomputed embeddings: {precomputed_embeddings_path}")
            self.embeddings = torch.load(precomputed_embeddings_path, map_location="cpu")
            assert len(self.embeddings) == len(self.df), (
                f"Embedding count {len(self.embeddings)} != rows {len(self.df)}"
            )
            print(f"  Embed shape: {self.embeddings.shape}")
        else:
            raise FileNotFoundError(
                f"Precomputed embeddings not found: {precomputed_embeddings_path}\n"
                "Please provide BGE Large .pt embedding files."
            )

        # Compute domain labels
        print("  Computing domain labels (math vs general)...")
        self.domain_ids = torch.tensor(
            self.df["prompt"].apply(_domain_id).values,
            dtype=torch.long,
        )
        math_count = (self.domain_ids == 1).sum().item()
        print(f"  Math queries: {math_count:,} ({math_count/len(self.df)*100:.1f}%)")
        print(f"  General queries: {len(self.df)-math_count:,} ({(len(self.df)-math_count)/len(self.df)*100:.1f}%)")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding  = self.embeddings[idx]
        label      = torch.tensor(self.df.iloc[idx]["binary_label"], dtype=torch.float32)
        domain_id  = self.domain_ids[idx]
        return embedding, label, domain_id


def create_domain_dataloaders(
    train_path: str,
    train_embeddings_path: str,
    val_path: Optional[str] = None,
    val_embeddings_path: Optional[str] = None,
    batch_size: int = 64,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, int]:
    """Create train/val DataLoaders for the DomainRouterDataset."""

    if val_path:
        train_ds = DomainRouterDataset(train_path, train_embeddings_path)
        val_ds   = DomainRouterDataset(val_path,   val_embeddings_path)
    else:
        full_ds  = DomainRouterDataset(train_path, train_embeddings_path)
        n_val    = int(len(full_ds) * val_split)
        n_train  = len(full_ds) - n_val
        torch.manual_seed(seed)
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    def collate_fn(batch):
        embs, labels, domains = zip(*batch)
        return (
            torch.stack(embs),
            torch.stack(labels),
            torch.stack(domains),
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    emb_dim = train_ds.dataset.embeddings.shape[1] if hasattr(train_ds, "dataset") \
              else train_ds.embeddings.shape[1]

    return train_loader, val_loader, emb_dim
