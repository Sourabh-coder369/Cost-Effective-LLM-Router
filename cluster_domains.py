"""
Embedding-Based Domain Clustering for the Unbalanced Dataset
=============================================================
Groups prompts into semantic domains (math, coding, creative writing, etc.)
using pre-cached sentence embeddings + KMeans clustering.

Two approaches:
  1. KMeans unsupervised clustering — discover all domain clusters
  2. Seed-based similarity — find math queries using example anchors

Usage:
    python cluster_domains.py
"""

import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import os

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
DATA_PATH = "gpt4_llama7b_data_unbalanced/router_train.parquet"
EMBEDDING_CACHE_DIR = "gpt4_llama7b_data_unbalanced/.embedding_cache"
# The 768-dim cached embeddings file
EMBEDDING_FILE = os.path.join(EMBEDDING_CACHE_DIR, "router_train.parquet_6b59c6b9cae834b6.pt")

NUM_CLUSTERS = 12       # Number of KMeans clusters
MATH_SIM_THRESHOLD = 0.5  # Cosine similarity threshold for seed-based math detection
OUTPUT_DIR = "cluster_results"


def load_data():
    """Load the parquet data and cached embeddings."""
    print("=" * 60)
    print("  Loading Data")
    print("=" * 60)

    df = pd.read_parquet(DATA_PATH)
    print(f"  Loaded {len(df)} prompts from {DATA_PATH}")

    embeddings = torch.load(EMBEDDING_FILE, map_location="cpu").numpy()
    print(f"  Loaded embeddings: shape {embeddings.shape}")
    
    assert len(df) == len(embeddings), (
        f"Mismatch: {len(df)} rows vs {len(embeddings)} embeddings"
    )
    return df, embeddings


# ──────────────────────────────────────────────────────────────
# APPROACH 1: KMeans Unsupervised Clustering
# ──────────────────────────────────────────────────────────────
def kmeans_clustering(df, embeddings, n_clusters=NUM_CLUSTERS):
    """
    Cluster all prompt embeddings using KMeans.
    
    Steps:
      1. Normalize embeddings (for cosine-like behavior)
      2. Fit KMeans
      3. For each cluster, print sample prompts and label distribution
      4. Identify which cluster(s) are "math"
    """
    print("\n" + "=" * 60)
    print(f"  APPROACH 1: KMeans Clustering (k={n_clusters})")
    print("=" * 60)

    # Step 1: L2-normalize embeddings so KMeans uses cosine-like distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed_embeddings = embeddings / (norms + 1e-10)

    # Step 2: Fit KMeans
    print(f"\n  Fitting KMeans with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(normed_embeddings)
    
    df["cluster"] = cluster_labels
    print(f"  Done. Inertia: {kmeans.inertia_:.2f}")

    # Step 3: Analyze each cluster
    print(f"\n  {'Cluster':>8} | {'Size':>6} | {'Strong%':>8} | {'Weak%':>7} | Sample Prompts")
    print("  " + "-" * 90)

    cluster_summaries = []

    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_df = df[mask]
        size = mask.sum()
        
        label_dist = cluster_df["label"].value_counts()
        wins_pct = label_dist.get("wins", 0) / size * 100
        winw_pct = label_dist.get("winw", 0) / size * 100

        # Get 3 short sample prompts
        samples = []
        for _, row in cluster_df.head(3).iterrows():
            # Extract just the user prompt, strip formatting
            prompt = row["prompt"].strip()
            # Try to get just the human part
            if "Human:" in prompt:
                prompt = prompt.split("Human:")[-1].split("Assistant:")[0].strip()
            samples.append(prompt[:80].replace("\n", " "))

        sample_str = " | ".join(samples)
        print(f"  {c:>8} | {size:>6} | {wins_pct:>7.1f}% | {winw_pct:>6.1f}% | {sample_str[:70]}")

        cluster_summaries.append({
            "cluster": c,
            "size": int(size),
            "strong_win_pct": round(wins_pct, 1),
            "weak_win_pct": round(winw_pct, 1),
            "sample_1": samples[0] if len(samples) > 0 else "",
            "sample_2": samples[1] if len(samples) > 1 else "",
            "sample_3": samples[2] if len(samples) > 2 else "",
        })

    # Step 4: Identify math cluster(s) using keyword heuristic
    math_keywords = [
        "math", "calculate", "equation", "algebra", "solve", "derivative",
        "integral", "theorem", "formula", "geometry", "calculus", "probability",
        "arithmetic", "polynomial", "fraction", "percentage",
    ]
    print("\n  Identifying math-heavy clusters...")
    
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_prompts = df[mask]["prompt"].str.lower()
        
        math_count = 0
        for kw in math_keywords:
            math_count += cluster_prompts.str.contains(kw, na=False).sum()
        
        # Normalize by cluster size
        math_density = math_count / mask.sum()
        if math_density > 0.3:  # If >30% of prompts contain math keywords
            print(f"  ★ Cluster {c} is likely MATH (math keyword density: {math_density:.2f})")

    return df, kmeans, cluster_summaries


# ──────────────────────────────────────────────────────────────
# APPROACH 2: Seed-Based Math Detection
# ──────────────────────────────────────────────────────────────
def seed_based_math_detection(df, embeddings, threshold=MATH_SIM_THRESHOLD):
    """
    Identify math queries by similarity to math 'seed' examples.
    
    Steps:
      1. Define seed math queries
      2. Embed them using the same model
      3. Compute centroid of seed embeddings
      4. Measure cosine similarity of all prompts to the centroid
      5. Threshold to identify math queries
    """
    print("\n" + "=" * 60)
    print("  APPROACH 2: Seed-Based Math Detection")
    print("=" * 60)

    # Step 1: Define math seed queries (diverse math types)
    math_seeds = [
        "Calculate the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
        "Solve the quadratic equation 2x^2 - 5x + 3 = 0",
        "What is the probability of rolling two sixes with two dice?",
        "Find the area of a triangle with base 5 cm and height 3 cm",
        "Prove that the sum of angles in a triangle is 180 degrees",
        "Compute the integral of sin(x) from 0 to pi",
        "Simplify the expression (3x + 2)(x - 4)",
        "What is 15% of 240?",
        "Find the eigenvalues of the matrix [[1,2],[3,4]]",
        "A train travels 120 km in 2 hours. What is its average speed?",
        "How many ways can you choose 3 items from 10?",
        "Solve the system of equations: 2x + y = 10, x - y = 2",
    ]

    # Step 2: Embed seeds using the same sentence transformer
    print("\n  Embedding seed math queries...")
    try:
        from sentence_transformers import SentenceTransformer
        # Detect model from embedding dim
        dim = embeddings.shape[1]
        if dim == 768:
            model_name = "sentence-transformers/all-mpnet-base-v2"
        elif dim == 384:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        elif dim == 1024:
            model_name = "BAAI/bge-large-en-v1.5"
        else:
            model_name = "sentence-transformers/all-mpnet-base-v2"
        
        print(f"  Using model: {model_name} (dim={dim})")
        st_model = SentenceTransformer(model_name)
        seed_embeddings = st_model.encode(math_seeds, convert_to_numpy=True)
    except Exception as e:
        print(f"  Error loading model: {e}")
        print("  Falling back to keyword-based approach.")
        return df

    # Step 3: Compute centroid of seed embeddings
    seed_centroid = seed_embeddings.mean(axis=0, keepdims=True)
    # Also normalize
    seed_centroid = seed_centroid / (np.linalg.norm(seed_centroid) + 1e-10)
    
    # Normalize all embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed_embeddings = embeddings / (norms + 1e-10)

    # Step 4: Cosine similarity to math centroid
    similarities = cosine_similarity(normed_embeddings, seed_centroid).flatten()

    # Step 5: Try multiple thresholds
    print(f"\n  Similarity statistics:")
    print(f"    Min:    {similarities.min():.4f}")
    print(f"    Max:    {similarities.max():.4f}")
    print(f"    Mean:   {similarities.mean():.4f}")
    print(f"    Median: {np.median(similarities):.4f}")
    print(f"    Std:    {similarities.std():.4f}")

    print(f"\n  {'Threshold':>10} | {'Math Queries':>12} | {'% of Total':>10} | {'Strong Win%':>11} | {'Weak Win%':>9}")
    print("  " + "-" * 65)

    best_threshold = threshold
    for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        math_mask = similarities >= t
        n_math = math_mask.sum()
        
        if n_math > 0:
            math_df = df[math_mask]
            wins_pct = (math_df["label"] == "wins").mean() * 100
            winw_pct = (math_df["label"] == "winw").mean() * 100
        else:
            wins_pct = winw_pct = 0
        
        marker = " ◄" if t == threshold else ""
        print(f"  {t:>10.2f} | {n_math:>12,} | {n_math/len(df)*100:>9.1f}% | {wins_pct:>10.1f}% | {winw_pct:>8.1f}%{marker}")

    # Apply chosen threshold
    df["math_similarity"] = similarities
    df["is_math"] = similarities >= best_threshold

    math_count = df["is_math"].sum()
    print(f"\n  Using threshold {best_threshold}: {math_count:,} math queries identified ({math_count/len(df)*100:.1f}%)")

    # Show top math queries (highest similarity)
    print(f"\n  Top 10 most 'math-like' prompts:")
    top_math = df.nlargest(10, "math_similarity")
    for i, (_, row) in enumerate(top_math.iterrows()):
        prompt = row["prompt"].strip()
        if "Human:" in prompt:
            prompt = prompt.split("Human:")[-1].split("Assistant:")[0].strip()
        print(f"    {i+1:>2}. (sim={row['math_similarity']:.3f}, label={row['label']}) {prompt[:90]}")

    # Show borderline queries (near threshold)
    print(f"\n  Borderline prompts (sim ≈ {best_threshold}):")
    near_threshold = df.iloc[
        (df["math_similarity"] - best_threshold).abs().argsort()[:5]
    ]
    for _, row in near_threshold.iterrows():
        prompt = row["prompt"].strip()
        if "Human:" in prompt:
            prompt = prompt.split("Human:")[-1].split("Assistant:")[0].strip()
        print(f"    (sim={row['math_similarity']:.3f}, label={row['label']}) {prompt[:90]}")

    return df


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    df, embeddings = load_data()

    # Run both approaches
    df, kmeans, cluster_summaries = kmeans_clustering(df, embeddings)
    df = seed_based_math_detection(df, embeddings)

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save cluster assignments
    cluster_output = df[["prompt", "label", "binary_label", "cluster"]].copy()
    if "is_math" in df.columns:
        cluster_output["is_math"] = df["is_math"]
        cluster_output["math_similarity"] = df["math_similarity"]
    
    output_path = os.path.join(OUTPUT_DIR, "clustered_train.parquet")
    cluster_output.to_parquet(output_path, index=False)
    print(f"\n  Saved cluster assignments to: {output_path}")

    # Save cluster summaries
    summary_df = pd.DataFrame(cluster_summaries)
    summary_path = os.path.join(OUTPUT_DIR, "cluster_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved cluster summary to: {summary_path}")

    # Final summary
    if "is_math" in df.columns:
        math_df = df[df["is_math"]]
        non_math_df = df[~df["is_math"]]
        print(f"\n" + "=" * 60)
        print(f"  FINAL SUMMARY")
        print(f"=" * 60)
        print(f"  Total prompts:       {len(df):,}")
        print(f"  Math prompts:        {len(math_df):,} ({len(math_df)/len(df)*100:.1f}%)")
        print(f"  Non-math prompts:    {len(non_math_df):,} ({len(non_math_df)/len(df)*100:.1f}%)")
        print(f"")
        print(f"  Math — strong wins:  {(math_df['label']=='wins').sum():,} ({(math_df['label']=='wins').mean()*100:.1f}%)")
        print(f"  Math — weak wins:    {(math_df['label']=='winw').sum():,} ({(math_df['label']=='winw').mean()*100:.1f}%)")
        print(f"  Other — strong wins: {(non_math_df['label']=='wins').sum():,} ({(non_math_df['label']=='wins').mean()*100:.1f}%)")
        print(f"  Other — weak wins:   {(non_math_df['label']=='winw').sum():,} ({(non_math_df['label']=='winw').mean()*100:.1f}%)")
        print(f"=" * 60)


if __name__ == "__main__":
    main()
