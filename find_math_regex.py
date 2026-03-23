"""
Rigorous Regex-Based Math Query Finder
========================================
Finds STRICTLY math-related prompts using high-precision patterns.

Strategy:
  - Every pattern is tightly scoped to avoid false positives.
  - A SCORING system is used: a prompt needs >= MIN_SCORE to qualify.
  - Weak / noisy patterns (solve, how many, money) are REMOVED or tightened.

Pattern Groups (with score weights):
  [3pts] explicit_math_notation  - actual symbolic math: x^2+3=0, d/dx, ∫, ∑
  [3pts] calculus                - derivative, integral, limit, d/dx notation
  [3pts] linear_algebra          - eigenvalue, determinant, matrix operations
  [2pts] algebraic_equation      - solve for x/y/z with an actual equation
  [2pts] geometry_precise        - area/volume WITH numbers, pythagorean, etc.
  [2pts] probability_rigorous    - P(A), Bayes, expected value, variance formula
  [2pts] number_theory_strict    - GCD, LCM, prime factorization, modular arith
  [2pts] math_domain_explicit    - "math problem", "mathematics", "algebra question"
  [1pt]  arithmetic_expression   - bare numeric operations: 12*4+7, (3+5)/2
  [1pt]  powers_roots_notation   - x^2, sqrt(n), x², x³
  [1pt]  math_symbols            - ∑ ∏ ∫ ∂ ∇ ∞ π ≠ ≤ ≥ ∈

MIN_SCORE = 2  →  a prompt must accumulate at least 2 points.

Usage:
    python find_math_regex.py
"""

import re
import pandas as pd
import os

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
DATA_PATH  = "gpt4_llama7b_data_unbalanced/router_train.parquet"
OUTPUT_DIR = "cluster_results"
MIN_SCORE  = 1       # minimum score to classify as math (includes score-1 queries)

# ─────────────────────────────────────────────────────────────
# Scored Pattern Groups
# Each entry: (score_weight, compiled_regex)
# ─────────────────────────────────────────────────────────────

MATH_PATTERNS = {

    # ── [3pts] Explicit symbolic math notation ────────────────────────────────
    # Matches things like: 2x^2 - 5x + 3 = 0, d/dx(f(x)), f'(x), ∂f/∂x
    "explicit_math_notation": (3, re.compile(
        r"d\/d[a-z]\s*[\(\[]|"            # d/dx(...), d/dy[...]
        r"∂[a-zA-Z]\/∂[a-zA-Z]|"         # partial derivative notation
        r"f'\s*\(|f''\s*\(|"              # f'(x), f''(x)
        r"\b\d*[a-z]\s*\^\s*\d+\s*[\+\-]\s*\d*[a-z]|"  # 2x^2 + 3x
        r"\b[a-z]\s*\^\s*\d+\s*[+\-]\s*\d+\s*=\s*\d|"  # x^2 - 4 = 0
        r"\b\d+[a-z]\s*[\+\-]\s*\d+[a-z]\s*=|"         # 3x + 2y =
        r"\b[a-z]\s*=\s*[-]?\d+[a-z]\s*[\+\-]",        # y = 2x + 1
        re.IGNORECASE
    )),

    # ── [3pts] Calculus ───────────────────────────────────────────────────────
    # Only fire on actual calculus terms — not generic use of "limit"
    "calculus": (3, re.compile(
        r"\b(derivative|differentiate|antiderivative|indefinite integral"
        r"|definite integral|taylor series|maclaurin series|fourier transform"
        r"|laplace transform|partial derivative|gradient descent"
        r"|chain rule|product rule|quotient rule|riemann sum"
        r"|fundamental theorem of calculus)\b"
        r"|∫\s*[a-zA-Z\d]"                # ∫ followed by something
        r"|d\/d[a-zA-Z]\b",               # d/dx notation
        re.IGNORECASE
    )),

    # ── [3pts] Linear Algebra ─────────────────────────────────────────────────
    "linear_algebra": (3, re.compile(
        r"\b(eigenvalue|eigenvector|determinant|matrix multiplication"
        r"|dot product|cross product|transpose|null space|column space"
        r"|linear independence|orthogonal|orthonormal|rank of (?:a |the )?matrix"
        r"|singular value decomposition|SVD|LU decomposition|row echelon)\b",
        re.IGNORECASE
    )),

    # ── [2pts] Algebraic equation — must have a variable AND an equation ──────
    # e.g. "solve 2x + 5 = 11", "find x if 3x - 7 = 2"
    "algebraic_equation": (2, re.compile(
        r"\bsolve\b.{0,40}[a-z]\s*[\+\-\*\/\^].{0,20}=\s*[-\d]|"   # solve ... x+... = N
        r"\bfind\s+[a-z]\b.{0,30}=\s*[-\d]|"                        # find x ... = N
        r"\b(for\s+x|for\s+y|for\s+z|for\s+n)\b.{0,25}=|"          # "for x" ... =
        r"solve\s+(?:the\s+)?(?:equation|quadratic|system)|"         # solve the equation
        r"quadratic\s+(?:equation|formula)|"                          # quadratic equation
        r"system\s+of\s+(?:linear\s+)?equations|"                    # system of equations
        r"simultaneous\s+equations",
        re.IGNORECASE
    )),

    # ── [2pts] Geometry — must include numeric context or precise term ─────────
    "geometry_precise": (2, re.compile(
        r"\b(pythagorean theorem|pythagoras|hypotenuse|"
        r"law of sines|law of cosines|"
        r"area of (?:a |the )?(?:triangle|circle|rectangle|polygon|sector|ellipse)|"
        r"volume of (?:a |the )?(?:sphere|cube|cylinder|cone|pyramid)|"
        r"surface area|circumference of|"
        r"angle bisector|inscribed angle|central angle|"
        r"trigonometric identity|sin\^2|cos\^2|tan\^2|"
        r"(?:sin|cos|tan|sec|csc|cot)\s*\([\d°])\b",
        re.IGNORECASE
    )),

    # ── [2pts] Probability & Statistics — rigorous only ───────────────────────
    "probability_rigorous": (2, re.compile(
        r"\bP\s*\(\s*[A-Z]|"                                   # P(A), P(B|A)
        r"\b(Bayes(?:'s)? theorem|conditional probability|"
        r"probability distribution|normal distribution|binomial distribution"
        r"|poisson distribution|expected value|variance|standard deviation"
        r"|hypothesis test|null hypothesis|p-value|confidence interval"
        r"|central limit theorem|random variable|probability of|"
        r"permutation|combination|n choose k|binomial coefficient)\b",
        re.IGNORECASE
    )),

    # ── [2pts] Number Theory — strict terms only ──────────────────────────────
    "number_theory_strict": (2, re.compile(
        r"\b(greatest common (?:divisor|factor)|"
        r"least common multiple|"
        r"prime factori(?:z|s)ation|"
        r"modular arithmetic|modulo|"
        r"euler(?:'s)? (?:theorem|totient)|"
        r"fermat(?:'s)? (?:little theorem|last theorem)|"
        r"divisibility rule|"
        r"coprime|relatively prime|"
        r"arithmetic(?:\s+modulo|\s+progression\s+with)|"
        r"geometric progression with|"
        r"sum of (?:first\s+)?\d+ (?:terms|natural numbers))\b",
        re.IGNORECASE
    )),

    # ── [2pts] Explicit math domain declaration ───────────────────────────────
    # e.g. "solve this math problem", "help with my algebra homework"
    "math_domain_explicit": (2, re.compile(
        r"\b(math(?:ematics)?\s+(?:problem|question|homework|exercise|exam|test)|"
        r"algebra\s+(?:problem|question|homework)|"
        r"calculus\s+(?:problem|question|homework)|"
        r"geometry\s+(?:problem|question|homework)|"
        r"statistics\s+(?:problem|question|homework)|"
        r"(?:solve|answer)\s+this\s+(?:math|algebra|calculus|geometry)\b)",
        re.IGNORECASE
    )),

    # ── [1pt] Arithmetic expression — bare numeric computation ───────────────
    # Must have operator between two numbers, not just a version number (1.0.3)
    "arithmetic_expression": (1, re.compile(
        r"(?<![.\d])\d+\s*[\+\-\*×÷]\s*\d+(?![.\d])|"   # 12 + 4, 5 × 3
        r"\(\s*\d+\s*[\+\-\*]\s*\d+\s*\)\s*[\+\-\*\/]|"  # (3+5)*2
        r"\d+\s*\/\s*\d+\s*[\+\-\*]",                     # 3/4 + 1
    )),

    # ── [1pt] Powers and roots ────────────────────────────────────────────────
    "powers_roots": (1, re.compile(
        r"\b\w+\s*\^\s*\d+|"               # x^2, 3^n
        r"√\s*\d+|"                        # √25
        r"\bsqrt\s*\(\s*\d+|"             # sqrt(9)
        r"\bsquare\s+root\s+of\s+\d+|"    # square root of 16
        r"\bcube\s+root\s+of\s+\d+|"      # cube root of 27
        r"[²³⁴⁵⁶⁷⁸⁹]\b",                 # Unicode superscripts
        re.IGNORECASE
    )),

    # ── [1pt] Math symbols ────────────────────────────────────────────────────
    "math_symbols": (1, re.compile(
        r"[∑∏∫∂∇∞π±≠≤≥≈∈∉⊂⊃∪∩θλμσΩ]"
    )),
}


def extract_prompt_text(raw: str) -> str:
    """Strip chat formatting to get just the user's question."""
    if "Human:" in raw:
        raw = raw.split("Human:")[-1].split("Assistant:")[0]
    return raw.strip()


def score_prompt(prompt: str) -> tuple[int, list[str]]:
    """
    Returns (total_score, matched_pattern_names).
    Higher score = more confidently a math query.
    """
    text = extract_prompt_text(prompt)
    total_score = 0
    matched = []
    for name, (weight, pattern) in MATH_PATTERNS.items():
        if pattern.search(text):
            total_score += weight
            matched.append(f"{name}(+{weight})")
    return total_score, matched


def main():
    print("=" * 65)
    print("  Rigorous Regex-Based Math Query Finder")
    print(f"  Minimum score to qualify: {MIN_SCORE} pts")
    print("=" * 65)

    df = pd.read_parquet(DATA_PATH)
    print(f"\n  Loaded {len(df):,} prompts from {DATA_PATH}")

    print("\n  Scoring prompts...")
    scored = df["prompt"].apply(score_prompt)
    df["math_score"]      = scored.apply(lambda x: x[0])
    df["matched_patterns"]= scored.apply(lambda x: x[1])
    df["is_math"]         = df["math_score"] >= MIN_SCORE

    math_df     = df[df["is_math"]]
    non_math_df = df[~df["is_math"]]

    # ── Overall results ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Total prompts    : {len(df):,}")
    print(f"  Math prompts     : {len(math_df):,}  ({len(math_df)/len(df)*100:.1f}%)")
    print(f"  Non-math prompts : {len(non_math_df):,}  ({len(non_math_df)/len(df)*100:.1f}%)")

    print(f"\n  Label breakdown — MATH queries (rigorously filtered):")
    print(f"    Strong model wins (wins) : "
          f"{(math_df['label']=='wins').sum():,}  "
          f"({(math_df['label']=='wins').mean()*100:.1f}%)")
    print(f"    Weak model wins   (winw) : "
          f"{(math_df['label']=='winw').sum():,}  "
          f"({(math_df['label']=='winw').mean()*100:.1f}%)")

    print(f"\n  Label breakdown — NON-MATH queries:")
    print(f"    Strong model wins (wins) : "
          f"{(non_math_df['label']=='wins').sum():,}  "
          f"({(non_math_df['label']=='wins').mean()*100:.1f}%)")
    print(f"    Weak model wins   (winw) : "
          f"{(non_math_df['label']=='winw').sum():,}  "
          f"({(non_math_df['label']=='winw').mean()*100:.1f}%)")

    # ── Score distribution ────────────────────────────────────────────────────
    print(f"\n  Math score distribution:")
    for score in sorted(df["math_score"].unique()):
        count = (df["math_score"] == score).sum()
        tag   = " ← classified as MATH" if score >= MIN_SCORE else ""
        bar   = "█" * min(count * 30 // max((df["math_score"] > 0).sum(), 1), 40)
        print(f"    Score {score:>2} : {count:>6,}  {bar}{tag}")

    # ── Pattern hit frequency ─────────────────────────────────────────────────
    print(f"\n  Pattern hit frequency (math queries only):")
    for name, (weight, _) in MATH_PATTERNS.items():
        count = df[df["is_math"]]["matched_patterns"].apply(
            lambda x: any(name in p for p in x)
        ).sum()
        bar = "█" * (count * 30 // max(len(math_df), 1))
        print(f"    {name:<28} [+{weight}pt] : {count:>5,}  {bar}")

    # ── Sample rigorous math prompts ──────────────────────────────────────────
    print(f"\n  Sample RIGOROUS math prompts (sorted by score):")
    top_math = df[df["is_math"]].nlargest(20, "math_score")
    for i, (_, row) in enumerate(top_math.iterrows()):
        text = extract_prompt_text(row["prompt"])
        patterns_str = ", ".join(row["matched_patterns"][:3])
        print(f"    {i+1:>2}. [score={row['math_score']}] [{row['label']}] "
              f"({patterns_str})")
        print(f"        {text[:100].replace(chr(10), ' ')}")

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_path = os.path.join(OUTPUT_DIR, "rigorous_math_queries.parquet")
    math_df[["prompt", "label", "binary_label", "math_score", "matched_patterns"]].to_parquet(
        out_path, index=False
    )
    print(f"\n  Saved rigorous math queries to : {out_path}")

    out_csv = os.path.join(OUTPUT_DIR, "rigorous_math_queries.csv")
    math_df[["prompt", "label", "binary_label", "math_score"]].to_csv(out_csv, index=False)
    print(f"  Saved CSV to                   : {out_csv}")
    print("=" * 65)


if __name__ == "__main__":
    main()
