"""Compute the risk-coverage curve and confidence intervals from a saved run.

The saved `_per_row` block (per-question confidence and correctness) is enough to
rebuild the whole deployment artifact without retraining. Run after
`lora_vs_head_only.py`; writes a markdown table to stdout.

Clopper-Pearson intervals, exact binomial, no SciPy. Both bounds verified against
the rule of three: 0 failures in n trials gives a 95% lower bound on accuracy of
1 - 0.05**(1/n), i.e. 0.819 at n=15.
"""
from __future__ import annotations

import json
import sys
from math import comb


def p_ge(k: int, n: int, p: float) -> float:
    """P(X >= k) for X~Bin(n,p). Increasing in p."""
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def p_le(k: int, n: int, p: float) -> float:
    """P(X <= k) for X~Bin(n,p). Decreasing in p when k < n."""
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def cp_lower(kc: int, n: int, a: float = 0.05) -> float:
    if kc == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if p_ge(kc, n, mid) < a:
            lo = mid
        else:
            hi = mid
    return hi


def cp_upper(kc: int, n: int, a: float = 0.05) -> float:
    if kc == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if p_le(kc, n, mid) > a:
            lo = mid
        else:
            hi = mid
    return hi


def curve(conf, corr, budgets=(0.02, 0.05, 0.10, 0.20, 0.30)):
    n = len(corr)
    order = sorted(range(n), key=lambda i: -conf[i])
    rows = []
    for frac in budgets:
        k = max(1, int(frac * n))
        kc = sum(corr[i] for i in order[:k])
        rows.append((k / n, k, k - kc, kc / k, cp_lower(kc, k), cp_upper(kc, k)))
    return rows, n


def main(path: str) -> int:
    data = json.load(open(path))
    for r in data["results"]:
        pr = r.get("_per_row")
        if not pr:
            continue
        rows, n = curve(pr["confidence"], pr["correct"])
        print(f"\n### {r['arm']}  (n={n}, fitted T={r['temperature']})\n")
        print(f"| coverage | k | errors | accuracy | 95% CI |")
        print("|---:|---:|---:|---:|---:|")
        for cov, k, err, acc, lo, hi in rows:
            print(f"| {cov:.0%} | {k} | {err} | {acc:.3f} | [{lo:.3f}, {hi:.3f}] |")
        errs = 0
        best = 0
        conf, corr = pr["confidence"], pr["correct"]
        order = sorted(range(n), key=lambda i: -conf[i])
        print()
        for a in (0.02, 0.05, 0.10, 0.20, 0.30):
            e = 0
            b = 0
            for k, i in enumerate(order, 1):
                e += 0 if corr[i] else 1
                if e / k <= a:
                    b = k
            print(f"  error <= {a:>4.0%}  ->  coverage {b / n:>6.1%}  ({b}/{n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "runs/cov-full/customer_service.json"))
