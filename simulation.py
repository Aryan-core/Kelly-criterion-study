"""
simulation.py

Kelly Criterion Monte Carlo simulation (no plots).

Model:
- Repeated i.i.d. bet on a single risky opportunity.
- With prob p: return is +b (profit multiple on the bet fraction)
- With prob 1-p: return is -a (loss multiple on the bet fraction)

If you bet fraction f of wealth each round, wealth update is:
    W_{t+1} = W_t * (1 + f * R_t),
where R_t is +b or -a.

We report, across M simulated paths of length T:
- mean(log W_T) / T  (empirical long-run log-growth rate)
- median(log W_T) / T
- probability of ending below starting wealth (W_T < 1)
- mean maximum drawdown (MDD) along the path
- median maximum drawdown

Admissibility:
- Must keep 1 + f*R_t > 0 in every outcome, so require f < 1/a.
"""

from __future__ import annotations
import math
import numpy as np


# ----------------------------
# Core simulation
# ----------------------------
def simulate_paths(
    p: float,
    b: float,
    a: float,
    f: float,
    T: int,
    M: int,
    seed: int = 0,
) -> dict:
    """
    Simulate M paths of length T for a given Kelly fraction f.

    Returns summary stats on:
      - per-round log growth: log(W_T)/T
      - max drawdown (MDD) over each path
      - prob of finishing below 1
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1).")
    if b <= 0 or a <= 0:
        raise ValueError("b and a must be > 0.")
    if T <= 0 or M <= 0:
        raise ValueError("T and M must be positive integers.")

    # Admissibility: worst outcome is R = -a, need 1 - f*a > 0
    # We'll allow tiny numerical slack.
    if f >= (1.0 / a):
        raise ValueError(f"Fraction f={f} violates admissibility: need f < 1/a = {1.0/a:.6f}")

    rng = np.random.default_rng(seed)

    # Pre-sample outcomes: True = win, False = loss
    wins = rng.random((M, T)) < p

    # Returns R_t in {+b, -a}
    R = np.where(wins, b, -a).astype(float)

    # Wealth path computed via log-wealth for numerical stability:
    # log W_T = sum log(1 + f R_t)
    increments = 1.0 + f * R

    # This should be strictly > 0 by admissibility
    if np.any(increments <= 0):
        # If it happens, treat as ruin for that path (should not if f is admissible)
        # But we’ll handle safely.
        # Set those to tiny positive to avoid nan and mark as ruined.
        ruined = np.any(increments <= 0, axis=1)
        increments = np.maximum(increments, 1e-300)
    else:
        ruined = np.zeros(M, dtype=bool)

    log_incr = np.log(increments)
    logW = np.cumsum(log_incr, axis=1)  # log wealth over time (starting from 0)

    logW_T = logW[:, -1]
    growth = logW_T / T  # per-round log growth

    # Max drawdown in wealth terms:
    # wealth_t = exp(logW_t), peak_t = max_{s<=t} wealth_s, DD_t = 1 - wealth_t/peak_t
    # Since exp is monotone, compute in log space:
    # peak_log = max_{s<=t} logW_s, drawdown_t = 1 - exp(logW_t - peak_log)
    peak_log = np.maximum.accumulate(logW, axis=1)
    dd = 1.0 - np.exp(logW - peak_log)
    mdd = np.max(dd, axis=1)

    # End below start: W_T < 1  <=> logW_T < 0
    end_below_1 = (logW_T < 0) | ruined

    return {
        "f": f,
        "mean_growth": float(np.mean(growth)),
        "median_growth": float(np.median(growth)),
        "p_end_below_1": float(np.mean(end_below_1)),
        "mean_mdd": float(np.mean(mdd)),
        "median_mdd": float(np.median(mdd)),
    }


# ----------------------------
# Kelly formulas (single bet)
# ----------------------------
def kelly_fraction(p: float, b: float, a: float) -> float:
    """
    Exact Kelly for two-point return:
      R = +b w.p. p, R = -a w.p. 1-p

    Maximize E[log(1+fR)] gives:
      f* = (p*b - (1-p)*a) / (a*b)

    Note: must also satisfy admissibility f < 1/a.
    """
    return (p * b - (1.0 - p) * a) / (a * b)


# ----------------------------
# Pretty printing
# ----------------------------
def fmt_pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


def print_table(rows: list[dict], title: str):
    print("\n" + title)
    print("-" * len(title))
    header = (
        f"{'Strategy':<14}"
        f"{'f':>10}"
        f"{'mean g':>12}"
        f"{'median g':>12}"
        f"{'P(W_T<1)':>12}"
        f"{'mean MDD':>12}"
        f"{'med MDD':>12}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['label']:<14}"
            f"{r['f']:>10.4f}"
            f"{r['mean_growth']:>12.4f}"
            f"{r['median_growth']:>12.4f}"
            f"{fmt_pct(r['p_end_below_1']):>12}"
            f"{fmt_pct(r['mean_mdd']):>12}"
            f"{fmt_pct(r['median_mdd']):>12}"
        )
    print()


def main():
    # ----------------------------
    # Default experiment settings
    # ----------------------------
    # Match the style you mentioned:
    # p=0.55, b=1, a=1, T=10000, M=5000
    p = 0.55
    b = 1.0
    a = 1.0
    T = 10_000
    M = 5_000
    seed = 42

    f_star = kelly_fraction(p, b, a)

    # Clamp f_star to admissible range if needed (should already be if edge is positive)
    # Also disallow negative f (if the bet has negative edge, full Kelly would be <=0).
    if f_star <= 0:
        raise ValueError(
            f"Kelly fraction f*={f_star:.6f} is <= 0 (no positive edge). "
            "Choose parameters with positive edge or interpret as 'don't bet'."
        )

    # Ensure strict admissibility
    f_star = min(f_star, (1.0 / a) - 1e-12)

    strategies = [
        ("Full Kelly", 1.0 * f_star),
        ("Half Kelly", 0.5 * f_star),
        ("Quarter Kelly", 0.25 * f_star),
    ]

    print("Kelly Criterion Monte Carlo (no plots)")
    print("-------------------------------------")
    print(f"Params: p={p}, b={b}, a={a}, T={T}, M={M}, seed={seed}")
    print(f"Kelly f* = (p*b - (1-p)*a)/(a*b) = {f_star:.6f}")
    print(f"Admissibility requires f < 1/a = {1.0/a:.6f}\n")

    rows = []
    for i, (label, f) in enumerate(strategies):
        stats = simulate_paths(p=p, b=b, a=a, f=f, T=T, M=M, seed=seed + 100 * i)
        stats["label"] = label
        rows.append(stats)

    print_table(rows, title="Results (log-growth per round and drawdown)")

    # Quick “headline” summary like in your write-up:
    print("Headline summary (for write-up)")
    print("------------------------------")
    for r in rows:
        print(
            f"{r['label']}: mean log-growth={r['mean_growth']:.4f}, "
            f"mean max drawdown={fmt_pct(r['mean_mdd'])}"
        )


if __name__ == "__main__":
    main()
