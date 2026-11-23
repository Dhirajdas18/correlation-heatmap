"""Diversification: how many independent bets, and hierarchical risk parity.

- Participation ratio of the correlation matrix, (Σλ)² / Σλ²: N for unrelated stocks, 1 when all
  move as one.
- Diversification ratio of a portfolio, Σ wᵢσᵢ / σ_p (Choueifaty); its square is the effective
  number of independent bets (N for N unrelated, equally weighted, equally risky stocks).
- Hierarchical risk parity (López de Prado 2016): a single-linkage tree on √(2(1 − ρ)); going
  down from the root, each branch's weight is split between its two sub-branches in inverse
  proportion to their variance (inverse-variance weights inside each). This splits along the
  tree's branches rather than the original's halves of the ordered list, which can cut a cluster
  in two. No matrix inversion.
- Costs: Zerodha equity delivery (as ef/04): STT 0.1% both sides, stamp 0.015% on buys, NSE and
  SEBI charges with GST, ₹15.93 DP charge per stock sold, 0.05% slippage per side (assumed).
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

RATES = {"stt": 0.001, "stamp_buy": 0.00015, "exchange": 0.0000322, "sebi": 0.000001,
         "gst": 0.18, "dp_per_sell": 15.93, "slippage": 0.0005}


def participation_ratio(corr) -> float:
    ev = np.linalg.eigvalsh(np.asarray(corr, dtype=float))
    return float(ev.sum() ** 2 / (ev**2).sum())


def diversification_ratio(w, cov) -> float:
    w, cov = np.asarray(w, float), np.asarray(cov, float)
    return float(w @ np.sqrt(np.diag(cov)) / np.sqrt(w @ cov @ w))


def inverse_vol(cov) -> np.ndarray:
    iv = 1 / np.sqrt(np.diag(cov))
    return iv / iv.sum()


def min_var_long_only(cov) -> np.ndarray:
    cov = np.asarray(cov, float)
    n = len(cov)
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), jac=lambda w: 2 * cov @ w,
                   bounds=[(0, 1)] * n, method="SLSQP",
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-15, "maxiter": 500})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def _cluster_var(cov, idx):
    sub = cov[np.ix_(idx, idx)]
    w = 1 / np.diag(sub)
    w /= w.sum()
    return float(w @ sub @ w)


def hrp_weights(cov, corr) -> np.ndarray:
    cov, corr = np.asarray(cov, float), np.asarray(corr, float)
    d = np.sqrt(np.clip(2 * (1 - corr), 0, None))
    np.fill_diagonal(d, 0)
    Z = linkage(squareform(d, checks=False), "single")
    n = len(cov)
    members = {i: [i] for i in range(n)}
    for k, (a, b, _, _) in enumerate(Z):
        members[n + k] = members[int(a)] + members[int(b)]
    w = np.ones(n)
    stack = [2 * n - 2]  # the root; split along the tree's own branches
    while stack:
        node = stack.pop()
        if node < n:
            continue
        a, b = (int(v) for v in Z[node - n, :2])
        left, right = members[a], members[b]
        vl, vr = _cluster_var(cov, left), _cluster_var(cov, right)
        alpha = 1 - vl / (vl + vr)
        w[left] *= alpha
        w[right] *= 1 - alpha
        stack += [a, b]
    return w / w.sum()


def trade_cost(buy: float, sell: float, n_sells: int, capital: float,
               rates: dict = RATES) -> float:
    fees = (rates["exchange"] + rates["sebi"]) * (1 + rates["gst"])
    side = fees + rates["slippage"] + rates["stt"]
    return (buy * (side + rates["stamp_buy"]) + sell * side
            + n_sells * rates["dp_per_sell"] / capital)


def walk_forward(R: pd.DataFrame, weigh, lookback: int, start: str, capital: float,
                 costs: bool = True) -> dict:
    """Rebalance at each month's last trading day to `weigh(cov, corr)` from the previous
    `lookback` days (up to and including that day); hold through the next month, letting weights
    drift. Daily wealth from ₹1 at `start`; costs on the trades."""
    R = R.dropna()
    month = R.index.to_period("M")
    ends = R.index[np.append(month[1:] != month[:-1], True)]
    x = R.to_numpy()
    first = int(np.searchsorted(R.index, pd.Timestamp(start)))
    w = np.zeros(R.shape[1])
    wealth = [1.0]
    weights, paid, turnover = {}, 0.0, 0.0
    end_set = set(R.index.get_indexer(ends))
    for t in range(first, len(R)):
        if t > first:
            g = 1 + w @ x[t]
            wealth.append(wealth[-1] * g)
            w = w * (1 + x[t]) / g
        if t in end_set or t == first:
            past = R.iloc[max(0, t - lookback + 1):t + 1]
            target = weigh(past.cov().to_numpy(), past.corr().to_numpy())
            trade = target - w
            if costs:
                c = trade_cost(trade.clip(min=0).sum(), (-trade).clip(min=0).sum(),
                               int((trade < -1e-9).sum()), wealth[-1] * capital)
                wealth[-1] *= 1 - c
                paid += c
            turnover += np.abs(trade).sum() / 2
            w = target
            weights[R.index[t]] = target
    idx = R.index[first:]
    years = (idx[-1] - idx[0]).days / 365.25
    return {"wealth": pd.Series(wealth, index=idx), "costs": paid / years,
            "turnover": turnover / years,
            "weights": pd.DataFrame(weights, index=R.columns).T.reindex(idx).ffill()}
