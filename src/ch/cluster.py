"""Correlation matrix, distances and hierarchical clustering of stocks.

Distance between two stocks: d = √(2(1 − ρ)), 0 for identical moves, √2 for unrelated, 2 for
opposite (Mantegna 1999). Average linkage joins the two groups with the smallest average distance,
repeatedly; cutting the tree into k groups gives clusters, and its leaf order puts similar stocks
next to each other, which is what makes a heatmap readable.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform


def corr_matrix(R: pd.DataFrame, symbols: list[str], method: str = "pearson") -> pd.DataFrame:
    """Correlation over the days on which every one of `symbols` has a return."""
    return R[symbols].dropna().corr(method=method)


def distance(corr: pd.DataFrame) -> pd.DataFrame:
    return np.sqrt(np.clip(2 * (1 - corr), 0, None))


def cluster(corr: pd.DataFrame, k: int, method: str = "average") -> dict:
    d = distance(corr).to_numpy().copy()  # pandas hands back a read-only view
    np.fill_diagonal(d, 0)
    Z = linkage(squareform(d, checks=False), method=method, optimal_ordering=True)
    names = list(corr.index)
    return {"linkage": Z, "order": [names[i] for i in leaves_list(Z)],
            "labels": pd.Series(fcluster(Z, k, criterion="maxclust"), index=names)}


def adjusted_rand(a, b) -> float:
    """Adjusted Rand index of two labelings: 1 = same grouping, about 0 = unrelated."""
    from math import comb

    table = pd.crosstab(np.asarray(a), np.asarray(b)).to_numpy()
    n = int(table.sum())
    pairs = sum(comb(int(x), 2) for x in table.ravel())
    rows = sum(comb(int(x), 2) for x in table.sum(axis=1))
    cols = sum(comb(int(x), 2) for x in table.sum(axis=0))
    expected = rows * cols / comb(n, 2)
    top = (rows + cols) / 2
    return float((pairs - expected) / (top - expected)) if top != expected else 1.0


def sector_fit(corr: pd.DataFrame, sectors: dict) -> pd.DataFrame:
    """For each stock: average correlation with the rest of its own sector and with each other
    sector; `fits` is False when some other sector is closer (or it is alone in its sector)."""
    rows = {}
    for s in corr.index:
        others = corr.loc[s].drop(s)
        by = others.groupby(lambda x: sectors[x]).mean()
        own = by.get(sectors[s], np.nan)
        rest = by.drop(sectors[s], errors="ignore")
        closest = rest.idxmax()
        fits = bool(np.isfinite(own) and own >= rest.max())
        rows[s] = {"sector": sectors[s], "own": own, "closest": sectors[s] if fits else closest,
                   "closest corr": max(own, rest.max()) if np.isfinite(own) else rest.max(),
                   "fits": fits}
    out = pd.DataFrame(rows).T
    return out.astype({"own": float, "closest corr": float, "fits": bool})


def remove_market(R: pd.DataFrame, symbols: list[str], index: str) -> pd.DataFrame:
    """Each stock's return minus beta × the index's, over the days all of them have returns;
    beta from the same days. What is left moves with sector and company news, not the market."""
    W = R[symbols + [index]].dropna()
    m = W[index]
    beta = W[symbols].apply(lambda s: s.cov(m) / m.var())
    return W[symbols] - np.outer(m, beta)
