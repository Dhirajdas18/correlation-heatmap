"""Signal against noise in a correlation matrix (random matrix theory).

For N series of T independent observations with no true correlation, the eigenvalues of the sample
correlation matrix fall between λ− = (1 − √q)² and λ+ = (1 + √q)², q = N/T, with the
Marchenko-Pastur density. Eigenvalues above λ+ carry real common movement (the first is the
market); those inside the band are indistinguishable from noise. The eigenvalues add up to N, so
λ1/N is the share of all movement the market factor explains.
"""

import numpy as np
import pandas as pd


def eigen(c) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (largest first) and eigenvectors (columns, same order)."""
    ev, vec = np.linalg.eigh(np.asarray(c, dtype=float))
    return ev[::-1], vec[:, ::-1]


def mp_bounds(n: int, t: int) -> tuple[float, float]:
    q = n / t
    return (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2


def mp_density(x, q: float):
    lo, hi = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
    x = np.asarray(x, dtype=float)
    inside = (x > lo) & (x < hi)
    out = np.zeros_like(x)
    out[inside] = np.sqrt((hi - x[inside]) * (x[inside] - lo)) / (2 * np.pi * q * x[inside])
    return out


def signal_count(c, t: int) -> int:
    ev, _ = eigen(c)
    return int((ev > mp_bounds(len(ev), t)[1]).sum())


def shuffle(R: pd.DataFrame, seed=None) -> pd.DataFrame:
    """Each column's values in a random order of its own: same distribution, no co-movement."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({k: rng.permutation(R[k].to_numpy()) for k in R}, index=R.index)


def clip(c, t: int) -> np.ndarray:
    """Keep eigenvalues above λ+, replace the rest by their average (the trace is unchanged),
    rebuild and rescale to a unit diagonal (Laloux et al. 1999)."""
    ev, vec = eigen(c)
    noise = ev <= mp_bounds(len(ev), t)[1]
    ev = ev.copy()
    if noise.any():
        ev[noise] = ev[noise].mean()
    m = vec @ np.diag(ev) @ vec.T
    d = np.sqrt(np.diag(m))
    return m / np.outer(d, d)


def market_share(R: pd.DataFrame) -> float:
    """Share of all movement explained by the first factor: λ1 / N of the correlation matrix."""
    ev, _ = eigen(R.dropna().corr())
    return float(ev[0] / len(ev))
