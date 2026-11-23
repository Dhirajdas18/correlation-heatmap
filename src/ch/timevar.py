"""Correlation through time: EWMA, DCC, exceedance correlation and forecast loss.

All forecasts for day t use returns up to day t − 1.
- standardise: each return divided by the previous day's EWMA volatility (λ 0.94), so what is left
  has roughly unit variance and the correlation models only need to track correlation.
- EWMA correlation: Q_t = λ·Q_{t−1} + (1 − λ)·z_{t−1}z_{t−1}ᵀ, scaled to a unit diagonal.
- DCC(1,1) (Engle 2002): Q_t = (1 − a − b)·Q̄ + a·z_{t−1}z_{t−1}ᵀ + b·Q_{t−1}, scaled to a unit
  diagonal; a and b by composite likelihood (summed over pairs), so it works for many stocks.
- Exceedance correlation (Longin and Solnik 2001): correlation on the days both standardised
  returns are below −θ (or both above +θ).
- QLIKE: log σ̂² + r²/σ̂², lowest on average for the forecast that equals the true variance.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def avg_offdiag(c):
    c = np.asarray(c, dtype=float)
    n = c.shape[-1]
    return (c.sum(axis=(-1, -2)) - np.trace(c, axis1=-2, axis2=-1)) / (n * (n - 1))


def standardise(R: pd.DataFrame, lam: float = 0.94, seed: int = 30,
                with_vol: bool = False):
    """Returns divided by the previous day's EWMA volatility; with `with_vol`, also that
    volatility (the forecast each day was scaled by)."""
    x = R.to_numpy()
    out, vol = np.full(x.shape, np.nan), np.full(x.shape, np.nan)
    v = np.nanmean(x[:seed] ** 2, axis=0)
    for t in range(seed, len(x)):
        vol[t] = np.sqrt(v)
        out[t] = x[t] / vol[t]
        v = lam * v + (1 - lam) * np.nan_to_num(x[t] ** 2, nan=v)
    z = pd.DataFrame(out, index=R.index, columns=R.columns)
    if with_vol:
        return z, pd.DataFrame(vol, index=R.index, columns=R.columns)
    return z


def _scale(q):
    d = np.sqrt(np.diagonal(q, axis1=-2, axis2=-1))
    return q / (d[..., :, None] * d[..., None, :])


def ewma_corr(z: np.ndarray, lam: float = 0.94, seed: int = 60) -> np.ndarray:
    """Correlation forecast for each day from the standardised returns before it."""
    t, n = z.shape
    out = np.full((t, n, n), np.nan)
    q = np.cov(z[:seed].T)
    for i in range(seed, t):
        out[i] = _scale(q)
        q = lam * q + (1 - lam) * np.outer(z[i], z[i])
    return out


def dcc_corr(z: np.ndarray, a: float, b: float, qbar="expanding", seed: int = 60) -> np.ndarray:
    """DCC correlation forecasts. The long-run target Q̄ is "expanding" (the average of z zᵀ over
    the days before each forecast, so nothing from the day or after is used), or a fixed matrix
    estimated elsewhere (for example on a period before the test). Days before `seed` are NaN."""
    t, n = z.shape
    out = np.full((t, n, n), np.nan)
    acc = np.einsum("ti,tj->ij", z[:seed], z[:seed])
    fixed = None if isinstance(qbar, str) else np.asarray(qbar, float)
    q = (acc / seed) if fixed is None else fixed.copy()
    for i in range(seed, t):
        out[i] = q
        target = acc / i if fixed is None else fixed
        acc += np.outer(z[i], z[i])
        q = (1 - a - b) * target + a * np.outer(z[i], z[i]) + b * q
    return _scale(out)


def dcc_fit(z: np.ndarray, max_pairs: int = 300, seed: int = 0) -> tuple[float, float]:
    """a, b maximising the sum of bivariate normal log-likelihoods over (up to `max_pairs`)
    pairs, with the same recursion run pair by pair (vectorised)."""
    n = z.shape[1]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng = np.random.default_rng(seed)
    if len(pairs) > max_pairs:
        pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs, replace=False)]
    x, y = z[:, [p[0] for p in pairs]], z[:, [p[1] for p in pairs]]
    sxx, syy, sxy = (x * x).mean(0), (y * y).mean(0), (x * y).mean(0)

    def nll(params):
        a, b = params
        if a < 0 or b < 0 or a + b >= 0.999:
            return 1e12
        qxx, qyy, qxy = sxx.copy(), syy.copy(), sxy.copy()
        total = 0.0
        for i in range(len(z)):
            rho = qxy / np.sqrt(qxx * qyy)
            det = 1 - rho**2
            u, v = x[i], y[i]  # already standardised; only the correlation is modelled
            total += np.sum(np.log(det) + (u * u - 2 * rho * u * v + v * v) / det)
            qxx = (1 - a - b) * sxx + a * x[i] ** 2 + b * qxx
            qyy = (1 - a - b) * syy + a * y[i] ** 2 + b * qyy
            qxy = (1 - a - b) * sxy + a * x[i] * y[i] + b * qxy
        return 0.5 * total

    res = minimize(nll, [0.02, 0.95], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 200})
    return float(res.x[0]), float(res.x[1])


def exceedance_corr(x, y, theta: float) -> float:
    x, y = np.asarray(x), np.asarray(y)
    m = (x < theta) & (y < theta) if theta < 0 else (x > theta) & (y > theta)
    if m.sum() < 10:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def qlike(var_forecast, r) -> float:
    var_forecast, r = np.asarray(var_forecast, float), np.asarray(r, float)
    return float(np.mean(np.log(var_forecast) + r**2 / var_forecast))
