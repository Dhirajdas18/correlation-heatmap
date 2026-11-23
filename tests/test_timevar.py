import numpy as np
import pandas as pd
import pytest

from ch.timevar import (
    avg_offdiag,
    dcc_corr,
    dcc_fit,
    ewma_corr,
    exceedance_corr,
    qlike,
    standardise,
)


def _two_regimes(seed=0, t=3000):
    """Correlation 0.2 in the first half, 0.7 in the second; unit variance."""
    rng = np.random.default_rng(seed)
    out = []
    for rho, n in ((0.2, t // 2), (0.7, t - t // 2)):
        out.append(rng.multivariate_normal([0, 0, 0], [[1, rho, rho], [rho, 1, rho],
                                                       [rho, rho, 1]], n))
    return pd.DataFrame(np.vstack(out), index=pd.bdate_range("2010-01-01", periods=t))


def test_avg_offdiag():
    c = np.array([[1, 0.2, 0.4], [0.2, 1, 0.6], [0.4, 0.6, 1]])
    assert avg_offdiag(c) == pytest.approx(0.4)
    stack = np.stack([c, c * 0 + np.eye(3)])
    assert avg_offdiag(stack) == pytest.approx([0.4, 0.0])


def test_ewma_corr_tracks_the_regime_and_uses_past_days_only():
    Z = _two_regimes()
    c = ewma_corr(Z.to_numpy(), lam=0.97)
    assert avg_offdiag(c[1400]) == pytest.approx(0.2, abs=0.12)
    assert avg_offdiag(c[2900]) == pytest.approx(0.7, abs=0.12)
    x = Z.to_numpy().copy()
    x[2000:] *= -1  # change the future
    c2 = ewma_corr(x, lam=0.97)
    assert np.allclose(c2[2000], c[2000])  # day t's forecast ignores day t onward


def test_dcc_fit_finds_persistence_and_tracks_the_regime():
    Z = _two_regimes(seed=1)
    a, b = dcc_fit(Z.to_numpy())
    assert 0 < a < 0.2 and 0.7 < a + b < 1
    c = dcc_corr(Z.to_numpy(), a, b)
    assert avg_offdiag(c[2900]) > avg_offdiag(c[1400]) + 0.25


def test_standardise_divides_by_the_previous_days_ewma_vol():
    r = pd.DataFrame({"A": np.r_[np.full(40, 0.01), 0.05]},
                     index=pd.bdate_range("2020-01-01", periods=41))
    z = standardise(r, lam=0.94, seed=30)
    assert z["A"].iloc[-1] == pytest.approx(5.0)
    assert z["A"].iloc[:30].isna().all()


def test_exceedance_corr_is_symmetric_for_normal_data():
    rng = np.random.default_rng(2)
    x = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 400_000)
    down = exceedance_corr(x[:, 0], x[:, 1], -1.0)
    up = exceedance_corr(x[:, 0], x[:, 1], 1.0)
    assert down == pytest.approx(up, abs=0.02)
    assert down < 0.5  # conditioning on the tail lowers it, even for normal data


def test_qlike_is_smallest_at_the_true_variance():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 2, 100_000)
    losses = {v: qlike(np.full(len(r), v), r) for v in (2.0, 4.0, 8.0)}
    assert min(losses, key=losses.get) == 4.0
