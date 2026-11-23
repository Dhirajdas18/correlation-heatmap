import numpy as np
import pandas as pd
import pytest

from ch.diversify import (
    diversification_ratio,
    hrp_weights,
    inverse_vol,
    min_var_long_only,
    participation_ratio,
    trade_cost,
    walk_forward,
)


def _blocks():
    """Two blocks: A (3 stocks, vol 1%, correlation 0.8 inside) and B (2 stocks, vol 2%, 0.8)."""
    vol = np.array([0.01, 0.01, 0.01, 0.02, 0.02])
    corr = np.eye(5)
    corr[:3, :3] = 0.8
    corr[3:, 3:] = 0.8
    np.fill_diagonal(corr, 1)
    return corr * np.outer(vol, vol), corr


def test_participation_ratio():
    assert participation_ratio(np.eye(10)) == pytest.approx(10)
    assert participation_ratio(np.ones((10, 10))) == pytest.approx(1)


def test_diversification_ratio_squared_counts_independent_bets():
    cov = np.eye(9) * 0.0004
    w = np.full(9, 1 / 9)
    assert diversification_ratio(w, cov) ** 2 == pytest.approx(9)
    assert diversification_ratio(np.array([1.0]), np.array([[0.0004]])) == pytest.approx(1)


def test_hrp_splits_between_clusters_by_inverse_variance():
    cov, corr = _blocks()
    w = hrp_weights(cov, corr)
    assert w.sum() == pytest.approx(1) and (w > 0).all()
    assert w[3] == pytest.approx(w[4])  # two identical stocks share their branch equally
    # cluster variances with inverse-variance weights inside each cluster
    va = 0.0001 * (1 + 2 * 0.8) / 3
    vb = 0.0004 * (1 + 0.8) / 2
    assert w[:3].sum() == pytest.approx(vb / (va + vb), rel=1e-6)


def test_inverse_vol_and_min_var():
    cov, _ = _blocks()
    iv = inverse_vol(cov)
    assert iv[0] / iv[3] == pytest.approx(2)
    mv = min_var_long_only(cov)
    assert mv.sum() == pytest.approx(1) and (mv >= -1e-9).all()
    for _ in range(20):  # no random long-only portfolio has lower variance
        r = np.random.default_rng(_).dirichlet(np.ones(5))
        assert mv @ cov @ mv <= r @ cov @ r + 1e-12


def test_trade_cost_matches_the_zerodha_delivery_model():
    c = trade_cost(buy=0.1, sell=0.1, n_sells=2, capital=1e6)
    fees = (0.0000322 + 0.000001) * 1.18
    side = fees + 0.0005 + 0.001
    assert c == pytest.approx(0.1 * (side + 0.00015) + 0.1 * side + 2 * 15.93 / 1e6)


def test_walk_forward_decides_from_past_data_and_charges_costs():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", "2018-12-31")
    R = pd.DataFrame(rng.normal(0.0004, 0.01, (len(idx), 4)), index=idx, columns=list("ABCD"))
    res = walk_forward(R, lambda cov, corr: np.full(4, 0.25), 250, "2016-01-01", 1e6)
    assert res["wealth"].index[0] >= pd.Timestamp("2016-01-01")
    assert res["costs"] > 0 and res["turnover"] > 0
    free = walk_forward(R, lambda cov, corr: np.full(4, 0.25), 250, "2016-01-01", 1e6,
                        costs=False)
    assert free["wealth"].iloc[-1] > res["wealth"].iloc[-1]
    R2 = R.copy()
    R2.loc["2017-06-01":] *= -1  # a different future
    a = walk_forward(R, inverse_vol_fn := (lambda cov, corr: inverse_vol(cov)), 250,
                     "2016-01-01", 1e6)["weights"]
    b = walk_forward(R2, inverse_vol_fn, 250, "2016-01-01", 1e6)["weights"]
    assert np.allclose(a.loc[:"2017-05-31"], b.loc[:"2017-05-31"])
