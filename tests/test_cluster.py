import numpy as np
import pandas as pd
import pytest

from ch.cluster import adjusted_rand, cluster, corr_matrix, distance, sector_fit


def _blocks(seed=0, n=600):
    """Three groups of four stocks: each stock = market + its group's factor + noise."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, n)
    cols = {}
    for g in "ABC":
        factor = rng.normal(0, 0.01, n)
        for i in range(4):
            cols[f"{g}{i}"] = market + factor + rng.normal(0, 0.006, n)
    return pd.DataFrame(cols, index=pd.bdate_range("2020-01-01", periods=n))


def test_corr_matrix_uses_only_days_every_stock_traded():
    R = _blocks()
    R.iloc[:100, 0] = np.nan
    c = corr_matrix(R, list(R.columns))
    assert c.loc["A0", "A1"] == pytest.approx(R.iloc[100:][["A0", "A1"]].corr().iloc[0, 1])
    assert np.allclose(np.diag(c), 1)


def test_distance_is_zero_for_identical_and_two_for_opposite():
    c = pd.DataFrame([[1, 1, -1], [1, 1, -1], [-1, -1, 1]], dtype=float)
    d = distance(c)
    assert d.iloc[0, 1] == pytest.approx(0) and d.iloc[0, 2] == pytest.approx(2)


def test_cluster_recovers_the_groups_and_orders_them_together():
    R = _blocks()
    res = cluster(corr_matrix(R, list(R.columns)), k=3)
    truth = [c[0] for c in R.columns]
    assert adjusted_rand(res["labels"].to_numpy(), truth) == pytest.approx(1.0)
    order = [s[0] for s in res["order"]]
    assert all(order[i] == order[i + 1] for i in (0, 1, 2, 4, 5, 6, 8, 9, 10))


def test_adjusted_rand_known_values():
    assert adjusted_rand([0, 0, 1, 1], [5, 5, 7, 7]) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 5, 5000), rng.integers(0, 5, 5000)
    assert abs(adjusted_rand(a, b)) < 0.01  # unrelated labels score about 0


def test_sector_fit_flags_a_stock_closer_to_another_sector():
    R = _blocks()
    sectors = {c: c[0] for c in R.columns}
    sectors["A0"] = "B"  # mislabelled: behaves like A
    fit = sector_fit(corr_matrix(R, list(R.columns)), sectors)
    assert fit.loc["A0", "closest"] == "A" and not fit.loc["A0", "fits"]
    assert fit.loc["B1", "fits"]


def test_remove_market_takes_out_each_stocks_beta_times_the_index():
    from ch.cluster import remove_market

    R = _blocks()
    R["NIFTY"] = R.mean(axis=1)
    res = remove_market(R, [c for c in R.columns if c != "NIFTY"], "NIFTY")
    m = R["NIFTY"]
    beta = R["A0"].cov(m) / m.var()
    assert res["A0"].to_numpy() == pytest.approx((R["A0"] - beta * m).to_numpy())
    assert abs(res["A0"].corr(m)) < 1e-10  # nothing left that moves with the index
