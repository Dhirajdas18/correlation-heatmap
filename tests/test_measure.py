import numpy as np
import pandas as pd
import pytest

from ch.measure import (
    bootstrap_avg_corr,
    kendall_to_pearson,
    resampled_returns,
    spearman_to_pearson,
    split_day,
    upper,
)


def test_upper_takes_each_pair_once():
    c = pd.DataFrame([[1, 0.2, 0.3], [0.2, 1, 0.4], [0.3, 0.4, 1]])
    assert sorted(upper(c)) == [0.2, 0.3, 0.4]


def test_rank_correlations_convert_back_for_normal_data():
    rng = np.random.default_rng(0)
    x = rng.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], 200_000)
    d = pd.DataFrame(x)
    assert spearman_to_pearson(d.corr("spearman").iloc[0, 1]) == pytest.approx(0.3, abs=0.01)
    tau = d.iloc[:20_000].corr("kendall").iloc[0, 1]
    assert kendall_to_pearson(tau) == pytest.approx(0.3, abs=0.02)


def test_resampled_returns_compound_daily_returns():
    idx = pd.bdate_range("2021-01-01", "2021-03-31")
    R = pd.DataFrame({"A": 0.01}, index=idx)
    m = resampled_returns(R, "ME")
    feb = idx[(idx.month == 2)]
    assert m.loc["2021-02-28", "A"] == pytest.approx(1.01 ** len(feb) - 1)
    assert len(m) == 2  # the first month has no starting price


def test_split_day_reconstructs_close_to_close_and_drops_bad_opens():
    idx = pd.bdate_range("2021-01-01", periods=4)
    close = pd.DataFrame({"A": [100.0, 102.0, 101.0, 103.0]}, index=idx)
    open_ = pd.DataFrame({"A": [100.0, 101.0, 300.0, 102.0]}, index=idx)  # day 3's open is bad
    table = pd.concat({"open": open_, "close": close, "adj_close": close}, axis=1)
    table.columns.names = ["field", "symbol"]
    on, intra = split_day(table, ["A"])
    assert (1 + on.iloc[1, 0]) * (1 + intra.iloc[1, 0]) == pytest.approx(102 / 100)
    assert np.isnan(on.iloc[2, 0]) and np.isnan(intra.iloc[2, 0])
    assert on.iloc[3, 0] == pytest.approx(102 / 101 - 1)


def test_bootstrap_avg_corr_is_centred_on_the_sample_value():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", "2020-12-31")
    m = rng.normal(0, 0.01, len(idx))
    R = pd.DataFrame({k: m + rng.normal(0, 0.01, len(idx)) for k in "ABCD"}, index=idx)
    point, lo, hi = bootstrap_avg_corr(R, n_boot=200, seed=0)
    assert lo < point < hi
    assert point == pytest.approx(upper(R.corr()).mean())


def test_split_day_masks_days_in_the_fixes_file(tmp_path):
    idx = pd.bdate_range("2021-01-01", periods=4)
    close = pd.DataFrame({"A": [100.0, 102.0, 51.0, 52.0]}, index=idx)  # unadjusted 1:1 bonus
    table = pd.concat({"open": close.shift(1).fillna(100.0) * [1, 1, 0.5, 1][0], "close": close,
                       "adj_close": close}, axis=1)
    table.columns.names = ["field", "symbol"]
    fixes = tmp_path / "f.csv"
    fixes.write_text("symbol,date,action,reason\nA,2021-01-05,zero,bonus\n")
    on, intra = split_day(table, ["A"], fixes=fixes)
    assert np.isnan(on.loc["2021-01-05", "A"]) and np.isnan(intra.loc["2021-01-05", "A"])
    assert np.isfinite(on.loc["2021-01-06", "A"])


def test_resampled_returns_drop_an_unfinished_last_week():
    idx = pd.bdate_range("2024-01-01", "2024-01-17")  # ends on a Wednesday
    R = pd.DataFrame({"A": 0.01}, index=idx)
    w = resampled_returns(R, "W-FRI")
    assert w.index[-1] == pd.Timestamp("2024-01-12")
