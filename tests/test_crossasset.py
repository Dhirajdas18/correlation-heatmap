import numpy as np
import pandas as pd
import pytest

from ch.crossasset import regress, rolling_corr, us_during_gap, weekly_returns


def test_us_during_gap_compounds_the_us_sessions_between_nse_closes():
    nse = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-04"])  # NSE shut on 2 Jan
    us_ret = pd.Series([0.01, 0.02, -0.01, 0.03],
                       index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03",
                                             "2024-01-04"]))
    g = us_during_gap(nse, us_ret)
    assert np.isnan(g.iloc[0])  # no previous NSE close
    # gap into 3 Jan: US sessions of 1 and 2 Jan, both after NSE's close on 1 Jan
    assert g.loc["2024-01-03"] == pytest.approx(1.01 * 1.02 - 1)
    assert g.loc["2024-01-04"] == pytest.approx(-0.01)  # the 3 Jan US session


def test_weekly_returns_use_each_series_own_last_close():
    idx_a = pd.bdate_range("2024-01-01", "2024-01-31")
    a = pd.Series(np.linspace(100, 110, len(idx_a)), index=idx_a)
    b = a.drop(a.index[a.index.day_of_week == 4])  # b never trades on Fridays
    w = weekly_returns({"a": a, "b": b})
    wa = a.resample("W-FRI").last().pct_change()
    assert w["a"].to_numpy() == pytest.approx(wa.dropna().to_numpy()[: len(w)])


def test_regress_and_rolling_corr():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    y = 0.5 * x + rng.normal(scale=0.5, size=5000)
    b, t, r = regress(y, x)
    assert b == pytest.approx(0.5, abs=0.02) and t > 30
    assert r == pytest.approx(np.corrcoef(x, y)[0, 1])
    idx = pd.date_range("2020-01-03", periods=200, freq="W-FRI")
    df = pd.DataFrame({"x": x[:200], "y": y[:200]}, index=idx)
    rc = rolling_corr(df["x"], df["y"], 52)
    assert rc.iloc[-1] == pytest.approx(df.iloc[-52:].corr().iloc[0, 1])
