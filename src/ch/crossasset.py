"""NIFTY against the US market, gold and the rupee, and what crosses time zones.

India's market closes at 15:30 IST; the US cash session runs about 19:00-01:30 IST, so a US
session dated d (on or after the date of NSE's previous close) happens entirely inside NIFTY's
overnight gap into the next NSE day. `us_during_gap` compounds those sessions for each NSE day.
Cross-asset correlation uses Friday-to-Friday returns, each series at its own last close of the
week, so the different holiday calendars don't put fake zero days into either series.
"""

import numpy as np
import pandas as pd


def us_during_gap(nse_dates, us_ret: pd.Series) -> pd.Series:
    """For each NSE date t (after the first), the compounded US return over sessions dated from
    the previous NSE date up to the day before t."""
    nse_dates = pd.DatetimeIndex(nse_dates)
    growth = (1 + us_ret.dropna()).cumprod()
    out = pd.Series(np.nan, index=nse_dates)
    for prev, t in zip(nse_dates[:-1], nse_dates[1:], strict=True):
        inside = growth[(growth.index >= prev) & (growth.index < t)]
        if inside.empty:
            out[t] = 0.0
            continue
        before = growth[growth.index < prev]
        start = before.iloc[-1] if len(before) else 1.0
        out[t] = inside.iloc[-1] / start - 1
    return out


def weekly_returns(series: dict) -> pd.DataFrame:
    """Friday-to-Friday returns; a final week the data hasn't reached the end of is dropped."""
    out = pd.DataFrame({k: v.dropna().resample("W-FRI").last().pct_change()
                        for k, v in series.items()}).dropna()
    last = min(v.dropna().index[-1] for v in series.values())
    return out[out.index <= last]


def regress(y, x) -> tuple[float, float, float]:
    """Slope of y on x, its t-statistic, and the correlation."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    xc, yc = x - x.mean(), y - y.mean()
    b = (xc @ yc) / (xc @ xc)
    resid = yc - b * xc
    se = np.sqrt(resid @ resid / (len(x) - 2) / (xc @ xc))
    return float(b), float(b / se), float(np.corrcoef(x, y)[0, 1])


def rolling_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window).corr(y)
