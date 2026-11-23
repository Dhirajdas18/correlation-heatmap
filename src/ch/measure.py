"""How the way correlation is measured changes the number.

- Pearson uses the values; Spearman and Kendall use only ranks, so a few huge days count for less.
  For normally distributed data they convert back to Pearson's scale: ρ = 2·sin(π·ρs/6) and
  ρ = sin(π·τ/2).
- Horizon: correlation of daily, weekly, monthly or quarterly returns (non-overlapping).
- Overnight (previous close to open) against intraday (open to close): the two multiply back to the
  close-to-close return. Yahoo's opens are not always right (they jump on demerger days), so a
  stock-day whose overnight and intraday moves are both bigger than 10% and opposite is dropped.
"""

import numpy as np
import pandas as pd


def upper(c: pd.DataFrame) -> np.ndarray:
    a = np.asarray(c, dtype=float)
    return a[np.triu_indices(len(a), 1)]


def spearman_to_pearson(rs):
    return 2 * np.sin(np.pi * np.asarray(rs) / 6)


def kendall_to_pearson(tau):
    return np.sin(np.pi * np.asarray(tau) / 2)


def resampled_returns(R: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Non-overlapping returns over each period (e.g. "W-FRI", "ME", "QE") from daily returns."""
    px = (1 + R.fillna(0)).cumprod().where(R.notna())
    out = px.resample(freq).last().pct_change(fill_method=None).iloc[1:]
    if len(out) and out.index[-1] > R.index[-1]:  # the last period hasn't finished
        out = out.iloc[:-1]
    return out


def split_day(table: pd.DataFrame, symbols: list[str], max_reversal: float = 0.10,
              fixes=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overnight and intraday returns. Opens are put on the adjusted scale with each day's
    adj_close / close factor. Stock-days repaired in the fixes file (unadjusted corporate actions,
    bad prints) are dropped, and so are days before a `start` row."""
    from ch.data import FIXES

    table_fixes = pd.read_csv(fixes or FIXES, parse_dates=["date"])
    adj, close = table["adj_close"][symbols], table["close"][symbols]
    open_adj = table["open"][symbols] * adj / close
    overnight = open_adj / adj.shift(1) - 1
    intraday = adj / open_adj - 1
    bad = ((overnight.abs() > max_reversal) & (intraday.abs() > max_reversal)
           & (np.sign(overnight) != np.sign(intraday)))
    for row in table_fixes.itertuples():
        if row.symbol not in symbols:
            continue
        if row.action == "start":
            bad.loc[bad.index < row.date, row.symbol] = True
        elif row.date in bad.index:
            bad.loc[row.date, row.symbol] = True
            later = bad.index[bad.index > row.date]
            if row.action == "blank" and len(later):  # the day after a blanked price too
                bad.loc[later[0], row.symbol] = True
    return overnight.mask(bad), intraday.mask(bad)


def bootstrap_avg_corr(R: pd.DataFrame, n_boot: int = 500, seed=None) -> tuple:
    """Average pairwise correlation and its 90% range, resampling whole calendar months (so
    each month's returns stay together)."""
    R = R.dropna()
    point = float(upper(R.corr()).mean())
    months = R.index.to_period("M")
    groups = [np.flatnonzero(months == m) for m in months.unique()]
    rng = np.random.default_rng(seed)
    x = R.to_numpy()
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append(upper(np.corrcoef(x[idx], rowvar=False)).mean())
    return point, float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))
