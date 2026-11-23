"""README findings, one function per topic: `ch report <topic>` writes charts to out/."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ch import plots

SHORT = {"Financial Services": "Financials", "Healthcare": "Healthcare",
         "Automobile and Auto Components": "Autos", "Metals & Mining": "Metals",
         "Information Technology": "IT", "Fast Moving Consumer Goods": "FMCG",
         "Oil Gas & Consumable Fuels": "Oil & gas", "Services": "Services",
         "Consumer Durables": "Durables", "Consumer Services": "Cons. services",
         "Construction Materials": "Cement", "Power": "Power", "Capital Goods": "Capital goods",
         "Telecommunication": "Telecom", "Construction": "Construction"}
K = 12  # clusters to cut the tree into


def _rupee_log(ax) -> None:
    """Log wealth axis labelled ₹1, ₹2, ₹5… instead of powers of ten."""
    from matplotlib.ticker import FuncFormatter, LogLocator

    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"₹{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def returns() -> pd.DataFrame:
    from ch.data import DATA_PATH, daily_returns

    return daily_returns(pd.read_parquet(DATA_PATH)["adj_close"])


def last_years(R: pd.DataFrame, years: int) -> pd.DataFrame:
    return R.loc[R.index[-1] - pd.DateOffset(years=years):]


def report_heatmap(out_dir: Path) -> dict:
    """The NIFTY 50 correlation matrix, ordered by sector and by clusters, with and without the
    market, over the last three years."""
    from scipy.cluster.hierarchy import dendrogram

    from ch.cluster import adjusted_rand, cluster, corr_matrix, remove_market, sector_fit
    from ch.data import SECTORS, SYMBOLS

    W = last_years(returns(), 3)[SYMBOLS + ["NIFTY"]].dropna()
    raw = corr_matrix(W, SYMBOLS)
    res = remove_market(W, SYMBOLS, "NIFTY").corr()
    sectors = [SECTORS[s] for s in SYMBOLS]
    cl_raw, cl_res = cluster(raw, K), cluster(res, K)
    by_sector = sorted(SYMBOLS, key=lambda s: (SECTORS[s], s))
    iu = np.triu_indices(len(SYMBOLS), 1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.6))
    cmap = plt.get_cmap("RdBu_r")
    for ax, mat, order, title in ((a1, raw, by_sector, "all moves, by NSE sector"),
                                  (a2, res, cl_res["order"], "market removed, by cluster")):
        m = mat.loc[order, order].to_numpy()
        im = ax.imshow(m, cmap=cmap, vmin=-0.8, vmax=0.8, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks(range(len(order)), order, fontsize=5)
        ax.set_title(title, fontsize=10, fontweight="normal")
        ax.grid(False)
    fig.colorbar(im, ax=[a1, a2], shrink=0.7, label="correlation of daily returns")
    fig.suptitle(f"Average correlation {raw.values[iu].mean():.2f}; with the market removed, "
                 f"stocks fall into sector-like blocks", fontweight="bold", y=0.93)
    plots.save(fig, out_dir, "heat_matrix")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(right=0.82)
    names = list(res.index)
    dn = dendrogram(cl_res["linkage"], labels=names, ax=ax, color_threshold=0,
                    above_threshold_color=plots.TEXT_2, leaf_font_size=6)
    palette = plots.SERIES + [plots.MUTED, plots.TEXT] + plots.BLUE_RAMP[3:]
    big = [s for s, _ in pd.Series(sectors).value_counts().items()]
    colour = {s: palette[i % len(palette)] for i, s in enumerate(big)}
    for lab in ax.get_xticklabels():
        lab.set_color(colour[SECTORS[lab.get_text()]])
    ax.set_ylabel("distance √(2(1 − correlation)), market removed")
    ax.grid(False)
    handles = [plt.Line2D([], [], marker="s", ls="", color=colour[s], label=SHORT[s])
               for s in big[:8]]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7,
              title="label colour", title_fontsize=7)
    ari = adjusted_rand(cl_res["labels"].to_numpy(), [SECTORS[n] for n in names])
    ari_raw = adjusted_rand(cl_raw["labels"].to_numpy(), [SECTORS[n] for n in raw.index])
    ax.set_title(f"Cut into {K} clusters, the market-removed tree matched NSE sectors at "
                 f"{ari:.2f} (raw: {ari_raw:.2f})")
    plots.save(fig, out_dir, "heat_tree")
    del dn

    fit = sector_fit(res, SECTORS)
    multi = pd.Series(SECTORS).value_counts()
    odd = fit[(~fit["fits"]) & (fit["sector"].map(multi) > 1)].sort_values("closest corr")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(odd))
    ax.barh(y + 0.2, odd["own"], 0.38, color=plots.MUTED, label="with the rest of its NSE sector")
    ax.barh(y - 0.2, odd["closest corr"], 0.38, color=plots.SERIES[0],
            label="with the closest other sector")
    ax.set_yticks(y, [f"{s} ({SHORT[odd.loc[s, 'sector']]} → {SHORT[odd.loc[s, 'closest']]})"
                      for s in odd.index], fontsize=8)
    ax.axvline(0, color=plots.TEXT_2, lw=0.8)
    ax.set_xlabel("average correlation, market removed, last 3 years")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"{len(odd)} of {int((pd.Series(SECTORS).map(multi) > 1).sum())} stocks moved "
                 "more with another sector than with their own")
    plots.save(fig, out_dir, "heat_misfits")

    half = W.index[len(W) // 2]
    halves = [cluster(remove_market(part, SYMBOLS, "NIFTY").corr(), K)["labels"]
              for part in (W.loc[:half], W.loc[half:])]
    pairs = raw.where(np.triu(np.ones(raw.shape, bool), 1)).stack().dropna().sort_values()
    clusters = cl_res["labels"].groupby(cl_res["labels"]).apply(lambda x: sorted(x.index))
    return {
        "window": f"{W.index[0]:%d %b %Y} to {W.index[-1]:%d %b %Y}, {len(W)} days",
        "average / min / max correlation": (round(raw.values[iu].mean(), 3),
                                            round(raw.values[iu].min(), 2),
                                            round(raw.values[iu].max(), 2)),
        "average correlation within NSE sectors": round(float(np.mean(
            [raw.iloc[i, j] for i, j in zip(*iu, strict=True)
             if sectors[i] == sectors[j]])), 3),
        "highest pairs": {f"{a}-{b}": round(v, 2) for (a, b), v in pairs.tail(6).items()},
        "lowest pairs": {f"{a}-{b}": round(v, 2) for (a, b), v in pairs.head(4).items()},
        "adjusted Rand vs sectors, raw / market removed (k=12)": (round(ari_raw, 3),
                                                                  round(ari, 3)),
        "raw cluster sizes": cl_raw["labels"].value_counts().tolist(),
        "market-removed clusters": clusters.to_dict(),
        "adjusted Rand, first vs second half": round(adjusted_rand(*halves), 3),
        "misfits (own, closest, closest corr)": {
            s: (round(r["own"], 2), r["closest"], round(r["closest corr"], 2))
            for s, r in odd.iterrows()},
    }


MEASURE_START = "2014-01-01"
START_ROLL = "2009-01-02"  # rolling windows start once the ETFs and most stocks trade


def report_measure(out_dir: Path) -> dict:
    """Pearson against rank correlation, daily against longer horizons, overnight against intraday,
    for the NIFTY 50 stocks with prices through 2014-2026."""
    from ch.data import DATA_PATH, SYMBOLS
    from ch.measure import (
        bootstrap_avg_corr,
        kendall_to_pearson,
        resampled_returns,
        spearman_to_pearson,
        split_day,
        upper,
    )

    table = pd.read_parquet(DATA_PATH)
    R = returns().loc[MEASURE_START:]
    syms = [s for s in SYMBOLS if R[s].notna().mean() > 0.99]
    W = R[syms].dropna()
    pear, spear, kend = (float(upper(W.corr(method=m)).mean())
                         for m in ("pearson", "spearman", "kendall"))
    extreme = W.mean(axis=1).abs().sort_values(ascending=False).index[:10]
    pear_ex = float(upper(W.drop(extreme).corr()).mean())
    vals = {"Pearson": pear, "Pearson without the 10\nbiggest market days": pear_ex,
            "Spearman (ranks),\non Pearson's scale": float(spearman_to_pearson(spear)),
            "Kendall (ranks),\non Pearson's scale": float(kendall_to_pearson(kend))}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.arange(len(vals))
    ax.bar(xs, list(vals.values()), 0.6, color=[plots.SERIES[0], plots.MUTED, plots.SERIES[2],
                                                plots.SERIES[3]])
    raw = [None, None, spear, kend]
    for i, (v, r_) in enumerate(zip(vals.values(), raw, strict=True)):
        ax.text(i, v + 0.004, f"{v:.3f}" + (f"\n(raw {r_:.3f})" if r_ is not None else ""),
                ha="center", fontsize=8)
    ax.set_xticks(xs, list(vals), fontsize=8)
    ax.set_ylim(0, 0.33)
    ax.set_ylabel("average correlation of daily returns")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Ten of {len(W):,} days lifted the average correlation from {pear_ex:.2f} to "
                 f"{pear:.2f}")
    plots.save(fig, out_dir, "meas_methods")

    horizons = {"daily": W, "weekly": resampled_returns(W, "W-FRI"),
                "monthly": resampled_returns(W, "ME"), "quarterly": resampled_returns(W, "QE")}
    boot = {k: bootstrap_avg_corr(v, 500, seed=1) for k, v in horizons.items()}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.arange(len(boot))
    pts = [b[0] for b in boot.values()]
    err = np.array([[b[0] - b[1], b[2] - b[0]] for b in boot.values()]).T
    ax.errorbar(xs, pts, yerr=err, fmt="o", ms=8, color=plots.SERIES[0], ecolor=plots.TEXT_2,
                capsize=5, lw=1.2)
    for i, (k, b) in enumerate(boot.items()):
        ax.text(i + 0.08, b[0], f"{b[0]:.3f}\n({len(horizons[k])} returns)", fontsize=8,
                va="center")
    ax.set_xticks(xs, list(boot))
    ax.set_xlim(-0.4, len(boot) - 0.3)
    ax.set_ylabel("average correlation (90% range)")
    ax.set_title(f"Longer horizons averaged higher ({pts[0]:.2f} daily, {pts[3]:.2f} quarterly), "
                 "but the ranges overlap")
    plots.save(fig, out_dir, "meas_horizon")

    on, intra = split_day(table, syms)
    on, intra = on.loc[MEASURE_START:], intra.loc[MEASURE_START:]
    dropped = int(on.isna().sum().sum() - R[syms].isna().sum().sum())
    years = sorted(set(on.index.year))
    by_year = {lab: [float(upper(x[x.index.year == y].corr()).mean()) for y in years]
               for lab, x in (("overnight", on), ("intraday", intra))}
    var_share = float((on.var() / (on.var() + intra.var())).mean())
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for (lab, v), c in zip(by_year.items(), (plots.SERIES[1], plots.SERIES[0]), strict=True):
        span = "previous close to open" if lab == "overnight" else "open to close"
        ax.plot(years, v, "o-", color=c, lw=2, label=f"{lab}: {span}")
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("average correlation between stocks")
    ax.legend(loc="upper left", fontsize=8)
    o_all, i_all = (float(upper(x.corr()).mean()) for x in (on, intra))
    ax.set_title(f"Overnight gaps moved together twice as much as daytime moves: {o_all:.2f} "
                 f"against {i_all:.2f}")
    plots.save(fig, out_dir, "meas_overnight")

    return {
        "stocks, days": (len(syms), len(W), f"{W.index[0]:%d %b %Y} to {W.index[-1]:%d %b %Y}"),
        "average correlation: Pearson, without 10 extreme days, Spearman, Kendall": (
            round(pear, 3), round(pear_ex, 3), round(spear, 3), round(kend, 3)),
        "on Pearson's scale: Spearman, Kendall": (round(float(spearman_to_pearson(spear)), 3),
                                                  round(float(kendall_to_pearson(kend)), 3)),
        "10 biggest market days": [f"{d:%d %b %Y}" for d in sorted(extreme)],
        "by horizon (avg, 5%, 95%, n)": {k: (*(round(x, 3) for x in b), len(horizons[k]))
                                         for k, b in boot.items()},
        "overnight / intraday average correlation": (round(o_all, 3), round(i_all, 3)),
        "overnight share of variance (average stock)": round(var_share, 3),
        "stock-days dropped for bad opens": dropped,
        "by year (overnight, intraday)": {y: (round(a, 2), round(b, 2)) for y, a, b in
                                          zip(years, *by_year.values(), strict=True)},
    }


def _gmv(cov: np.ndarray) -> np.ndarray:
    """Global minimum-variance weights (shorting allowed): Σ⁻¹1 / 1ᵀΣ⁻¹1."""
    w = np.linalg.solve(cov, np.ones(len(cov)))
    return w / w.sum()


def report_rmt(out_dir: Path) -> dict:
    """How much of the NIFTY 50 correlation matrix is signal, the market factor's share over time,
    and whether cleaning the matrix helps out of sample."""
    from scipy.stats import spearmanr

    from ch.data import SECTORS, SYMBOLS
    from ch.rmt import clip, eigen, market_share, mp_bounds, mp_density, shuffle

    R = returns()
    W = R.loc[MEASURE_START:]
    syms = [s for s in SYMBOLS if W[s].notna().mean() > 0.99]
    W = W[syms].dropna()
    N, T = W.shape[1], W.shape[0]
    ev, vec = eigen(W.corr())
    lo, hi = mp_bounds(N, T)
    ev_sh, _ = eigen(shuffle(W, seed=0).corr())
    n_sig = int((ev > hi).sum())

    fig, ax = plt.subplots(figsize=(9, 4.6))
    bins = np.linspace(0, 3, 61)
    ax.hist(ev[ev < 3], bins=bins, color=plots.SERIES[0], alpha=0.85, density=True,
            label="NIFTY 50 stocks, 2014–26")
    ax.hist(ev_sh, bins=bins, histtype="step", color=plots.TEXT_2, lw=1.2, density=True,
            label="same returns, shuffled in time")
    x = np.linspace(lo * 0.99, hi * 1.01, 400)
    ax.plot(x, mp_density(x, N / T), color=plots.SERIES[1], lw=2,
            label="pure noise (Marchenko-Pastur)")
    ax.axvline(hi, color=plots.SERIES[1], ls="--", lw=1)
    for e in ev[1:n_sig]:
        ax.annotate(f"{e:.2f}", (e, 0.1), xytext=(0, 18), textcoords="offset points", ha="center",
                    fontsize=8, arrowprops=dict(arrowstyle="-", color=plots.MUTED))
    ax.text(2.95, 1.6, f"market factor: {ev[0]:.1f} →", ha="right", fontsize=9, color=plots.TEXT)
    ax.set_xlim(0, 3)
    ax.set_xlabel(f"eigenvalue of the {N}×{N} correlation matrix")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1, 0.88))
    ax.set_title(f"Only {n_sig} of {N} eigenvalues stood above the noise band; the market "
                 f"factor alone explained {ev[0] / N:.0%}")
    plots.save(fig, out_dir, "rmt_spectrum")

    fig, axes = plt.subplots(1, n_sig - 1, figsize=(10, 4.8), sharex=True)
    factors = {}
    for k, ax in zip(range(1, n_sig), np.atleast_1d(axes), strict=True):
        v = pd.Series(vec[:, k], index=syms)
        if v.sort_values().iloc[-1] < -v.sort_values().iloc[0]:
            v = -v  # sign is arbitrary: show the larger side positive
        ends = pd.concat([v.sort_values().head(5), v.sort_values().tail(5)])
        factors[k + 1] = ends.round(2).to_dict()
        cols = [plots.SERIES[0] if x > 0 else plots.SERIES[1] for x in ends]
        ax.barh(range(len(ends)), ends.values, color=cols)
        ax.set_yticks(range(len(ends)), [f"{s} ({SHORT[SECTORS[s]]})" for s in ends.index],
                      fontsize=7)
        ax.axvline(0, color=plots.TEXT_2, lw=0.8)
        ax.set_title(f"factor {k + 1} (eigenvalue {ev[k]:.2f})", fontsize=9, fontweight="normal")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Beyond the market: IT, commodities and state-owned firms, and pharma each set "
                 "against the rest", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out_dir, "rmt_factors")

    full = R.loc[START_ROLL:]
    ends_m = full.resample("ME").last().index
    share = {}
    for t in ends_m:
        win = full.loc[t - pd.DateOffset(years=1):t, SYMBOLS]
        win = win.loc[:, win.notna().mean() > 0.98].dropna()
        if win.shape[1] >= 30 and len(win) >= 200:
            share[t] = market_share(win)
    share = pd.Series(share)
    nifty = (1 + full["NIFTY"].fillna(0)).cumprod()
    dd = nifty / nifty.cummax() - 1
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    a1.plot(share.index, share.values * 100, color=plots.SERIES[0], lw=2)
    a1.set_ylabel("market factor's share, %\n(past year)")
    a2.fill_between(dd.index, dd.values * 100, 0, color=plots.SERIES[1], alpha=0.5, lw=0)
    a2.set_ylabel("NIFTY below\nits peak, %")
    a1.set_title(f"The market factor's share peaked at {share.max():.0%} "
                 f"({share.idxmax():%b %Y}), after the 2020 crash, not before it")
    plots.save(fig, out_dir, "rmt_share")

    m_px = nifty.resample("ME").last()
    daily = full["NIFTY"]
    nxt_ret = (m_px.shift(-1) / m_px - 1)
    vol = daily.groupby(daily.index.to_period("M")).std()
    vol.index = vol.index.to_timestamp("M")
    nxt_vol = vol.shift(-1).reindex(share.index, method="nearest")
    cur_vol = vol.reindex(share.index, method="nearest")
    df = pd.DataFrame({"share": share, "d3": share.diff(3), "ret": nxt_ret.reindex(share.index),
                       "vol": nxt_vol, "cur": cur_vol}).dropna()
    tests = {"level vs next-month return": spearmanr(df["share"], df["ret"]),
             "3-month change vs next-month return": spearmanr(df["d3"], df["ret"]),
             "level vs next-month volatility": spearmanr(df["share"], df["vol"]),
             "this month's volatility vs next month's": spearmanr(df["cur"], df["vol"])}

    oos = {"raw": [], "clipped": []}
    hist = R.loc[START_ROLL:, syms].dropna()  # estimation windows reach back before 2014
    starts = pd.date_range("2014-01-01", W.index[-1] - pd.DateOffset(months=6), freq="6MS")
    for t0 in starts:
        est = hist.loc[t0 - pd.DateOffset(years=2):t0 - pd.Timedelta(days=1)]
        test = hist.loc[t0:t0 + pd.DateOffset(months=6) - pd.Timedelta(days=1)]
        sd = est.std().to_numpy()
        for name, c in (("raw", est.corr().to_numpy()), ("clipped", clip(est.corr(), len(est)))):
            cov = c * np.outer(sd, sd)
            w = _gmv(cov)
            pred = float(np.sqrt(w @ cov @ w))
            real = float((test.to_numpy() @ w).std())
            oos[name].append((pred, real))
    oos_out = {k: {"predicted vol (daily %)": round(float(np.mean([a for a, _ in v])) * 100, 3),
                   "realised vol (daily %)": round(float(np.mean([b for _, b in v])) * 100, 3),
                   "realised / predicted": round(float(np.mean([b / a for a, b in v])), 2),
                   "periods": len(v)} for k, v in oos.items()}
    return {
        "N, T, q": (N, T, round(N / T, 4)),
        "noise band": (round(lo, 3), round(hi, 3)),
        "top eigenvalues": ev[:8].round(2).tolist(),
        "signal eigenvalues": n_sig,
        "eigenvalues below the band": int((ev < lo).sum()),
        "shuffled max / min eigenvalue": (round(ev_sh[0], 3), round(ev_sh[-1], 3)),
        "share of variance: market, other signal": (round(ev[0] / N, 3),
                                                     round(ev[1:n_sig].sum() / N, 3)),
        "factor loadings (ends)": factors,
        "rolling share: mean, min (date), max (date)": (
            round(share.mean(), 3), round(share.min(), 3), f"{share.idxmin():%b %Y}",
            round(share.max(), 3), f"{share.idxmax():%b %Y}"),
        "predictive tests (Spearman, p)": {k: (round(v.statistic, 3), round(v.pvalue, 3))
                                           for k, v in tests.items()},
        "months tested": len(df),
        "minimum variance out of sample (2-year estimate, next 6 months)": oos_out,
    }


def report_time(out_dir: Path) -> dict:
    """Average NIFTY 50 correlation through time by four trackers, which forecasts portfolio risk
    best out of sample, and whether stocks fall together more than they rise together."""
    from ch.data import SECTORS, SYMBOLS
    from ch.timevar import (
        avg_offdiag,
        dcc_corr,
        dcc_fit,
        ewma_corr,
        exceedance_corr,
        qlike,
        standardise,
    )

    R = returns()
    syms = [s for s in SYMBOLS if R.loc[MEASURE_START:, s].notna().mean() > 0.99]
    X = R.loc["2010-01-01":, syms].dropna()
    Zdf, vol = standardise(X, with_vol=True)
    Zdf = Zdf.dropna()
    Xs, vol = X.loc[Zdf.index], vol.loc[Zdf.index]
    z = Zdf.to_numpy()
    pre = Zdf.index < MEASURE_START
    a, b = dcc_fit(z[pre])  # settings from before the test period only
    test = ~pre
    T = len(z)

    paths = {"EWMA (λ 0.94)": ewma_corr(z, 0.94), "DCC": dcc_corr(z, a, b)}
    for w in (63, 250):
        roll = np.full((T, len(syms), len(syms)), np.nan)
        for i in range(w, T):
            roll[i] = np.corrcoef(z[i - w:i].T)
        paths[f"rolling {w} days"] = roll
    expanding = np.full((T, len(syms), len(syms)), np.nan)
    for i in range(250, T):
        expanding[i] = np.corrcoef(z[:i].T)
    paths["all past days"] = expanding

    fig, ax = plt.subplots(figsize=(9, 4.8))
    idx = Zdf.index[test]
    avg = {k: avg_offdiag(v[test]) for k, v in paths.items()}
    for (k, v), c, lw in zip(avg.items(), (plots.SERIES[1], plots.SERIES[0], plots.MUTED,
                                           plots.SERIES[2], plots.TEXT),
                             (1.2, 1.8, 0.8, 1.4, 1.4), strict=True):
        ax.plot(idx, v, color=c, lw=lw, label=k)
    ax.set_ylabel("average correlation between stocks (forecast for the day)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    d = pd.Series(avg["DCC"], index=idx)
    ax.set_title(f"DCC's average correlation ranged from {d.min():.2f} to {d.max():.2f} "
                 f"({d.idxmax():%b %Y})")
    plots.save(fig, out_dir, "time_avg")

    it = [s for s in syms if SECTORS[s] == "Information Technology"]
    banks = [s for s in ("HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK") if s in syms]
    ports = {"equal weight, all stocks": np.full(len(syms), 1 / len(syms)),
             "IT minus private banks": np.array([(1 / len(it)) if s in it else
                                                 (-1 / len(banks)) if s in banks else 0.0
                                                 for s in syms])}
    xs, vs = Xs.to_numpy(), vol.to_numpy()
    losses = {}
    for pname, w in ports.items():
        r_p = xs[test] @ w
        wd = w * vs[test]  # weight × each stock's forecast volatility, per day
        losses[pname] = {}
        for k, cpath in paths.items():
            var = np.einsum("ti,tij,tj->t", wd, cpath[test], wd)
            losses[pname][k] = qlike(var, r_p)
    base = "all past days"
    diffs = {}
    for pname, w in ports.items():
        r_p = xs[test] @ w
        wd = w * vs[test]
        vb = np.einsum("ti,tij,tj->t", wd, paths[base][test], wd)
        lb = np.log(vb) + r_p**2 / vb
        diffs[pname] = {}
        for k, cpath in paths.items():
            if k == base:
                continue
            vk = np.einsum("ti,tij,tj->t", wd, cpath[test], wd)
            dl = (np.log(vk) + r_p**2 / vk) - lb
            n = len(dl)
            # Newey-West standard error with 10 lags (losses are autocorrelated)
            g = [np.mean((dl[j:] - dl.mean()) * (dl[:n - j] - dl.mean())) for j in range(11)]
            lr = g[0] + 2 * sum((1 - j / 11) * g[j] for j in range(1, 11))
            diffs[pname][k] = (float(dl.mean()), float(dl.mean() / np.sqrt(lr / n)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    methods = [k for k in paths if k != base]
    for ax, pname in zip(axes, ports, strict=True):
        vals = [diffs[pname][k][0] for k in methods]
        tvals = [diffs[pname][k][1] for k in methods]
        cols = [plots.SERIES[0] if v < 0 else plots.SERIES[1] for v in vals]
        ax.barh(range(len(methods)), vals, color=cols)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.axvline(0, color=plots.TEXT_2, lw=0.8)
        ax.set_yticks(range(len(methods)),
                      [f"{m} (t {t_:.1f})".replace("-", "−") for m, t_ in
                       zip(methods, tvals, strict=True)], fontsize=8)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_title(pname, fontsize=10, fontweight="normal")
        ax.set_xlabel("QLIKE loss minus 'all past days' (below 0 = better)")
        ax.grid(axis="y", visible=False)
    best_ls = min(methods, key=lambda k: diffs["IT minus private banks"][k][0])
    fig.suptitle(f"Forecasting portfolio risk 2014–26: {best_ls} did best on the IT-minus-banks "
                 "portfolio", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out_dir, "time_forecast")

    zt = z[test]
    thetas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    n = len(syms)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng = np.random.default_rng(0)
    down, up, norm_ex = [], [], []
    for th in thetas:
        dv, uv, nv = [], [], []
        for i, j in pairs:
            dv.append(exceedance_corr(zt[:, i], zt[:, j], -th if th else -1e-12))
            uv.append(exceedance_corr(zt[:, i], zt[:, j], th if th else 1e-12))
        down.append(np.nanmean(dv))
        up.append(np.nanmean(uv))
        rho = avg_offdiag(np.corrcoef(zt.T))
        sim = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 2_000_000)
        nv.append(exceedance_corr(sim[:, 0], sim[:, 1], th if th else 1e-12))
        norm_ex.append(nv[0])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs_th = [-t for t in thetas[::-1]] + thetas
    ax.plot([-t for t in thetas], down, "o-", color=plots.SERIES[1], lw=2,
            label="both stocks down more than θ")
    ax.plot(thetas, up, "o-", color=plots.SERIES[0], lw=2, label="both up more than θ")
    ax.plot(xs_th, norm_ex[::-1] + norm_ex, "--", color=plots.TEXT_2, lw=1.2,
            label="normal distribution, same overall correlation")
    ax.set_xlabel("θ, in standard deviations (left: falls, right: rises)")
    ax.set_ylabel("average correlation on those days")
    ax.legend(loc="center right", fontsize=8)
    ax.set_title(f"Beyond 1 standard deviation, stocks moved together {down[4]:.2f} on falls "
                 f"and {up[4]:.2f} on rises")
    plots.save(fig, out_dir, "time_exceed")

    return {
        "stocks, test days": (len(syms), int(test.sum()), f"{idx[0]:%d %b %Y}"),
        "DCC a, b (fit 2010-13)": (round(a, 4), round(b, 4)),
        "average correlation by tracker: mean, min, max": {
            k: (round(float(np.nanmean(v)), 3), round(float(np.nanmin(v)), 3),
                round(float(np.nanmax(v)), 3)) for k, v in avg.items()},
        "DCC max date": f"{d.idxmax():%d %b %Y}",
        "QLIKE by portfolio": {p_: {k: round(v, 4) for k, v in l_.items()}
                               for p_, l_ in losses.items()},
        "QLIKE minus all past days (diff, t)": {p_: {k: (round(v[0], 4), round(v[1], 1))
                                                     for k, v in d_.items()}
                                                for p_, d_ in diffs.items()},
        "IT / banks used": (it, banks),
        "exceedance (theta: down, up, normal)": {t_: (round(dn, 3), round(u_, 3), round(nn, 3))
                                                 for t_, dn, u_, nn in
                                                 zip(thetas, down, up, norm_ex, strict=True)},
    }


def report_network(out_dir: Path) -> dict:
    """The minimum spanning tree of the NIFTY 50: its shape now, how tight it was each year, how
    many links survived from year to year, and which stocks were hubs."""
    from matplotlib.collections import LineCollection

    from ch.data import SECTORS, SYMBOLS
    from ch.network import degree, edge_overlap, mst_edges, radial_layout, tree_corr

    R = returns()
    W = last_years(R, 3)[SYMBOLS].dropna()
    c = W.corr()
    names = list(c.index)
    e = mst_edges(c)
    deg = degree(e, len(names))
    root = int(np.argmax(deg))
    xy = radial_layout(e, len(names), root)
    order = list(pd.Series([SECTORS[s] for s in names]).value_counts().index)
    colour = {sec: plots.SERIES[i] if i < 7 else plots.MUTED for i, sec in enumerate(order)}

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.add_collection(LineCollection([[xy[i], xy[j]] for i, j, _ in e], colors=plots.GRID,
                                     linewidths=1.5))
    for sec in order:
        idx = [i for i, s in enumerate(names) if SECTORS[s] == sec]
        ax.scatter(xy[idx, 0], xy[idx, 1], s=[25 + 25 * deg[i] for i in idx], color=colour[sec],
                   zorder=3, label=SHORT[sec] if colour[sec] != plots.MUTED else None)
    ax.scatter([], [], color=plots.MUTED, label="other sectors")
    for i, s in enumerate(names):
        ax.annotate(s, xy[i], xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6,
                    color=plots.TEXT_2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7, frameon=False)
    fig.subplots_adjust(right=0.84)
    hubs = pd.Series(deg, index=names).sort_values(ascending=False)
    ax.set_title(f"The market's skeleton, last 3 years: {len(e)} links averaging "
                 f"{tree_corr(e, c):.2f} correlation\n{(deg == 1).sum()} of {len(names)} stocks "
                 f"are leaves; biggest hub {names[root]} ({deg[root]} links)")
    plots.save(fig, out_dir, "net_tree")

    syms = [s for s in SYMBOLS if R.loc[MEASURE_START:, s].notna().mean() > 0.99]
    years = {}
    for y in range(2010, R.index[-1].year + 1):
        w = R.loc[str(y), syms].dropna()
        if len(w) >= 150:
            cc = w.corr()
            years[y] = (mst_edges(cc), cc)
    ys = sorted(years)
    tc = [tree_corr(*years[y]) for y in ys]
    allc = [float(np.mean(years[y][1].to_numpy()[np.triu_indices(len(syms), 1)])) for y in ys]
    surv = [edge_overlap(years[y][0], years[y - 1][0]) if y - 1 in years else np.nan for y in ys]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    a1.plot(ys, tc, "o-", color=plots.SERIES[0], lw=2, label="links in the tree")
    a1.plot(ys, allc, "o-", color=plots.MUTED, lw=2, label="all pairs")
    a1.set_ylabel("average correlation")
    a1.legend(loc="upper left", fontsize=8)
    a2.bar(ys, [v * 100 for v in surv], color=plots.SERIES[2])
    a2.set_ylabel("links kept from\nthe year before, %")
    a2.set_ylim(0, 100)
    a2.grid(axis="x", visible=False)
    s_ = [v for v in surv if np.isfinite(v)]
    a1.set_title(f"The tree was tightest in {ys[int(np.argmax(tc))]} ({max(tc):.2f} a link); "
                 f"{min(s_):.0%}–{max(s_):.0%} of links survived each year")
    plots.save(fig, out_dir, "net_years")

    degs = pd.DataFrame({y: degree(years[y][0], len(syms)) for y in ys}, index=syms)
    top = degs.idxmax()
    changes = int((top != top.shift()).iloc[1:].sum())
    keep = sorted(set(top) | set(degs.index[degs.max(axis=1) >= 6]),
                  key=lambda s: -degs.loc[s].max())
    fig, ax = plt.subplots(figsize=(9, 4.8))
    grid = degs.loc[keep].to_numpy()
    ax.imshow(grid, cmap=plt.matplotlib.colors.ListedColormap(plots.BLUE_RAMP), aspect="auto",
              vmin=1, vmax=max(8, grid.max()))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=7,
                    color="white" if grid[i, j] >= 6 else plots.TEXT)
    ax.set_xticks(range(len(ys)), ys, fontsize=7)
    ax.set_yticks(range(len(keep)), keep, fontsize=7)
    ax.grid(False)
    most = top.value_counts()
    ax.set_title(f"Links per stock in each year's tree: the top hub changed in {changes} of "
                 f"{len(ys) - 1} years; {most.index[0]} led {most.iloc[0]} times")
    plots.save(fig, out_dir, "net_hubs")

    return {
        "3y window": f"{W.index[0]:%d %b %Y} to {W.index[-1]:%d %b %Y}, {len(W)} days",
        "3y tree: links, average link correlation, all-pairs average": (
            len(e), round(tree_corr(e, c), 3), round(float(np.mean(c.to_numpy()[np.triu_indices(
                len(names), 1)])), 3)),
        "3y hubs": hubs.head(6).to_dict(),
        "3y leaves": int((deg == 1).sum()),
        "yearly (tree corr, all corr, survival, top hub, its links)": {
            y: (round(t_, 3), round(a_, 3), None if not np.isfinite(sv) else round(sv, 2),
                top[y], int(degs[y].max()))
            for y, t_, a_, sv in zip(ys, tc, allc, surv, strict=True)},
        "stocks in yearly trees": len(syms),
        "hub changes": changes,
        "top-hub counts": top.value_counts().to_dict(),
    }


def report_diversify(out_dir: Path) -> dict:
    """How many independent bets the NIFTY 50 is worth, and hierarchical risk parity walk-forward
    against equal weight, inverse volatility and minimum variance, after costs."""
    from ch.data import SYMBOLS
    from ch.diversify import (
        diversification_ratio,
        hrp_weights,
        inverse_vol,
        min_var_long_only,
        participation_ratio,
        walk_forward,
    )

    R = returns()
    syms = [s for s in SYMBOLS if R.loc[MEASURE_START:, s].notna().mean() > 0.99]
    X = R.loc["2010-01-01":, syms].dropna()
    ew = np.full(len(syms), 1 / len(syms))
    years = sorted(set(X.index.year))
    bets = {y: (diversification_ratio(ew, X.loc[str(y)].cov()) ** 2,
                participation_ratio(X.loc[str(y)].corr())) for y in years}

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(years, [bets[y][1] for y in years], "o-", color=plots.MUTED, lw=2,
            label="whole market: participation ratio of the correlation matrix")
    ax.plot(years, [bets[y][0] for y in years], "o-", color=plots.SERIES[0], lw=2,
            label="equal-weight portfolio: diversification ratio²")
    ax.set_ylabel(f"effective number of independent bets\n(out of {len(syms)} stocks)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", fontsize=8)
    low = min(years, key=lambda y: bets[y][0])
    lo_b, hi_b = min(b[0] for b in bets.values()), max(b[0] for b in bets.values())
    ax.set_title(f"{len(syms)} stocks held equally were worth {lo_b:.1f} to {hi_b:.1f} "
                 f"independent bets; the low was {low}")
    plots.save(fig, out_dir, "div_bets")

    methods = {"equal weight": lambda cov, corr: np.full(len(cov), 1 / len(cov)),
               "inverse volatility": lambda cov, corr: inverse_vol(cov),
               "minimum variance": lambda cov, corr: min_var_long_only(cov),
               "hierarchical risk parity": hrp_weights}
    res = {k: walk_forward(X, f, 250, MEASURE_START, 1e6) for k, f in methods.items()}
    stats = {}
    for k, r_ in res.items():
        wl = r_["wealth"]
        yrs = (wl.index[-1] - wl.index[0]).days / 365.25
        lr = np.log(wl).diff().dropna()
        dr = [diversification_ratio(w, X.loc[:t].iloc[-250:].cov()) ** 2
              for t, w in r_["weights"].iloc[::21].iterrows()]
        stats[k] = {"growth": float(wl.iloc[-1] ** (1 / yrs) - 1),
                    "vol": float(lr.std() * np.sqrt(len(lr) / yrs)),
                    "worst": float((wl / wl.cummax() - 1).min()),
                    "turnover": r_["turnover"], "costs": r_["costs"],
                    "largest weight": float(r_["weights"].max(axis=1).mean()),
                    "bets": float(np.mean(dr))}
    cols = {"equal weight": plots.TEXT, "inverse volatility": plots.SERIES[2],
            "minimum variance": plots.SERIES[1], "hierarchical risk parity": plots.SERIES[0]}

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k, r_ in res.items():
        st = stats[k]
        ax.plot(r_["wealth"].index, r_["wealth"].values, color=cols[k], lw=1.6,
                label=f"{k}: {st['growth']:.1%} a year, volatility {st['vol']:.1%}, "
                      f"worst fall {st['worst']:.0%}")
    ax.set_yscale("log")
    _rupee_log(ax)
    ax.set_ylabel("wealth from ₹1 in January 2014 (log scale)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("After costs, hierarchical risk parity landed between inverse volatility and "
                 "minimum variance")
    plots.save(fig, out_dir, "div_walk")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = list(stats)
    xs = np.arange(len(names))
    vals = [stats[k]["costs"] * 100 for k in names]
    ax.bar(xs, vals, 0.55, color=[cols[k] for k in names])
    for i, k in enumerate(names):
        ax.text(i, vals[i] + 0.02, f"{vals[i]:.2f}%\nturnover {stats[k]['turnover']:.1f}× a year"
                f"\nlargest holding {stats[k]['largest weight']:.0%}", ha="center", fontsize=8)
    ax.set_xticks(xs, names, fontsize=9)
    ax.set_ylim(0, max(vals) * 1.6)
    ax.set_ylabel("trading costs, % of wealth a year")
    ax.grid(axis="x", visible=False)
    hrp = stats["hierarchical risk parity"]
    ax.set_title(f"Hierarchical risk parity traded {hrp['turnover']:.1f}× its portfolio a year "
                 f"and paid {hrp['costs']:.2%}")
    plots.save(fig, out_dir, "div_costs")

    return {
        "stocks, test from": (len(syms), MEASURE_START),
        "effective bets by year (equal weight DR², participation ratio)": {
            y: (round(a_, 2), round(b_, 2)) for y, (a_, b_) in bets.items()},
        "walk-forward 2014-26, 250-day estimates, monthly, ₹10 lakh, costs": {
            k: {m: round(v, 4) for m, v in st.items()} for k, st in stats.items()},
        "growth / volatility": {k: round(st["growth"] / st["vol"], 2) for k, st in stats.items()},
    }


REPORTS = {"heatmap": report_heatmap, "measure": report_measure, "rmt": report_rmt,
           "time": report_time, "network": report_network, "diversify": report_diversify}
