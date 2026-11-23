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


REPORTS = {"heatmap": report_heatmap}
