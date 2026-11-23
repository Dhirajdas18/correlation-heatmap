"""Interactive dashboard: a live correlation heatmap of the NIFTY 50, as one static page."""

import json
import re
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ch import plots

FONT = "system-ui, -apple-system, \"Segoe UI\", Roboto, sans-serif"
ICON = (
    "data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22%3E"
    "%3Crect x=%221%22 y=%221%22 width=%226%22 height=%226%22 fill=%22%23b2182b%22/%3E"
    "%3Crect x=%229%22 y=%229%22 width=%226%22 height=%226%22 fill=%22%23b2182b%22/%3E"
    "%3Crect x=%229%22 y=%221%22 width=%226%22 height=%226%22 fill=%22%232166ac%22/%3E"
    "%3Crect x=%221%22 y=%229%22 width=%226%22 height=%226%22 fill=%22%232166ac%22/%3E%3C/svg%3E"
)
HERE = Path(__file__).parent
JS = (HERE / "heat.js").read_text(encoding="utf-8")
HEATMAP = (HERE / "heatmap.html").read_text(encoding="utf-8")
PAGE = Template((HERE / "page.html").read_text(encoding="utf-8"))
REPO = "https://github.com/Dhirajdas18/correlation-heatmap"
DAYS = 1250  # five years of trading days embedded in the page
SCALE = 1e-5  # returns stored as integers in units of 0.001%


def heatmap_data(r: pd.DataFrame) -> dict:
    """The last five years of daily returns of the NIFTY 50 and the index, as integers."""
    from ch.data import SECTORS, SYMBOLS
    from ch.reports import SHORT

    symbols = sorted(SYMBOLS)
    recent = r[symbols + ["NIFTY"]].iloc[-DAYS:]
    data = {s: [None if pd.isna(x) else int(round(x / SCALE)) for x in recent[s]]
            for s in symbols + ["NIFTY"]}
    order = list(pd.Series([SECTORS[s] for s in symbols]).value_counts().index)
    # seven biggest sectors get the categorical colours; the rest share grey (dataviz: ≤ 8)
    colours = {sec: plots.SERIES[i] if i < 7 else plots.MUTED for i, sec in enumerate(order)}
    return {"symbols": symbols, "sectors": {s: SECTORS[s] for s in symbols}, "short": SHORT,
            "colours": colours, "other": plots.MUTED,
            "dates": list(recent.index.strftime("%Y-%m-%d")),
            "last": f"{recent.index[-1]:%d %b %Y}", "r": data}


def heatmap_section(data: pd.DataFrame) -> str:
    return (HEATMAP.replace("$js", JS)
            .replace("$data", json.dumps(heatmap_data(data), separators=(",", ":"))))



def load_global() -> pd.DataFrame:
    """The US index and USD/INR table, with the fixes file's blank rows dropped."""
    from ch.data import FIXES, GLOBAL_PATH

    glob = pd.read_parquet(GLOBAL_PATH).copy()
    fixes = pd.read_csv(FIXES, parse_dates=["date"])
    for row in fixes[fixes["action"] == "blank"].itertuples():
        if row.symbol in glob.columns and row.date in glob.index:
            glob.loc[row.date, row.symbol] = np.nan
    return glob


def cross_figure(data: pd.DataFrame) -> go.Figure:
    """Past-year correlation of NIFTY's weekly returns with the US market, gold and USD/INR."""
    from ch.crossasset import rolling_corr, weekly_returns

    glob = load_global()
    nifty = (1 + data["NIFTYBEES"].dropna()).cumprod()
    gold = (1 + data["GOLDBEES"].dropna()).cumprod()
    W = weekly_returns({"NIFTY": nifty, "US market ($)": glob["SP500TR"], "gold (₹)": gold,
                        "USD/INR": glob["USDINR"]})
    fig = go.Figure()
    for k, c in (("US market ($)", plots.SERIES[0]), ("gold (₹)", plots.SERIES[3]),
                 ("USD/INR", plots.SERIES[1])):
        rc = rolling_corr(W["NIFTY"], W[k], 52).dropna()
        fig.add_scatter(x=list(rc.index.strftime("%Y-%m-%d")), y=rc.round(3).tolist(),
                        name=f"NIFTY with {k}", mode="lines", line=dict(color=c, width=2),
                        hovertemplate=f"{k}: %{{y:.2f}}<extra></extra>")
    fig.update_layout(paper_bgcolor=plots.SURFACE, plot_bgcolor=plots.SURFACE,
                      font=dict(family=FONT, color=plots.TEXT_2, size=13), hovermode="x unified",
                      margin=dict(l=56, r=16, t=56, b=40),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
                      xaxis=dict(gridcolor=plots.GRID),
                      yaxis=dict(title="correlation of weekly returns, past year",
                                 gridcolor=plots.GRID, range=[-0.8, 1], zeroline=True,
                                 zerolinecolor=plots.GRID))
    return fig


SECTIONS = [
    ("Correlation heatmap",
     "How closely each pair of today’s NIFTY 50 stocks moved, day by day. Removing the market "
     "takes out each stock’s beta × NIFTY’s move, so what is left shows which stocks share "
     "sector and company news. Ordered by cluster, similar stocks sit next to each other. Click "
     "a square to see that pair’s correlation through time.",
     heatmap_section),
    ("Across assets",
     "NIFTY’s weekly returns against the US market, gold in rupees and the dollar’s price in "
     "rupees, each correlation over the past year. Weekly returns let each market close on its "
     "own calendar.",
     cross_figure),
]


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build_html(data: pd.DataFrame) -> str:
    parts, nav = [], []
    for title, blurb, make in SECTIONS:
        out = make(data)
        body = out if isinstance(out, str) else out.to_html(
            full_html=False, include_plotlyjs=False, div_id=f"fig-{_slug(title)}",
            config={"displaylogo": False, "responsive": True})
        sid = _slug(title)
        nav.append(f'<li><a href="#{sid}">{title}</a></li>')
        parts.append(f'<section id="{sid}">\n<h2>{title}</h2>\n<p>{blurb}</p>\n{body}\n</section>')
    d = plots.DARK
    return PAGE.substitute(
        icon=ICON, font=FONT, repo=REPO, updated=f"{data.index[-1]:%d %b %Y}",
        surface=plots.SURFACE, text=plots.TEXT, text2=plots.TEXT_2, grid=plots.GRID,
        accent=plots.SERIES[0], muted=plots.MUTED, d_surface=d[plots.SURFACE], d_text=d[plots.TEXT],
        d_text2=d[plots.TEXT_2], d_grid=d[plots.GRID], d_accent=d[plots.SERIES[0]],
        d_muted="#6b6a64",
        dark=json.dumps({k.lower(): v for k, v in d.items()}),
        dark_lines=json.dumps({k.lower(): v for k, v in plots.DARK_LINES.items()}),
        nav="\n".join(nav), sections="\n".join(parts),
    )
