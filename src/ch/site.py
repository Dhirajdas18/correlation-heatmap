"""Interactive dashboard: a live correlation heatmap of the NIFTY 50, as one static page."""

import json
import re
from pathlib import Path
from string import Template

import pandas as pd

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
    return {"symbols": symbols, "sectors": {s: SECTORS[s] for s in symbols}, "short": SHORT,
            "dates": list(recent.index.strftime("%Y-%m-%d")),
            "last": f"{recent.index[-1]:%d %b %Y}", "r": data}


def heatmap_section(data: pd.DataFrame) -> str:
    return (HEATMAP.replace("$js", JS)
            .replace("$data", json.dumps(heatmap_data(data), separators=(",", ":"))))


SECTIONS = [
    ("Correlation heatmap",
     "How closely each pair of today’s NIFTY 50 stocks moved, day by day. Removing the market "
     "takes out each stock’s beta × NIFTY’s move, so what is left shows which stocks share "
     "sector and company news. Ordered by cluster, similar stocks sit next to each other. Click "
     "a square to see that pair’s correlation through time.",
     heatmap_section),
]


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build_html(data: pd.DataFrame) -> str:
    parts, nav = [], []
    for title, blurb, make in SECTIONS:
        out = make(data)
        body = out if isinstance(out, str) else out.to_html(
            full_html=False, include_plotlyjs=False,
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
