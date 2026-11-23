import json

import numpy as np
import pandas as pd

from ch.data import SYMBOLS
from ch.site import build_html, heatmap_data


def _returns():
    idx = pd.bdate_range("2020-01-01", periods=1300)
    r = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (1300, len(SYMBOLS) + 3)),
                     index=idx, columns=SYMBOLS + ["NIFTY", "NIFTYBEES", "GOLDBEES"])
    r.iloc[:400, 0] = np.nan  # a recent listing
    return r


def test_heatmap_data_is_compact_and_keeps_gaps():
    d = heatmap_data(_returns())
    assert len(d["dates"]) == 1250 and len(d["r"]["NIFTY"]) == 1250
    first = SYMBOLS[0]
    assert d["r"][first][0] is None and isinstance(d["r"][first][-1], int)
    assert json.loads(json.dumps(d)) == d
    assert set(d["sectors"]) == set(SYMBOLS)


def test_page_builds(monkeypatch):
    import ch.site

    idx = pd.bdate_range("2020-01-01", periods=1300)
    glob = pd.DataFrame({"SP500TR": np.linspace(1000, 1500, 1300),
                         "USDINR": np.linspace(70, 90, 1300)}, index=idx)
    monkeypatch.setattr(ch.site, "load_global", lambda: glob)  # no data files in CI
    html = build_html(_returns())
    assert 'id="across-assets"' in html
    assert "function clusterOrder" in html and 'id="correlation-heatmap"' in html
    assert "$" + "js" not in html and "$" + "data" not in html
