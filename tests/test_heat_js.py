import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ch.cluster import adjusted_rand, remove_market

JS = Path(__file__).resolve().parents[1] / "src" / "ch" / "heat.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _node(expr: str):
    script = f"const m = require({json.dumps(str(JS))}); console.log(JSON.stringify({expr}));"
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0, 0.01, n)
    cols = {"NIFTY": mkt}
    for g in "ABC":
        f = rng.normal(0, 0.01, n)
        for i in range(3):
            cols[f"{g}{i}"] = 0.8 * mkt + f + rng.normal(0, 0.008, n)
    R = pd.DataFrame(cols, index=pd.bdate_range("2021-01-01", periods=n))
    R.iloc[:30, 1] = np.nan
    return R


def test_browser_correlation_matches_pandas_on_complete_days():
    R = _data()
    names = [c for c in R.columns if c != "NIFTY"]
    cols = {k: [None if np.isnan(x) else x for x in R[k]] for k in R}
    js = _node(f"m.corrMatrix({json.dumps(cols)}, {json.dumps(names)}, null)")
    py = R[names].dropna().corr().to_numpy()
    assert np.array(js["c"]) == pytest.approx(py, abs=1e-12)
    resid = _node(f"m.corrMatrix({json.dumps(cols)}, {json.dumps(names)}, 'NIFTY')")
    pr = remove_market(R, names, "NIFTY").corr().to_numpy()
    assert np.array(resid["c"]) == pytest.approx(pr, abs=1e-10)


def test_browser_average_linkage_matches_scipy_clusters():
    R = _data(seed=3)
    names = [c for c in R.columns if c != "NIFTY"]
    c = R[names].dropna().corr()
    d = np.sqrt(np.clip(2 * (1 - c.to_numpy()), 0, None))
    np.fill_diagonal(d, 0)
    py = fcluster(linkage(squareform(d, checks=False), "average"), 3, criterion="maxclust")
    js = _node(f"m.clusterOrder({json.dumps(c.to_numpy().tolist())}, 3)")
    assert adjusted_rand(js["labels"], py) == pytest.approx(1.0)
    assert sorted(js["order"]) == list(range(len(names)))
    groups = [py[i] for i in js["order"]]
    assert sum(groups[i] != groups[i + 1] for i in range(len(groups) - 1)) == 2  # contiguous


def test_browser_rolling_pair_correlation():
    R = _data()
    cols = {k: [None if np.isnan(x) else x for x in R[k]] for k in R}
    js = _node(f"m.rollingPair({json.dumps(cols)}, 'A1', 'B2', 63, null)")
    py = R["A1"].rolling(63).corr(R["B2"])
    got = np.array([np.nan if v is None else v for v in js], dtype=float)
    mask = ~np.isnan(py.to_numpy())
    assert got[mask] == pytest.approx(py.to_numpy()[mask], abs=1e-10)


def test_browser_rank_correlation_matches_spearman():
    R = _data(seed=5)
    names = [c for c in R.columns if c != "NIFTY"]
    cols = {k: [None if np.isnan(x) else x for x in R[k]] for k in R}
    js = _node(f"m.corrMatrix({json.dumps(cols)}, {json.dumps(names)}, null, true)")
    py = R[names].dropna().corr(method="spearman").to_numpy()
    assert np.array(js["c"]) == pytest.approx(py, abs=1e-12)


def test_browser_weekly_returns_match_python():
    from ch.measure import resampled_returns

    R = _data(seed=6)
    R.iloc[:, 1] = R.iloc[:, 1].fillna(0)
    dates = list(R.index.strftime("%Y-%m-%d"))
    cols = {k: list(R[k]) for k in R}
    js = _node(f"m.toWeekly({json.dumps(cols)}, {json.dumps(dates)})")
    py = resampled_returns(R, "W-FRI")
    assert np.array(js["cols"]["A1"]) == pytest.approx(py["A1"].to_numpy(), rel=1e-9)
    assert js["dates"] == list(py.index.strftime("%Y-%m-%d"))


def test_browser_eigenvalues_match_numpy():
    from ch.rmt import mp_bounds

    R = _data(seed=7)
    c = R.dropna().corr().to_numpy()
    js = _node(f"m.eigenvalues({json.dumps(c.tolist())})")
    assert js == pytest.approx(sorted(np.linalg.eigvalsh(c), reverse=True), abs=1e-8)
    b = _node("m.mpBounds(43, 3181)")
    assert b == pytest.approx(list(mp_bounds(43, 3181)))


def test_browser_ewma_pair_correlation():
    R = _data(seed=8)
    x, y = R["A1"].to_numpy().copy(), R["C2"].to_numpy().copy()
    x[40] = np.nan  # a gap inside the seed period: seed on days both have
    cols = {k: [None if np.isnan(v) else v for v in R[k]] for k in R}
    cols["A1"][40] = None
    js = _node(f"m.ewmaPair({json.dumps(cols)}, 'A1', 'C2', 0.94, 60)")
    both = np.flatnonzero(~np.isnan(x) & ~np.isnan(y))[:60]
    start = both[-1] + 1
    sxx, syy, sxy = (np.mean(v) for v in (x[both] ** 2, y[both] ** 2, x[both] * y[both]))
    exp = {}
    for t in range(start, len(x)):
        exp[t] = sxy / np.sqrt(sxx * syy)  # forecast for day t from days before it
        if np.isnan(x[t]) or np.isnan(y[t]):
            continue
        sxx = 0.94 * sxx + 0.06 * x[t] ** 2
        syy = 0.94 * syy + 0.06 * y[t] ** 2
        sxy = 0.94 * sxy + 0.06 * x[t] * y[t]
    assert js[start - 1] is None
    assert [js[t] for t in exp] == pytest.approx(list(exp.values()), rel=1e-9)


def test_browser_mst_matches_scipy():
    from ch.network import mst_edges, radial_layout

    rng = np.random.default_rng(9)
    x = rng.normal(size=(300, 15)) + rng.normal(size=(300, 1))
    c = pd.DataFrame(x).corr()
    js = _node(f"m.mstEdges({json.dumps(c.to_numpy().tolist())})")
    py = mst_edges(c)
    assert sorted(tuple(sorted(e[:2])) for e in js) == [(a, b) for a, b, _ in py]
    root = int(np.argmax(np.bincount(np.array([e[:2] for e in py]).ravel(), minlength=15)))
    xy = _node(f"m.radialLayout({json.dumps([[a, b, w] for a, b, w in py])}, 15, {root})")
    assert np.array(xy) == pytest.approx(radial_layout(py, 15, root), abs=1e-12)


def test_browser_participation_ratio_matches_python():
    from ch.diversify import participation_ratio

    R = _data(seed=10)
    c = R.dropna().corr().to_numpy()
    js = _node(f"m.participationRatio(m.eigenvalues({json.dumps(c.tolist())}))")
    assert js == pytest.approx(participation_ratio(c), rel=1e-9)
