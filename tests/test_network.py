import numpy as np
import pandas as pd
import pytest

from ch.network import degree, edge_overlap, mst_edges, radial_layout, tree_corr


def _corr():
    # A star: stock 0 correlates 0.8 with 1, 2, 3; the leaves correlate 0.5 with each other
    c = np.full((4, 4), 0.5)
    c[0, 1:] = c[1:, 0] = 0.8
    np.fill_diagonal(c, 1)
    return pd.DataFrame(c, index=list("ABCD"), columns=list("ABCD"))


def test_mst_of_a_star_is_the_star():
    e = mst_edges(_corr())
    assert sorted((a, b) for a, b, _ in e) == [(0, 1), (0, 2), (0, 3)]
    assert all(d == pytest.approx(np.sqrt(2 * 0.2)) for _, _, d in e)
    assert degree(e, 4).tolist() == [3, 1, 1, 1]
    assert tree_corr(e, _corr()) == pytest.approx(0.8)


def test_mst_has_n_minus_one_edges_and_minimum_total_length():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 12)) + rng.normal(size=(500, 1))
    c = pd.DataFrame(x).corr()
    e = mst_edges(c)
    assert len(e) == 11
    d = np.sqrt(2 * (1 - c.to_numpy()))
    # every non-tree edge is at least as long as the longest tree edge on the path it would close
    total = sum(w for *_, w in e)
    alt = sum(sorted(d[np.triu_indices(12, 1)])[:11])
    assert total >= alt - 1e-12  # can't beat the 11 shortest edges


def test_edge_overlap_and_layout():
    e1 = [(0, 1, 0.1), (1, 2, 0.2), (2, 3, 0.3)]
    e2 = [(1, 0, 0.1), (2, 3, 0.2), (0, 3, 0.9)]
    assert edge_overlap(e1, e2) == pytest.approx(2 / 3)
    xy = radial_layout(e1, 4, root=1)
    assert xy[1] == pytest.approx([0, 0])
    assert np.hypot(*xy[0]) == pytest.approx(1) and np.hypot(*xy[3]) == pytest.approx(2)
