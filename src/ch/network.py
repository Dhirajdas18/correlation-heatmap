"""The market as a network: the minimum spanning tree (Mantegna 1999).

With distance d = √(2(1 − ρ)) between every pair, the minimum spanning tree is the N − 1 links that
connect all N stocks with the smallest total distance: each stock hangs off the one it moves most
with, without loops. A stock's degree (number of links) shows whether it is a hub. The tree's
average link correlation rises when stocks move together.
"""

from collections import deque

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree

from ch.cluster import distance


def mst_edges(corr: pd.DataFrame) -> list[tuple[int, int, float]]:
    d = distance(corr).to_numpy().copy()
    np.fill_diagonal(d, 0)
    d = np.where(d == 0, 1e-12, d)  # csgraph treats 0 as "no edge"
    np.fill_diagonal(d, 0)
    t = minimum_spanning_tree(d).tocoo()
    return sorted((int(min(i, j)), int(max(i, j)), float(w))
                  for i, j, w in zip(t.row, t.col, t.data, strict=True))


def degree(edges, n: int) -> np.ndarray:
    deg = np.zeros(n, dtype=int)
    for i, j, _ in edges:
        deg[i] += 1
        deg[j] += 1
    return deg


def tree_corr(edges, corr: pd.DataFrame) -> float:
    c = np.asarray(corr, dtype=float)
    return float(np.mean([c[i, j] for i, j, _ in edges]))


def edge_overlap(e1, e2) -> float:
    """Share of the first tree's links also in the second."""
    a = {(min(i, j), max(i, j)) for i, j, _ in e1}
    b = {(min(i, j), max(i, j)) for i, j, _ in e2}
    return len(a & b) / len(a)


def radial_layout(edges, n: int, root: int) -> np.ndarray:
    """Root at the centre, each node one ring further out than its parent, every subtree given an
    angle in proportion to its number of leaves."""
    nbrs = {i: [] for i in range(n)}
    for i, j, _ in edges:
        nbrs[i].append(j)
        nbrs[j].append(i)
    parent, order, depth = {root: None}, [], {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        order.append(u)
        for v in sorted(nbrs[u]):
            if v not in parent:
                parent[v], depth[v] = u, depth[u] + 1
                q.append(v)
    kids = {u: [v for v in sorted(nbrs[u]) if parent.get(v) == u] for u in order}
    leaves = {}
    for u in reversed(order):
        leaves[u] = max(1, sum(leaves[v] for v in kids[u]))
    xy = np.zeros((n, 2))
    span = {root: (0.0, 2 * np.pi)}
    for u in order:
        lo, hi = span[u]
        mid = (lo + hi) / 2
        xy[u] = depth[u] * np.cos(mid), depth[u] * np.sin(mid)
        start = lo
        for v in kids[u]:
            width = (hi - lo) * leaves[v] / leaves[u]
            span[v] = (start, start + width)
            start += width
    return xy
