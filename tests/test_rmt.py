import numpy as np
import pandas as pd
import pytest

from ch.rmt import clip, eigen, market_share, mp_bounds, mp_density, shuffle, signal_count


def _factors(n_stocks=40, t=2000, k=3, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 1, (t, k)) * np.array([1.0, 0.6, 0.5])[:k]
    load = rng.normal(0.5, 0.3, (n_stocks, k))
    x = f @ load.T + rng.normal(0, 1, (t, n_stocks))
    return pd.DataFrame(x)


def test_mp_bounds_and_density_integrate_to_one():
    lo, hi = mp_bounds(n=50, t=1000)
    q = 50 / 1000
    assert (lo, hi) == pytest.approx(((1 - q**0.5) ** 2, (1 + q**0.5) ** 2))
    x = np.linspace(lo, hi, 20001)
    assert np.trapezoid(mp_density(x, q), x) == pytest.approx(1, abs=1e-3)


def test_noise_has_no_signal_and_factors_are_found():
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.normal(size=(2000, 40)))
    assert signal_count(noise.corr(), len(noise)) == 0
    data = _factors()
    assert signal_count(data.corr(), len(data)) == 3


def test_shuffle_keeps_each_column_and_destroys_structure():
    data = _factors()
    s = shuffle(data, seed=2)
    assert sorted(s[0]) == pytest.approx(sorted(data[0]))
    assert signal_count(s.corr(), len(s)) == 0


def test_clip_keeps_signal_and_unit_diagonal():
    data = _factors(t=300)
    c = data.corr()
    cl = clip(c, len(data))
    assert np.diag(cl) == pytest.approx(1)
    ev_raw, _ = eigen(c)
    ev_cl, _ = eigen(cl)
    assert ev_cl[0] == pytest.approx(ev_raw[0], rel=0.05)  # the market factor survives
    assert np.allclose(cl, cl.T)


def test_market_share_is_top_eigenvalue_over_n():
    data = _factors()
    ev, _ = eigen(data.corr())
    assert market_share(data) == pytest.approx(ev[0] / 40)
