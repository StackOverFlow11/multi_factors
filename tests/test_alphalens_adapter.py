"""alphalens factor-metrics adapter (P2-4, network-free).

A thin, report-only wrapper over alphalens-reloaded: it consumes the factor Series
plus a wide price frame and produces IC mean / IC-IR / per-quantile mean returns,
tagging the ``backend`` it used. Unavailable/erroring alphalens must be disclosed
and fall back to the simple-pandas metrics — never a silent fake.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analytics import alphalens_adapter as aa
from analytics.alphalens_adapter import alphalens_factor_metrics


def _synthetic_factor_and_prices(n_days=40, n_sym=12):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    syms = [f"{i:06d}.SZ" for i in range(n_sym)]
    # each symbol a distinct deterministic upward drift; higher index = stronger.
    prices = pd.DataFrame(
        {s: 100.0 + j + 0.3 * np.arange(n_days) + j * 0.05 * np.arange(n_days)
         for j, s in enumerate(syms)},
        index=dates,
    )
    # factor = the symbol's drift rank (constant per symbol) -> well-defined IC.
    idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    factor = pd.Series(
        [syms.index(s) for _, s in idx], index=idx, dtype=float, name="momentum_20"
    )
    return factor, prices


def test_alphalens_metrics_reports_ic_and_quantiles():
    factor, prices = _synthetic_factor_and_prices()
    out = alphalens_factor_metrics(factor, prices, quantiles=5, period=1)
    assert out["backend"] == "alphalens"
    assert "ic_mean" in out and math.isfinite(out["ic_mean"])
    assert "ic_ir" in out
    # quantile_mean: one entry per bucket 1..5
    assert set(out["quantile_mean"].keys()) == {1, 2, 3, 4, 5}


def test_alphalens_unavailable_discloses_and_keeps_fallback(monkeypatch):
    def _raise():
        raise ImportError("alphalens not installed")

    monkeypatch.setattr(aa, "_import_alphalens", _raise)
    factor, prices = _synthetic_factor_and_prices()
    out = alphalens_factor_metrics(
        factor, prices, simple_fallback={"ic_mean": 0.05, "ic_ir": 0.4}
    )
    assert out["backend"] == "unavailable"  # honest
    assert out["ic_mean"] == 0.05 and out["ic_ir"] == 0.4  # simple fallback kept


def test_alphalens_error_discloses_error_type(monkeypatch):
    class _Boom:
        class utils:
            @staticmethod
            def get_clean_factor_and_forward_returns(*a, **k):
                raise ValueError("not enough names")

    monkeypatch.setattr(aa, "_import_alphalens", lambda: _Boom)
    factor, prices = _synthetic_factor_and_prices()
    out = alphalens_factor_metrics(factor, prices, simple_fallback={"ic_mean": 0.01})
    assert out["backend"] == "error"
    assert out["error_type"] == "ValueError"  # type only
    assert out["ic_mean"] == 0.01


def test_alphalens_suppresses_stdout(capsys):
    # alphalens prints a "Dropped ..." banner; the adapter must not pollute stdout.
    factor, prices = _synthetic_factor_and_prices()
    alphalens_factor_metrics(factor, prices, quantiles=5, period=1)
    captured = capsys.readouterr()
    assert "Dropped" not in captured.out
