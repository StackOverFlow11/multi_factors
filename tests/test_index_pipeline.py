"""Pipeline wiring for PIT index universe.

These tests focus on the pipeline boundary, not tushare itself. The feed is
monkeypatched so no network or token is touched.
"""

from __future__ import annotations

import pandas as pd

import qt.pipeline as pipeline
from qt.config import RootConfig


class _FakeLogger:
    def info(self, *_args, **_kwargs):
        return None


def _index_cfg() -> RootConfig:
    return RootConfig(
        data={
            "source": "tushare",
            "freq": "D",
            "start": "2024-02-15",
            "end": "2024-03-31",
            "external_secret_file": "/tmp/fake.json",
            "tushare_token_key": "tushare.token",
            "output_name": "daily",
        },
        universe={
            "type": "index",
            "index_code": "000300.SH",
            "symbols": [],
            "filters": {"missing_close": True},
        },
        factors=[{"name": "momentum_20", "enabled": True, "params": {"window": 20}}],
        alpha={"model": "equal_weight", "params": {}},
        portfolio={"constructor": "topn_equal_weight", "top_n": 3},
        backtest={"rebalance": "monthly"},
        cost={"fee_rate": 0.001},
        output={
            "root_dir": "artifacts",
            "data_dir": "artifacts/data",
            "factor_dir": "artifacts/factors",
            "report_dir": "artifacts/reports",
            "log_dir": "artifacts/logs",
        },
    )


def test_index_universe_fetches_pre_start_snapshot_for_asof(monkeypatch):
    """Pipeline must request pre-start snapshots so first as-of date is valid."""
    calls: list[tuple[str, str, str]] = []

    class _FakeIndexFeed:
        def __init__(self, secret_file, token_key="tushare.token"):
            self.secret_file = secret_file
            self.token_key = token_key

        def get_constituents(self, index_code: str, start: str, end: str) -> pd.DataFrame:
            calls.append((index_code, start, end))
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-31", "2024-03-29"]),
                    "symbol": ["000001.SZ", "000002.SZ"],
                    "weight": [1.0, 1.0],
                }
            )

    monkeypatch.setattr(pipeline, "IndexConstituentsFeed", _FakeIndexFeed)

    universe, symbols = pipeline._build_universe(_index_cfg(), _FakeLogger())

    assert symbols == ["000001.SZ", "000002.SZ"]
    assert calls == [("000300.SH", "2023-02-10", "2024-03-31")]
    # The run starts after the Jan snapshot but before the Mar snapshot. Without
    # the pre-start fetch this as-of membership would be empty.
    assert universe.members(pd.Timestamp("2024-02-15")) == ["000001.SZ"]
