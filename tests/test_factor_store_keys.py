"""Factor-store key derivation (refactor D3, commit 2): hashes / keys / fingerprint.

Network-free. Covers the complete-key contract (red line #6): params_hash,
code_hash (own module + enumerated shared set), the StoreKey path, and the data
fingerprint DERIVED FROM the adjustment declaration — plus the three key-mutation
teeth the D3 acceptance requires:

  * a shared-set member's CONTENT change -> code_hash changes;
  * a schema-bearing module's CONTENT change -> schema fingerprint changes;
  * (allowlist teeth live in test_factor_store_allowlist.py).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import factors.registry.builtin  # noqa: F401 - populate DEFAULT_REGISTRY
from data.availability_policy import Adjustment
from factors.base import Factor
from factors.registry.registry import build
from factors.spec import FactorSpec, PanelField
from factors.store import (
    adj_events_hash,
    code_hash,
    data_fingerprint,
    params_hash,
    schema_version_hash,
    shared_set_labeled_files,
    store_key,
)
from factors.store.code_hash import factor_module_labeled_file
from factors.store.hashing import content_hash_of_labeled_files, short_hash


def _vol():
    return build("volatility_20", {"window": 20})


def _ridge():
    return build("ridge_minute_return_20", {"lookback_days": 20})


def _value_ep():
    return build("value_ep")


# --------------------------------------------------------------------------- #
# params_hash
# --------------------------------------------------------------------------- #
def test_params_hash_is_stable_and_key_order_independent():
    assert params_hash({"window": 20}) == params_hash({"window": 20})
    assert params_hash({"a": 1, "b": 2}) == params_hash({"b": 2, "a": 1})


def test_params_hash_distinguishes_different_params():
    assert params_hash({"window": 20}) != params_hash({"window": 10})


def test_params_hash_none_equals_empty():
    assert params_hash(None) == params_hash({})


# --------------------------------------------------------------------------- #
# code_hash: the shared set + module identity
# --------------------------------------------------------------------------- #
def test_shared_set_is_the_enumerated_four_plus_ops():
    labels = {label for label, _ in shared_set_labeled_files()}
    assert "factors.compute.minute.primitives" in labels
    assert "factors.base" in labels
    assert "factors.spec" in labels
    # factors.ops expands to every module under the package
    assert any(label.startswith("factors.ops") for label in labels)


def test_code_hash_is_stable():
    assert code_hash(_vol()) == code_hash(_vol())


def test_code_hash_same_module_factors_share_it():
    # value_ep and volatility_20 are BOTH defined in candidates.py, so a change to
    # that module invalidates both — their code hash is deliberately identical.
    assert code_hash(_value_ep()) == code_hash(_vol())


def test_code_hash_distinguishes_different_modules():
    assert code_hash(_ridge()) != code_hash(_vol())


def test_code_hash_changes_when_a_shared_set_member_content_changes(tmp_path):
    # MUTATION (D3 acceptance): mutate a shared-set member's CONTENT -> the hash
    # MUST change, proving the shared set is genuinely folded into code_hash.
    factor = _vol()
    own = factor_module_labeled_file(factor)
    shared = list(shared_set_labeled_files())
    base = content_hash_of_labeled_files([own, *shared])

    target_label = "factors.compute.minute.primitives"
    mutated_items = [own]
    swapped = False
    for label, path in shared:
        if label == target_label:
            copy = tmp_path / "primitives_mutated.py"
            copy.write_bytes(Path(path).read_bytes() + b"\n# D3 mutation\n")
            mutated_items.append((label, copy))
            swapped = True
        else:
            mutated_items.append((label, path))
    assert swapped, "shared set unexpectedly lacks the primitives module"
    mutated = content_hash_of_labeled_files(mutated_items)
    assert mutated != base

    # And the real code_hash equals the un-mutated fold (the shared set IS in it).
    assert code_hash(factor) == base


def test_content_hash_is_path_independent(tmp_path):
    # Same content + same label under two different paths -> identical hash
    # (checkout independence: a moved repo must not invalidate stored values).
    a = tmp_path / "a.py"
    b = tmp_path / "sub" / "b.py"
    b.parent.mkdir()
    a.write_bytes(b"X = 1\n")
    b.write_bytes(b"X = 1\n")
    assert content_hash_of_labeled_files([("m", a)]) == content_hash_of_labeled_files([("m", b)])


# --------------------------------------------------------------------------- #
# StoreKey
# --------------------------------------------------------------------------- #
def test_store_key_filename_and_relpath():
    key = store_key(_vol(), view="decision", params={"window": 20})
    assert key.factor_id == "volatility_20"
    assert key.view == "decision"
    assert key.filename() == f"{key.params_hash}__{key.code_hash}__decision.parquet"
    assert key.relpath() == f"volatility_20/{key.filename()}"
    # short hashes keep filenames readable
    assert len(key.params_hash) == len(short_hash("0" * 64))


def test_store_key_view_must_be_legal():
    with pytest.raises(ValueError, match="unknown store view"):
        store_key(_vol(), view="intraday", params={})


def test_store_key_close_and_decision_differ():
    d = store_key(_vol(), view="decision", params={})
    c = store_key(_vol(), view="close", params={})
    assert d.view == "decision" and c.view == "close"
    assert d.filename() != c.filename()


# --------------------------------------------------------------------------- #
# data fingerprint DERIVED FROM adjustment
# --------------------------------------------------------------------------- #
def test_returns_invariant_fingerprint_omits_adj_events():
    fp = data_fingerprint(adjustment=Adjustment.RETURNS_INVARIANT)
    assert fp["adj_events"] is None
    assert fp["adjustment"] == "returns_invariant"
    assert fp["schema_version"] == schema_version_hash()


def test_none_fingerprint_omits_adj_events():
    fp = data_fingerprint(adjustment=Adjustment.NONE)
    assert fp["adj_events"] is None


def test_returns_invariant_with_adj_events_is_a_contradiction():
    with pytest.raises(ValueError, match="must NOT include an adj_factor event table"):
        data_fingerprint(
            adjustment="returns_invariant",
            adj_events={"000001.SZ": pd.Series([1.0], index=pd.to_datetime(["2024-01-02"]))},
        )


def test_price_level_requires_adj_events():
    with pytest.raises(ValueError, match="MUST include the per-symbol"):
        data_fingerprint(adjustment="price_level", adj_events=None)


def test_price_level_fingerprint_includes_per_symbol_adj_hash():
    events = {
        "000001.SZ": pd.Series([1.0, 1.1], index=pd.to_datetime(["2024-01-02", "2024-01-03"])),
        "600000.SH": pd.Series([2.0], index=pd.to_datetime(["2024-01-02"])),
    }
    fp = data_fingerprint(adjustment=Adjustment.PRICE_LEVEL, adj_events=events)
    assert set(fp["adj_events"]) == {"000001.SZ", "600000.SH"}
    assert fp["adj_events"]["000001.SZ"] == adj_events_hash(events["000001.SZ"])


def test_adj_events_hash_is_sensitive_to_the_anchor_level():
    # MUTATION (§六.17 qfq-anchor trap): perturbing the anchor VALUES changes the
    # hash even when the date grid is unchanged, so a re-based (ex-date) level can
    # never be silently reused for a price_level factor.
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    base = adj_events_hash(pd.Series([1.0, 1.0], index=idx))
    perturbed = adj_events_hash(pd.Series([1.0, 2.0], index=idx))
    assert base != perturbed


# --------------------------------------------------------------------------- #
# schema-version dimension mutation
# --------------------------------------------------------------------------- #
def test_schema_version_hash_changes_when_a_schema_module_changes(tmp_path, monkeypatch):
    import factors.store.fingerprint as fp_mod

    real = fp_mod._schema_module_files()
    base = fp_mod.schema_version_hash()
    copies: list[tuple[str, str]] = []
    for label, path in real:
        dst = tmp_path / (label.replace(".", "_") + ".py")
        dst.write_bytes(Path(path).read_bytes())
        copies.append((label, str(dst)))
    # mutate the first schema module's content
    first = Path(copies[0][1])
    first.write_bytes(first.read_bytes() + b"\n# schema drift\n")
    monkeypatch.setattr(fp_mod, "_schema_module_files", lambda: copies)
    assert fp_mod.schema_version_hash() != base


# --------------------------------------------------------------------------- #
# price_level FIXTURE factor (zero closing-set members is NOT waived, A5-F02)
# --------------------------------------------------------------------------- #
class _PriceLevelFixtureFactor(Factor):
    """A fixture factor declaring adjustment=price_level (no closing factor does)."""

    name = "fixture_price_level"
    spec = FactorSpec(
        factor_id="fixture_price_level",
        version="1.0",
        description="fixture: depends on the adjusted price LEVEL",
        expected_ic_sign=1,
        is_intraday=False,
        forward_return_horizon=1,
        return_basis="close_to_close",
        input_fields=("close",),
        requires=(PanelField("close", source="market_daily"),),
        adjustment="price_level",
        overnight_boundary="none",
    )

    def compute(self, panel: pd.DataFrame) -> pd.Series:  # pragma: no cover - fixture
        return panel["close"].rename(self.name)


def test_price_level_fixture_factor_fingerprint_uses_the_anchor():
    factor = _PriceLevelFixtureFactor()
    assert factor.spec.adjustment is Adjustment.PRICE_LEVEL
    events = {"000001.SZ": pd.Series([1.0], index=pd.to_datetime(["2024-01-02"]))}
    fp = data_fingerprint(adjustment=factor.spec.adjustment, adj_events=events)
    assert fp["adj_events"] == {"000001.SZ": adj_events_hash(events["000001.SZ"])}
    # omitting the anchor for a declared price_level factor is a loud error
    with pytest.raises(ValueError):
        data_fingerprint(adjustment=factor.spec.adjustment, adj_events=None)
