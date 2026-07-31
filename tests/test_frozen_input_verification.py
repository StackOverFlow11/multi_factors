"""The C5 harness's frozen inputs are now checked before they are believed.

Three read sites used to open frozen artifacts with no integrity check at all,
while the directory holding them is demonstrably writable — one file in it was
overwritten on 2026-07-25. So "the harness reconciled against whatever was on
disk" was never hypothetical, and these tests pin the checks that close it:

* ``verify_frozen_panel_file`` — one panel against the hash authored in git,
  following the same jump -> PR-C fork the reconciler's path helper takes;
* ``verify_anchor_record`` — the hand-anchor record's bytes AND its inventory;
* ``check_overwrite_allowed`` — the write side, refusing to clobber that record
  without intent.

Every conviction test starts from a state that verifies GREEN, so a red result
is attributable to the tampering rather than to a check that is red on
everything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qt.hand_anchor_manifest import (
    MANIFEST_SCHEMA,
    AnchorRecordMismatch,
    record_inventory,
    verify_anchor_record,
)
from qt.hand_anchors_d2 import (
    PRODUCED_RECORD_KEYS,
    RecordOverwriteRefused,
    check_overwrite_allowed,
)
from qt.panel_freeze import (
    D1_MANIFEST_DOC,
    PR_C_MANIFEST_DOC,
    PR_C_SUBDIR,
    FrozenPanelMismatch,
    canonical_content_hash,
    doc_for_frozen_panel,
    verify_frozen_panel_file,
)
from tests.fixtures.frozen_baseline import build_frozen_tree, make_panel, rewrite_panel

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# A/B — per-panel verification at the harness read sites
# --------------------------------------------------------------------------- #
def test_a_frozen_panel_that_matches_git_verifies(tmp_path: Path):
    doc = build_frozen_tree(tmp_path)
    panel_path = tmp_path / "panels" / "alpha_20.parquet"
    expected = canonical_content_hash(make_panel("alpha_20"))
    assert verify_frozen_panel_file(panel_path, "alpha_20", doc_path=doc) == expected


def test_a_drifted_panel_is_refused_at_the_read_site(tmp_path: Path):
    doc = build_frozen_tree(tmp_path)
    panel_path = tmp_path / "panels" / "alpha_20.parquet"
    assert verify_frozen_panel_file(panel_path, "alpha_20", doc_path=doc)  # green

    panel = make_panel("alpha_20")
    panel.iloc[0] = float(panel.iloc[0]) + 1.0
    rewrite_panel(tmp_path, "alpha_20", panel)
    with pytest.raises(FrozenPanelMismatch, match="canonical content hash"):
        verify_frozen_panel_file(panel_path, "alpha_20", doc_path=doc)


def test_a_missing_panel_is_refused_rather_than_skipped(tmp_path: Path):
    doc = build_frozen_tree(tmp_path)
    (tmp_path / "panels" / "alpha_20.parquet").unlink()
    with pytest.raises(FrozenPanelMismatch, match="missing from disk"):
        verify_frozen_panel_file(
            tmp_path / "panels" / "alpha_20.parquet", "alpha_20", doc_path=doc
        )


def test_a_factor_absent_from_the_document_is_refused(tmp_path: Path):
    """No expectation in git means nothing to verify against — which must be a
    refusal, not a pass."""
    doc = build_frozen_tree(tmp_path)
    from qt.panel_freeze import atomic_write_parquet

    atomic_write_parquet(make_panel("stranger_20"), tmp_path / "panels" / "stranger_20.parquet")
    with pytest.raises(FrozenPanelMismatch, match="no expectation"):
        verify_frozen_panel_file(
            tmp_path / "panels" / "stranger_20.parquet", "stranger_20", doc_path=doc
        )


def test_the_jump_fork_is_read_off_the_path_not_the_factor_id():
    """Both trees hold a ``jump_amount_corr_20.parquet`` and they are DIFFERENT
    factor definitions (v1.0 untruncated vs v1.1 truncated). Verification has to
    follow the same fork the reconciler's path helper takes, or it checks a panel
    against the other panel's hash."""
    d1 = doc_for_frozen_panel(REPO / "artifacts/refactor_baseline/panels/x.parquet", REPO)
    prc = doc_for_frozen_panel(
        REPO / f"artifacts/refactor_baseline/{PR_C_SUBDIR}/panels/x.parquet", REPO
    )
    assert d1 == REPO / D1_MANIFEST_DOC
    assert prc == REPO / PR_C_MANIFEST_DOC
    assert d1 != prc


def test_the_reconciler_routes_jump_the_same_way_verification_does():
    """Coupling: if frozen_panel_path ever stops routing jump to the PR-C tree,
    or verification stops following it, the pair must not drift silently."""
    from qt.factor_eval_reconcile import frozen_panel_path

    jump = frozen_panel_path("jump_amount_corr_20", REPO)
    other = frozen_panel_path("volume_peak_count_20", REPO)
    assert doc_for_frozen_panel(jump, REPO) == REPO / PR_C_MANIFEST_DOC
    assert doc_for_frozen_panel(other, REPO) == REPO / D1_MANIFEST_DOC


# --------------------------------------------------------------------------- #
# C-1 — the hand-anchor record
# --------------------------------------------------------------------------- #
def _record(tmp_path: Path, **overrides) -> tuple[Path, Path]:
    payload = {
        "seed": 1,
        "tolerance": 1e-12,
        "frozen14": [{"factor_id": "a", "ok": True}],
        "daily_pending_engine": [{"factor_id": "momentum_20"}],
        "all_ok_frozen14": True,
    }
    payload.update(overrides)
    record = tmp_path / "hand_anchors_d2.json"
    record.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    import hashlib

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
                "contents": record_inventory(payload),
            }
        ),
        encoding="utf-8",
    )
    return record, manifest


def test_an_untouched_record_verifies(tmp_path: Path):
    record, manifest = _record(tmp_path)
    check = verify_anchor_record(record, manifest)
    assert check.inventory["frozen14_rows"] == 1
    assert check.inventory["daily_engine_compared_rows"] == 0


def test_an_edited_record_is_refused(tmp_path: Path):
    record, manifest = _record(tmp_path)
    assert verify_anchor_record(record, manifest)  # green control
    payload = json.loads(record.read_text())
    payload["frozen14"][0]["ok"] = False  # flip a verdict
    record.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(AnchorRecordMismatch, match="sha256"):
        verify_anchor_record(record, manifest)


def test_a_manifest_updated_without_noticing_the_contents_change_is_refused(tmp_path: Path):
    """The sha alone cannot see this: someone reruns the tool, updates the sha to
    match, and the record silently loses rows. The inventory is what catches it,
    which is why it is asserted rather than merely written down."""
    record, manifest = _record(tmp_path, daily_engine_compared=[{"ok": True}])
    assert verify_anchor_record(record, manifest)  # green, 1 compared row

    payload = json.loads(record.read_text())
    payload["daily_engine_compared"] = []  # the 2026-07-25 shape
    record.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    import hashlib

    stale = json.loads(manifest.read_text())
    stale["sha256"] = hashlib.sha256(record.read_bytes()).hexdigest()  # sha "fixed"
    manifest.write_text(json.dumps(stale), encoding="utf-8")

    with pytest.raises(AnchorRecordMismatch, match="daily_engine_compared_rows"):
        verify_anchor_record(record, manifest)


def test_a_missing_manifest_is_refused_not_skipped(tmp_path: Path):
    record, _ = _record(tmp_path)
    with pytest.raises(AnchorRecordMismatch, match="cannot be verified"):
        verify_anchor_record(record, tmp_path / "nope.json")


def test_the_real_record_matches_its_committed_manifest():
    """Coupling to reality: the pinned sha must be the bytes actually on disk, or
    every anchors run fails."""
    record = REPO / "artifacts/refactor_baseline/hand_anchors_d2.json"
    manifest = REPO / "docs/factors/d5_hand_anchor_record_manifest.json"
    if not record.exists():  # gitignored bulk tree
        pytest.skip("frozen baseline not present in this checkout")
    check = verify_anchor_record(record, manifest)
    assert check.inventory["frozen14_rows"] == 70
    assert check.inventory["daily_pending_engine_rows"] == 20


def test_the_committed_manifest_states_the_record_is_incomplete():
    """Pinning an incomplete record is fine; letting a later reader take "pinned"
    for "complete" is not. The shortfall must be stated in the manifest."""
    manifest = json.loads(
        (REPO / "docs/factors/d5_hand_anchor_record_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["contents"]["daily_engine_compared_rows"] == 0
    prose = manifest["what_this_record_does_NOT_contain"]
    assert "daily_engine_compared" in prose and "EMPTY" in prose
    assert "cannot be completed" in prose
    # and it must say the D2 conclusion itself survived
    assert "88 hand anchors" in prose


# --------------------------------------------------------------------------- #
# D-1 — the write side
# --------------------------------------------------------------------------- #
def test_writing_a_fresh_record_is_not_refused(tmp_path: Path):
    """The tool must still be able to do its job."""
    check_overwrite_allowed(
        tmp_path / "absent.json", PRODUCED_RECORD_KEYS, allow_overwrite=False
    )


def test_overwriting_an_existing_record_is_refused_by_default(tmp_path: Path):
    target = tmp_path / "rec.json"
    target.write_text(json.dumps({k: 1 for k in PRODUCED_RECORD_KEYS}), encoding="utf-8")
    with pytest.raises(RecordOverwriteRefused, match="refusing to overwrite"):
        check_overwrite_allowed(target, PRODUCED_RECORD_KEYS, allow_overwrite=False)


def test_the_refusal_names_the_keys_this_run_would_drop(tmp_path: Path):
    """Computed by DIFFERENCE against what the run produces, so a future key is
    protected without anyone remembering to add it to a list."""
    target = tmp_path / "rec.json"
    payload = {k: 1 for k in PRODUCED_RECORD_KEYS}
    payload["daily_engine_compared"] = [{"ok": True}]
    payload["some_future_key"] = 1
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RecordOverwriteRefused) as caught:
        check_overwrite_allowed(target, PRODUCED_RECORD_KEYS, allow_overwrite=False)
    message = str(caught.value)
    assert "daily_engine_compared" in message and "some_future_key" in message
    assert "CANNOT be rebuilt" in message
    assert "d5_hand_anchor_record_manifest.json" in message  # what to do after


def test_explicit_intent_is_honoured(tmp_path: Path):
    target = tmp_path / "rec.json"
    target.write_text(json.dumps({"daily_engine_compared": []}), encoding="utf-8")
    check_overwrite_allowed(target, PRODUCED_RECORD_KEYS, allow_overwrite=True)


def test_an_unreadable_existing_record_still_refuses(tmp_path: Path):
    """Corrupt JSON must not be read as "nothing at risk, go ahead"."""
    target = tmp_path / "rec.json"
    target.write_text("{not json", encoding="utf-8")
    with pytest.raises(RecordOverwriteRefused):
        check_overwrite_allowed(target, PRODUCED_RECORD_KEYS, allow_overwrite=False)


def test_the_declared_key_set_is_what_the_writer_actually_produces():
    """The guard reasons about PRODUCED_RECORD_KEYS before the work runs, so the
    writer asserts its payload equals that set. Pin both halves here."""
    source = (REPO / "qt" / "hand_anchor_rows.py").read_text(encoding="utf-8")
    assert "PRODUCED_RECORD_KEYS" in source
    assert "payload keys drifted" in source
    assert PRODUCED_RECORD_KEYS == frozenset(
        {"seed", "tolerance", "elapsed_seconds", "frozen14",
         "daily_pending_engine", "all_ok_frozen14"}
    )


def test_the_verified_path_is_the_only_gateway_to_a_frozen_panel():
    """AST, not grep: every call to ``frozen_panel_path`` in the reconciler must
    sit inside ``verified_frozen_panel_path``.

    This is the property that makes A/B real. Checking "the read site calls
    something verified" per site turns "did we verify this read?" into a
    per-site question, and the site that forgets is precisely the one that will
    not announce itself. With one gateway there is nothing to forget.

    Range: it proves no OTHER site resolves a frozen panel path, not that the
    verifier's own body is correct — the tests above cover that.
    """
    import ast

    source = (REPO / "qt" / "factor_eval_reconcile.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    gateway = "verified_frozen_panel_path"

    def enclosing(node: ast.AST) -> str | None:
        for func in ast.walk(tree):
            if isinstance(func, ast.FunctionDef) and any(
                inner is node for inner in ast.walk(func)
            ):
                return func.name
        return None

    offenders = [
        enclosing(call)
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "frozen_panel_path"
        and enclosing(call) != gateway
    ]
    assert not offenders, (
        "these resolve a frozen panel path without the verifying gateway: "
        f"{offenders}"
    )
    # anti-vacuity: the gateway itself must actually make the call
    assert f"def {gateway}" in source and "frozen_panel_path(factor_id, repo_root)" in source


def test_both_frozen_panel_reads_go_through_the_gateway():
    """Companion to the above: the two read sites must still READ something, so
    the gateway is not merely unused."""
    import ast

    source = (REPO / "qt" / "factor_eval_reconcile.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for name in ("run_panels_mode", "run_anchors_mode"):
        func = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
        )
        body = ast.get_source_segment(source, func) or ""
        assert "read_parquet" in body, f"{name} no longer reads a panel"
        assert "verified_frozen_panel_path" in body, f"{name} bypasses the gateway"


# --------------------------------------------------------------------------- #
# The anchors read site: that the check is WIRED, not merely that it works
# --------------------------------------------------------------------------- #
def _anchor_repo(tmp_path: Path, *, tamper: bool) -> Path:
    """A miniature repo_root holding the record and its git-tracked manifest.

    Built through the production inventory helper, so the manifest states what
    the record actually contains — then optionally tampered AFTER the manifest
    is written, which is the real-world shape (bytes drift, expectation does
    not).
    """
    import hashlib

    from qt.factor_eval_reconcile import ANCHORS_JSON
    from qt.hand_anchor_manifest import DEFAULT_MANIFEST

    payload = {
        "seed": 1,
        "tolerance": 1e-12,
        "frozen14": [
            {"factor_id": "alpha_20", "date": "2024-01-02", "symbol": "000001.SZ",
             "hand": 1.0, "engine": 1.0, "rel_diff": 0.0, "ok": True},
        ],
        "daily_pending_engine": [],
        "all_ok_frozen14": True,
    }
    record = tmp_path / ANCHORS_JSON
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    manifest = tmp_path / DEFAULT_MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
                "contents": record_inventory(payload),
            }
        ),
        encoding="utf-8",
    )
    if tamper:
        payload["frozen14"][0]["ok"] = False  # flip a verdict after pinning
        record.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return tmp_path


def test_the_anchors_read_site_returns_rows_when_the_record_is_intact(tmp_path: Path):
    """GREEN CONTROL. Without it, the refusal test below could pass simply
    because load_anchor_rows raises on everything."""
    from qt.factor_eval_reconcile import load_anchor_rows

    rows = load_anchor_rows("alpha_20", _anchor_repo(tmp_path, tamper=False))
    assert len(rows) == 1 and rows[0]["symbol"] == "000001.SZ"


def test_the_anchors_read_site_refuses_a_record_that_disagrees_with_its_manifest(
    tmp_path: Path,
):
    """BEHAVIOURAL, on purpose — this pins the WIRING, which is what was missing.

    The verifier itself was thoroughly tested while nothing asserted that
    ``load_anchor_rows`` calls it: replacing that call with ``pass`` left the
    whole suite green. That is the identical shape as the first A/B mutation
    (function pinned, wiring not), surviving in the third read site after the
    other two were fixed.

    An AST assertion would only show the call is written down. A call with the
    wrong argument, or one whose exception is swallowed, passes AST and fails
    here — so the load-bearing test is the one that actually feeds the function
    a bad record and demands a refusal.
    """
    from qt.factor_eval_reconcile import ReconciliationError, load_anchor_rows

    with pytest.raises(ReconciliationError, match="sha256"):
        load_anchor_rows("alpha_20", _anchor_repo(tmp_path, tamper=True))


def test_the_anchors_read_site_refuses_before_it_parses(tmp_path: Path):
    """Order matters: a corrupt record must be refused for the RIGHT reason.
    If parsing came first, a tampered-and-unparseable record would surface as a
    JSON error and the integrity failure would never be named."""
    from qt.factor_eval_reconcile import ANCHORS_JSON, ReconciliationError, load_anchor_rows

    repo = _anchor_repo(tmp_path, tamper=False)
    (repo / ANCHORS_JSON).write_text("{ not json at all", encoding="utf-8")
    with pytest.raises(ReconciliationError, match="sha256"):  # not a JSONDecodeError
        load_anchor_rows("alpha_20", repo)


def test_a_new_key_cannot_appear_without_the_manifest_noticing(tmp_path: Path):
    """LOW-1: counts alone cannot see a key APPEAR.

    The reviewer added ``all_ok_daily: true`` to the record, refreshed the sha,
    and it was accepted — while the manifest's most prominent prose says exactly
    that key is absent. The key SET is now compared, so "pinned is not complete"
    no longer rests on prose.
    """
    import hashlib

    record, manifest = _record(tmp_path)
    assert verify_anchor_record(record, manifest)  # green control

    payload = json.loads(record.read_text())
    payload["all_ok_daily"] = True  # the key the manifest says is ABSENT
    record.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    refreshed = json.loads(manifest.read_text())
    refreshed["sha256"] = hashlib.sha256(record.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(refreshed), encoding="utf-8")

    with pytest.raises(AnchorRecordMismatch, match="top_level_keys"):
        verify_anchor_record(record, manifest)
