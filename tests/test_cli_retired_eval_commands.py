"""The eleven per-factor `run-eval-*` commands collapsed into `run-factor-eval`.

Two things are pinned here. First, the eleven names really are gone from the
parser — a retirement that left them registered would be a rename, not a
convergence. Second, typing one of them still produces the migration route:
argparse answers a removed subcommand with ``invalid choice``, which tells an
operator they mistyped rather than that the command was replaced and by what.

The historical list of retired names lives in THIS file on purpose. The
production map is derived from ``FACTOR_TO_REPORT_NAME`` so there is no second
copy to drift; a literal list is still needed to assert that the derivation
produces exactly the names that used to exist, and a test is where a historical
claim belongs.
"""

from __future__ import annotations

import pytest

from qt.cli import build_parser, main, retired_eval_commands
from qt.factor_eval_reconcile import FACTOR_TO_REPORT_NAME

#: The eleven subcommands qt/cli.py registered before D5 C6 (git history).
RETIRED_NAMES = (
    "run-eval-jump-amount-corr",
    "run-eval-minute-ideal-amplitude",
    "run-eval-amp-marginal-anomaly-vol",
    "run-eval-volume-peak-count",
    "run-eval-intraday-amp-cut",
    "run-eval-peak-interval-kurtosis",
    "run-eval-valley-relative-vwap",
    "run-eval-valley-ridge-vwap-ratio",
    "run-eval-ridge-minute-return",
    "run-eval-valley-price-quantile",
    "run-eval-peak-ridge-amount-ratio",
)


def _registered_commands() -> set[str]:
    parser = build_parser()
    actions = [a for a in parser._actions if getattr(a, "choices", None)]
    return set(actions[0].choices)


def test_the_derived_map_is_exactly_the_eleven_names_that_used_to_exist():
    assert set(retired_eval_commands()) == set(RETIRED_NAMES)
    assert len(RETIRED_NAMES) == 11


def test_every_retired_name_maps_to_a_real_factor_id():
    mapping = retired_eval_commands()
    assert set(mapping.values()) == set(FACTOR_TO_REPORT_NAME)
    assert mapping["run-eval-valley-price-quantile"] == "valley_price_quantile_20"
    # the one whose command name is NOT its factor id with dashes
    assert mapping["run-eval-minute-ideal-amplitude"] == "minute_ideal_amp_10"


@pytest.mark.parametrize("name", RETIRED_NAMES)
def test_the_retired_name_is_no_longer_a_subcommand(name: str):
    assert name not in _registered_commands()


def test_the_unified_command_is_the_one_that_remains():
    commands = _registered_commands()
    assert "run-factor-eval" in commands
    assert not any(c.startswith("run-eval-") for c in commands)


@pytest.mark.parametrize("name", RETIRED_NAMES)
def test_a_retired_name_gets_the_migration_route_not_invalid_choice(name, capsys):
    assert main([name, "--config", "config/factor_eval_csi500.yaml"]) == 1
    err = capsys.readouterr().err
    assert "retired in D5 C6" in err
    assert "run-factor-eval" in err
    assert f"--factor {retired_eval_commands()[name]}" in err
    assert "invalid choice" not in err


def test_an_unknown_run_eval_name_still_falls_through_to_argparse(capsys):
    """The hint is for names that really existed. Anything else is a typo and
    should get argparse's list of valid choices, not a fabricated migration."""
    with pytest.raises(SystemExit):
        main(["run-eval-not-a-real-factor"])
    assert "invalid choice" in capsys.readouterr().err


def test_the_hint_does_not_hijack_an_unrelated_command(capsys):
    """Only the retired prefix is intercepted; everything else parses normally."""
    with pytest.raises(SystemExit):
        main(["run-factor-eval"])  # missing required --config/--factor
    assert "retired in D5 C6" not in capsys.readouterr().err
