# Scheduling the all-A post-market cache warm (systemd user timer)

The daily 21:00 (Asia/Shanghai) all-A incremental cache warm is run by
`python -m qt.cli data-update --config config/data_update_all_a.yaml`. It ONLY
warms the read-through tushare caches (no factors / alpha / portfolio / backtest /
PanelStore). This doc installs it as a **systemd user timer**.

The unit files live in the repo at `deploy/systemd/`:

- `quant-data-update.service` — the `Type=oneshot` job (`ExecStart` = the CLI). It
  has no `[Install]` section on purpose; it is triggered only by the timer.
- `quant-data-update.timer` — `OnCalendar=*-*-* 21:00:00 Asia/Shanghai`, `Persistent=true`.

They are **artifacts, not installed by the repo**. Nothing is scheduled until you
copy them into `~/.config/systemd/user/` and enable the **timer** (not the
service) yourself.

**systemd version requirement:** the `Asia/Shanghai` timezone suffix in
`OnCalendar` needs **systemd ≥ 239**. This host runs 255 (fine). On an older host
without timezone-suffix support, either upgrade systemd or set the service
environment `TZ=Asia/Shanghai` and use a bare `OnCalendar=*-*-* 21:00:00`.

---

## ⚠️ LOUD PREFLIGHT — manual run FIRST, schedule SECOND

**Enable the timer ONLY AFTER a manual, observed `data-update` run succeeds.**
The first all-A live warm fetches the whole listed market (~5500 symbols) and is
the one most likely to hit rate limits, partial coverage, or a stale token. It
must be user-driven and watched — never silently scheduled as its debut.

Run it by hand and read the summary before you touch systemd:

```bash
cd /home/shaofl/Projects/financial_projects/stocks_market/Quantitative_Trading
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli \
  data-update --config config/data_update_all_a.yaml
```

Preflight checklist (all must hold before enabling the timer):

- [ ] The external secret file exists and holds a valid token:
      `/home/shaofl/Projects/financial_projects/.config.json` (key `tushare.token`).
      It is OUTSIDE the repo and is referenced by the config, never embedded in the
      unit. The token value must never appear in a unit file or a log.
- [ ] `validate-config --config config/data_update_all_a.yaml` passes.
- [ ] A **manual** `data-update` run (above) completes and prints an
      `OK data-update: … endpoints, … symbols` summary with a per-endpoint
      `requests / rows_written / not_ready` breakdown that looks sane.
- [ ] A **second** manual run *the same day* is mostly warm (the already-stored
      range makes ~0 new requests) — confirming the incremental read-through works
      before you automate. (Note the front-edge cost in "Steady-state cost" below:
      a *next-day* run genuinely refetches the new day per symbol.)
- [ ] The env python path in `quant-data-update.service` still exists:
      `/home/shaofl/Development/env_tools/envs/quant_mf/bin/python`.

Only when every box is checked, proceed to install.

---

## Install & enable (systemd USER units)

```bash
# 1. Copy the unit files into the user systemd directory.
mkdir -p ~/.config/systemd/user
cp /home/shaofl/Projects/financial_projects/stocks_market/Quantitative_Trading/deploy/systemd/quant-data-update.service ~/.config/systemd/user/
cp /home/shaofl/Projects/financial_projects/stocks_market/Quantitative_Trading/deploy/systemd/quant-data-update.timer   ~/.config/systemd/user/

# 2. Reload the user manager so it sees the new units.
systemctl --user daemon-reload

# 3. Enable + start the TIMER (not the service directly).
systemctl --user enable --now quant-data-update.timer

# 4. (Optional) let user services run without an active login session, so the
#    21:00 job fires even when you are not logged in:
#    sudo loginctl enable-linger "$USER"
```

## Verify & inspect

```bash
# Next scheduled fire and last trigger:
systemctl --user list-timers quant-data-update.timer

# Service state after a run:
systemctl --user status quant-data-update.service

# Trigger a one-off run immediately (same as the manual CLI run, via systemd):
systemctl --user start quant-data-update.service

# Logs for the last run (follow live with -f):
journalctl --user -u quant-data-update.service -n 200 --no-pager
journalctl --user -u quant-data-update.service -f
```

## Disable / remove

```bash
systemctl --user disable --now quant-data-update.timer
rm ~/.config/systemd/user/quant-data-update.timer ~/.config/systemd/user/quant-data-update.service
systemctl --user daemon-reload
```

---

## Steady-state cost (read before scheduling — NOT "≈ free")

The incremental semantics live entirely in the cache layer, but "incremental" here
does **not** mean "≈ 0 calls per night" at all-A scale. Because `lookback_days` is
relative to *today*, the requested window shifts forward one day every day, so a
fully-covered universe still has a genuine new ~1-trading-day gap per symbol per
dense endpoint **every night, indefinitely**. Order of magnitude at all-A scale:

- ~5500 symbols × ~6 dense endpoints ≈ **tens of thousands of API calls per night**
  (~30k order of magnitude), i.e. **~1+ hour of API-bound work every night**.
- That is comfortably inside the service `TimeoutStartSec=4h`, but it is NOT free —
  budget for a real, hour-plus nightly job.
- The part that IS ~0 calls is the already-stored *back* range (dates you already
  have); only the moving front edge and the recent-tail refetch cost calls. History
  is **not** backfilled by this job (that is a separate manual run / PR-2).

## ⚠️ No per-symbol failure isolation yet (PR-1 limitation)

`update_endpoints` currently has **no per-endpoint / per-symbol failure isolation**:
if a single symbol's fetch fails persistently (after the built-in retries), it
**raises and aborts every endpoint listed AFTER it for that night's run**. At ~5500
symbols this becomes statistically likely on any given night. Important properties:

- It is **delayed, not data-losing**: gaps already fetched-and-stored that night stay
  durable; a failed fetch records **no** coverage, so it is simply retried next time.
- **Re-running resumes**: `systemctl --user start quant-data-update.service` (or the
  next timer fire) skips durable gaps and only refetches what is still missing.
- The **mandatory manual observed run** in the preflight above is exactly where you
  first see this, before it is ever scheduled.
- **Per-symbol failure isolation is a planned PR-2 hardening** (changing the fail-fast
  behavior is a warm-path semantic change, deliberately out of scope for PR-1).

## Notes

- Scope is set by `data_update.universe_scope: all_a` in the config; the default
  (`config`) would warm only the config universe. The setting never leaks into the
  backtest universe.
- `index_weight` is pre-warmed only for the project's recurring backtest universes
  (SSE50 / CSI300 / CSI500, via `data_update.index_codes`); a backtest against any
  other index still warms its own `index_weight` lazily via read-through.
- Do not run `systemctl` from automation. These steps are for a human operator who
  has completed the preflight above.
