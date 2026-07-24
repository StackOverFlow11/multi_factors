"""``factors.compute.minute``: the minute-factor engine (D2, design v3.2 §3.2).

One factor per file, every file a small pure-function layer on top of
:mod:`factors.compute.minute.primitives` (the shared visible-bar preparation +
peak/ridge/valley taxonomy + trailing-window machinery). The ``data.clean``
intraday factor modules are re-export shims of these files until D6d deletes
them (§六.4: the single definition point lives HERE from D2 on).
"""
