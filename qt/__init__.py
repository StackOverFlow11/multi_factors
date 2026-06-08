"""qt: command-line entry + config models for the Phase 0 framework.

The CLI lives here (not in any business layer) so that ``data/``, ``factors/``,
``alpha/``, ``portfolio/`` and ``runtime/`` stay free of process orchestration.
"""

from __future__ import annotations

__version__ = "0.1.0"

from qt.config import RootConfig, load_config

__all__ = ["RootConfig", "load_config", "__version__"]
