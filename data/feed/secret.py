"""Read the tushare token from the external secret file — never echoed.

Shared by the tushare-backed feeds so the dotted-key lookup + readable errors
live in one place. Error messages name the key path but NEVER include any value,
so a malformed config cannot leak the token.
"""

from __future__ import annotations

import json
from pathlib import Path


def lookup_dotted(data: dict, dotted_key: str) -> str:
    """Resolve a dotted key (e.g. ``'tushare.token'``) in a nested dict."""
    node: object = data
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(
                f"Secret config is missing key '{dotted_key}'. "
                f"Expected a nested entry reachable via that dotted path."
            )
        node = node[part]
    if not isinstance(node, str) or not node:
        raise ValueError(
            f"Secret config key '{dotted_key}' must map to a non-empty string token."
        )
    return node


def read_token(secret_file: str, token_key: str = "tushare.token") -> str:
    """Load and return the token from ``secret_file`` (json, dotted ``token_key``)."""
    path = Path(secret_file)
    if not path.exists():
        raise ValueError(
            f"Secret config file not found: {secret_file}. "
            f"Set data.external_secret_file to your .config.json path."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Secret config file is not valid JSON: {secret_file} ({exc.msg})."
        ) from None
    return lookup_dotted(data, token_key)
