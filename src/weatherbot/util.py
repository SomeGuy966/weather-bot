"""Small typed helpers shared across modules."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """``df.to_dict("records")`` with string keys, for loops that read a few columns per row."""
    return cast(list[dict[str, Any]], df.to_dict("records"))
