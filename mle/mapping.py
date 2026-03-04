from __future__ import annotations

from typing import Literal, Optional

import numpy as np

KindStr = Literal["L", "C", "M"]
SideStr = Literal["ask", "bid"]

# LOBSTER event type mapping
# 1=new limit, 2=cancel partial, 3=delete total, 4=exec visible, 5=exec hidden, 7=halt
LOBSTER_KIND = {
    1: "L",
    2: "C",
    3: "C",
    4: "M",
    5: "M",
}


def event_kind_from_type(t: int) -> Optional[KindStr]:
    """Return 'L','C','M' or None for unsupported types (e.g. halts)."""
    return LOBSTER_KIND.get(int(t), None)


def affected_side(kind: KindStr, direction: int) -> SideStr:
    """
    direction in LOBSTER:
        +1 = buy, -1 = sell

    - For L/C: direction indicates the side of the resting order
    - For M: direction indicates aggressor; removes opposite side
    """
    d = int(direction)
    if kind in ("L", "C"):
        return "bid" if d == 1 else "ask"
    else:  # "M"
        return "ask" if d == 1 else "bid"


def make_ob_cols(K: int):
    ask_px = [f"ask_px_{i}" for i in range(1, K + 1)]
    ask_sz = [f"ask_sz_{i}" for i in range(1, K + 1)]
    bid_px = [f"bid_px_{i}" for i in range(1, K + 1)]
    bid_sz = [f"bid_sz_{i}" for i in range(1, K + 1)]
    return ask_px, ask_sz, bid_px, bid_sz


def find_level_index(price: int, pre_px_row: np.ndarray) -> int:
    """
    Return i in {1..K} if price matches the pre-book level i; else 0.
    pre_px_row must have shape (K,)
    """
    px = int(price)
    for j in range(pre_px_row.shape[0]):
        if int(pre_px_row[j]) == px:
            return j + 1
    return 0