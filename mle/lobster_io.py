from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# LOBSTER message columns (per sample readme)
MSG_COLS = ["time", "type", "order_id", "size", "price", "direction"]


def make_orderbook_cols(levels: int) -> list[str]:
    cols = []
    for lvl in range(1, levels + 1):
        cols += [f"ask_px_{lvl}", f"ask_sz_{lvl}", f"bid_px_{lvl}", f"bid_sz_{lvl}"]
    return cols


def load_lobster_day(message_csv: str | Path, orderbook_csv: str | Path, levels: int):
    """
    Load one LOBSTER day (message + orderbook), returning:
        msg_df, ob_df, df_concat

    Notes:
    - price is in $ * 10000 (int)
    - rows are aligned: row k in msg corresponds to row k in ob
    """
    message_csv = Path(message_csv)
    orderbook_csv = Path(orderbook_csv)

    msg = pd.read_csv(message_csv, header=None, names=MSG_COLS)
    ob = pd.read_csv(orderbook_csv, header=None, names=make_orderbook_cols(levels))

    if len(msg) != len(ob):
        raise ValueError(f"Row mismatch: message={len(msg)}, orderbook={len(ob)}")

    # Basic typing
    msg["time"] = msg["time"].astype(float)
    msg["type"] = msg["type"].astype(int)
    msg["order_id"] = msg["order_id"].astype(np.int64)
    msg["size"] = msg["size"].astype(np.int64)
    msg["price"] = msg["price"].astype(np.int64)
    msg["direction"] = msg["direction"].astype(int)

    # Useful convenience
    msg["price_dollars"] = msg["price"] / 10000.0

    df = pd.concat([msg, ob], axis=1)
    return msg, ob, df