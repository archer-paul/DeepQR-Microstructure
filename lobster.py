import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Literal

MSG_COLS = ["time", "type", "order_id", "size", "price", "direction"]
LOBSTER_KIND = {1: "L", 2: "C", 3: "C", 4: "M", 5: "M"}

KindStr = Literal["L", "C", "M"]
SideStr = Literal["ask", "bid"]


def make_orderbook_cols(levels: int) -> list[str]:
    cols=[]
    for lvl in range(1, levels+1):
        cols += [f"ask_px_{lvl}", f"ask_sz_{lvl}", f"bid_px_{lvl}", f"bid_sz_{lvl}"]
    return cols


def event_kind_from_type(t: int) -> Optional[KindStr]:
    return LOBSTER_KIND.get(int(t), None)


def affected_side(kind: KindStr, direction: int) -> SideStr:
    d = int(direction)
    if kind in ("L", "C"):
        return "bid" if d == 1 else "ask"
    else:
        return "ask" if d == 1 else "bid"


def infer_tick_size(df: pd.DataFrame, levels: int) -> int:
    ask = df[[f"ask_px_{i}" for i in range(1, levels+1)]].to_numpy(np.int64)
    bid = df[[f"bid_px_{i}" for i in range(1, levels+1)]].to_numpy(np.int64)
    
    diffs = []
    diffs.append(np.diff(ask, axis=1).ravel())
    diffs.append(np.diff(bid[:, ::-1], axis=1).ravel())
    d = np.concatenate(diffs)
    d = d[d > 0]
    if len(d) == 0:
        raise ValueError("Cannot infer tick size (no positive price difference found).")
    return int(d.min())


def compute_pref_mid(df: pd.DataFrame) -> np.ndarray:
    best_ask = df["ask_px_1"].to_numpy(np.int64)
    best_bid = df["bid_px_1"].to_numpy(np.int64)
    return (best_ask+best_bid) / 2.0


def estimate_pref_paper(
    best_bid: np.ndarray,
    best_ask: np.ndarray,
    tick_size: int,
) -> np.ndarray:
    """
    Implements the rule in the paper (your screenshot):
    - Let spread_ticks = (best_ask - best_bid)/tick_size (integer).
    - If spread_ticks is odd: p_ref = p_mid = (best_bid + best_ask)/2.
    - If spread_ticks is even: p_ref = p_mid +/- tick_size/2, choose the one closest to previous p_ref.
    Returns p_ref as float (can be half-tick).
    """
    best_bid = best_bid.astype(np.int64)
    best_ask = best_ask.astype(np.int64)

    spread = best_ask - best_bid
    if np.any(spread < 0):
        raise ValueError("Found best_ask < best_bid.")
    if np.any(spread % tick_size != 0):
        raise ValueError("Spread not multiple of tick_size; check tick_size / units.")

    spread_ticks = (spread // tick_size).astype(np.int64)
    p_mid = (best_bid + best_ask) / 2.0  # may be half-tick when spread is odd

    pref = np.empty_like(p_mid, dtype=np.float64)
    half_tick = tick_size / 2.0

    # first value: natural choice p_mid (paper doesn't specify init; this is standard)
    pref[0] = p_mid[0]

    for t in range(1, len(p_mid)):
        if spread_ticks[t] % 2 == 1:
            # odd spread in tick units -> use mid
            pref[t] = p_mid[t]
        else:
            # even spread -> choose pmid +/- half_tick closest to previous pref
            cand1 = p_mid[t] + half_tick
            cand2 = p_mid[t] - half_tick
            pref[t] = cand1 if abs(cand1 - pref[t-1]) <= abs(cand2 - pref[t-1]) else cand2

    # handle t=0 with the same rule if you want strict consistency:
    if spread_ticks[0] % 2 == 1:
        pref[0] = p_mid[0]
    else:
        # for the first observation, pick +half_tick by convention (or -half_tick)
        # Here: pick the side that keeps pref on the "ask side" of mid (arbitrary but consistent)
        pref[0] = p_mid[0] + half_tick

    return pref


def regrid_to_qr_levels(df: pd.DataFrame, levels: int, K: int, pref: np.ndarray | None = None, tick_size: int | None = None) -> pd.DataFrame:
    
    if tick_size is None:
        tick_size = infer_tick_size(df, levels)
        
    if pref is None:
        pref = compute_pref_mid(df)
        
    ask_px = df[[f"ask_px_{j}" for j in range(1, levels+1)]].to_numpy(np.int64)
    ask_sz = df[[f"ask_sz_{j}" for j in range(1, levels+1)]].to_numpy(np.int64)
    bid_px = df[[f"bid_px_{j}" for j in range(1, levels+1)]].to_numpy(np.int64)
    bid_sz = df[[f"bid_sz_{j}" for j in range(1, levels+1)]].to_numpy(np.int64)
    
    n = len(df)
    out = pd.DataFrame(index=df.index)
    out["pref"] = pref
    out["tick_size"] = tick_size
    
    half_tick = tick_size / 2.0
    price_to_half = lambda x: np.rint(x / half_tick).astype(np.int64)
    
    pref_half = price_to_half(pref)
    ask_px_half = price_to_half(ask_px)
    bid_px_half = price_to_half(bid_px)
    
    for i in range(1, K+1):
        target_ask_half = pref_half + (2*i - 1)
        target_bid_half = pref_half - (2*i - 1)

        # match sur les colonnes existantes; sinon 0
        out[f"P_{i}"] = pref + (i-0.5)*tick_size
        m_ask = (ask_px_half == target_ask_half[:, None])
        q_ask = np.where(m_ask, ask_sz, 0).sum(axis=1)  # un seul match attendu
        out[f"Q_{i}"] = q_ask

        out[f"P_-{i}"] = pref - (i-0.5)*tick_size
        m_bid = (bid_px_half == target_bid_half[:, None])
        q_bid = np.where(m_bid, bid_sz, 0).sum(axis=1)
        out[f"Q_-{i}"] = q_bid

    return out


def price_level_from_qr(msg: pd.DataFrame, qr: pd.DataFrame, K: int) -> pd.Series:
    """
    qr[k] is post-event grid -> use qr_pre = qr.shift(1) for event k.
    L/C: match msg.price to P_{±i} in qr_pre on the resting side.
    M: use best opposite quote => level is +/-1 by aggressor direction.
    """
    qr_pre = qr.shift(0)

    kind = msg["type"].to_numpy()                 # "L","C","M" or None
    direction = msg["direction"].to_numpy(np.int64)
    price = msg["price"].to_numpy(np.int64)

    lvl = np.zeros(len(msg), dtype=np.int64)
    lvl[0] = 0  # no pre snapshot

    # --- Market orders: best opposite level ---
    is_M = (kind == "M")
    # LOBSTER convention: direction=+1 buy, -1 sell (aggressor for M)
    lvl[is_M & (direction == 1)] = -1
    lvl[is_M & (direction != 1)] = +1

    # --- Limit/Cancel: match on resting side prices in pre-grid ---
    is_LC = np.isin(kind, ["L", "C"])

    # resting side for L/C
    side_is_bid = (direction == 1)  # L/C: +1 means bid side, -1 ask side

    # pre-grid price matrices
    P_ask = np.column_stack([np.rint(qr_pre[f"P_{i}"].to_numpy(np.float64)).astype(np.int64) for i in range(1, K+1)])
    P_bid = np.column_stack([np.rint(qr_pre[f"P_-{i}"].to_numpy(np.float64)).astype(np.int64) for i in range(1, K+1)])

    m_ask = (P_ask == price[:, None])
    m_bid = (P_bid == price[:, None])

    ask_idx = np.where(m_ask.any(axis=1), m_ask.argmax(axis=1) + 1, 0)  # 1..K else 0
    bid_idx = np.where(m_bid.any(axis=1), m_bid.argmax(axis=1) + 1, 0)

    # assign only for L/C rows
    mask_lc_ask = is_LC & (~side_is_bid) & (ask_idx > 0)
    mask_lc_bid = is_LC & (side_is_bid) & (bid_idx > 0)

    lvl[mask_lc_ask] = ask_idx[mask_lc_ask]
    lvl[mask_lc_bid] = -bid_idx[mask_lc_bid]

    # keep lvl=0 when no match (outside +/-K or first row)
    lvl[0] = 0
    return pd.Series(lvl, index=msg.index, name="price_level")


def load_lobster_data(
    message_csv: str | Path,
    orderbook_csv: str | Path,
    levels: int,
    K: int,
    market_open_seconds: float = 3600 * 9.5,
    tick_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    message_csv = Path(message_csv)
    orderbook_csv = Path(orderbook_csv)
    
    # original dataframes
    # Prevent column shifting by explicitly reading only the first 6 columns for msg 
    # and the first 4*levels columns for orderbook
    msg = pd.read_csv(message_csv, header=None, names=MSG_COLS, usecols=[0, 1, 2, 3, 4, 5], low_memory=False)
    ob = pd.read_csv(orderbook_csv, header=None, names=make_orderbook_cols(levels), usecols=range(4 * levels), low_memory=False)
    
    if len(msg) != len(ob):
        raise ValueError(f"Row mismatch: message={len(msg)}, orderbook={len(ob)}")
    
    # types
    msg["time"] = msg["time"].astype(float)
    msg["type"] = msg["type"].fillna(0).astype(int)
    msg["size"] = msg["size"].fillna(0).astype(np.int64)
    msg["price"] = msg["price"].fillna(0).astype(np.int64)
    msg["direction"] = msg["direction"].fillna(0).astype(int)
    
    # map types -> L/C/M
    msg["type"] = msg["type"].apply(event_kind_from_type)
    
    # tick size
    if tick_size is None:
        tick_size = infer_tick_size(ob, levels=levels)
    
    # p_ref as in paper
    best_bid = ob["bid_px_1"].to_numpy(np.int64)
    best_ask = ob["ask_px_1"].to_numpy(np.int64)
    pref = estimate_pref_paper(best_bid, best_ask, tick_size)
    
    # mapping from order book to order book model grid
    qr = regrid_to_qr_levels(df=ob.assign(p_ref=pref), levels=levels, K=K, pref=pref, tick_size=tick_size)

    # price level
    msg["lvl"] = price_level_from_qr(msg=msg, qr=qr, K=K)

    # base df = msg + ob
    df = pd.concat([msg[["time","type","price", "lvl", "size"]], qr.drop("tick_size", axis=1)], axis=1)
    df = df[df.lvl != 0]
    df = df.reset_index(drop=True)
    
    df.insert(1, "delta_time", df["time"].diff())
    df.loc[df.index[0], "delta_time"] = (df.loc[df.index[0], "time"] - market_open_seconds)

    return msg, ob, qr, df


def compute_aes_by_level(df: pd.DataFrame, K: int, lvl_col: str = "lvl", size_col: str = "size") -> pd.Series:
    # AES_i = mean size of all events at |lvl|=i (L/C/M included)
    abs_lvl = df[lvl_col].abs()
    aes = df.groupby(abs_lvl)[size_col].mean()
    # ensure index 1,...,K present
    aes = aes.reindex(range(1, K+1))
    aes.index.name = "level"
    aes.name = "AES"
    return aes


def compute_ait_by_level(df: pd.DataFrame, K: int, lvl_col: str = "lvl", time_col: str = "delta_time") -> pd.Series:
    # AIT_i = mean delta_time of all events at |lvl|=i (L/C/M included)
    abs_lvl = df[lvl_col].abs()
    ait = df.groupby(abs_lvl)[time_col].mean()
    # ensure index 1,...,K present
    ait = ait.reindex(range(1, K+1))
    ait.index.name = "level"
    ait.name = "AIT"
    return ait


def counts_by_level_and_type(df: pd.DataFrame, K: int, lvl_col="lvl", type_col="type") -> pd.DataFrame:
    tmp = df.copy()
    tmp["level"] = tmp[lvl_col].abs()
    tmp = tmp[tmp["level"].between(1, K)]
    ct = tmp.groupby(["level", type_col]).size().unstack(type_col, fill_value=0)
    for c in ["L", "C", "M"]:
        if c not in ct.columns:
            ct[c] = 0
    return ct[["L", "C", "M"]]


def make_descriptive_table(df: pd.DataFrame, K: int) -> pd.DataFrame:
    ct = counts_by_level_and_type(df, K=K)
    aes = compute_aes_by_level(df, K=K)
    ait = compute_ait_by_level(df, K=K)
    
    out = pd.concat([ct, aes, ait], axis=1)
    
    # scaling like the paper
    out["#L (×10^3)"] = out["L"] / 1e3
    out["#C (×10^3)"] = out["C"] / 1e3
    out["#M (×10^2)"] = out["M"] / 1e2
    out["AES"] = out["AES"]
    out["AIT (ms)"] = out["AIT"] * 1000.0

    out = out[["#L (×10^3)", "#C (×10^3)", "#M (×10^2)", "AES", "AIT (ms)"]]

    # formatting close to screenshot
    out = out.round({
        "#L (×10^3)": 2,
        "#C (×10^3)": 2,
        "#M (×10^2)": 2,
        "AES": 2,
        "AIT (ms)": 1,
    })

    out.index.name = "Level"
    
    return out


def normalize_by_aes(df: pd.DataFrame, aes: pd.Series, K: int,
                        lvl_col: str = "lvl", size_col: str = "size",
                        q_prefix: str = "Q_") -> pd.DataFrame:
    """
    - Queue normalization: q_i <- ceil(q_i / AES_i) for i=1..K on both sides (Q_i, Q_-i)
    - Event-size normalization: size <- size / AES_|lvl| (keeps float; if you want ceil/int, change below)

    Expects:
        df has columns: Q_1..Q_K, Q_-1..Q_-K, lvl, size
        aes indexed by level 1..K (mean event size at level)
    """
    out = df.copy()

    # --- 1) normalize queues ---
    for i in range(1, K + 1):
        a = float(aes.loc[i]) if pd.notna(aes.loc[i]) else np.nan
        if not np.isfinite(a) or a <= 0:
            out[f"{q_prefix}{i}"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="Int64")
            out[f"{q_prefix}-{i}"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="Int64")
        else:
            out[f"{q_prefix}{i}"]  = np.ceil(out[f"{q_prefix}{i}"]  / a).astype("Int64")
            out[f"{q_prefix}-{i}"] = np.ceil(out[f"{q_prefix}-{i}"] / a).astype("Int64")

    # --- 2) normalize event size by its level ---
    abs_lvl = out[lvl_col].abs().astype(int)
    denom = abs_lvl.map(aes)  # aligns per-row AES_|lvl|
    out[size_col] = np.ceil(out[size_col] / denom).astype("Int64")

    return out