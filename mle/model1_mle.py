from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .mapping import (
    KindStr,
    affected_side,
    event_kind_from_type,
    find_level_index,
    make_ob_cols,
)


@dataclass(frozen=True)
class Model1IntensityMLE:
    """
    Symmetric (±i averaged) intensities for Model I:
        lambda_L[i-1, n], lambda_C[i-1, n], lambda_M[i-1, n]
    i in 1..K, n in 0..n_max
    """
    K: int
    n_max: int
    lambda_L: np.ndarray  # (K, n_max+1)
    lambda_C: np.ndarray  # (K, n_max+1)
    lambda_M: np.ndarray  # (K, n_max+1)

    def _clip(self, n: int) -> int:
        nn = int(n)
        if nn < 0:
            nn = 0
        if nn > self.n_max:
            nn = self.n_max
        return nn

    # Methods compatible with IntensitiesModel Protocol interface
    def lambda_L_fn(self, level: int, n: int) -> float:
        i = abs(int(level))
        if not (1 <= i <= self.K):
            return 0.0
        return float(self.lambda_L[i - 1, self._clip(n)])

    def lambda_C_fn(self, level: int, n: int) -> float:
        i = abs(int(level))
        if not (1 <= i <= self.K):
            return 0.0
        return float(self.lambda_C[i - 1, self._clip(n)])

    def lambda_M_fn(self, level: int, n: int) -> float:
        i = abs(int(level))
        if not (1 <= i <= self.K):
            return 0.0
        return float(self.lambda_M[i - 1, self._clip(n)])


class EmpiricalIntensityModel:
    """
    Wrapper exposing the IntensityModel protocol:
        lambda_L(level,n), lambda_C(level,n), lambda_M(level,n)
    """
    def __init__(self, mle: Model1IntensityMLE):
        self.mle = mle

    def lambda_L(self, level: int, n: int) -> float:
        return self.mle.lambda_L_fn(level, n)

    def lambda_C(self, level: int, n: int) -> float:
        return self.mle.lambda_C_fn(level, n)

    def lambda_M(self, level: int, n: int) -> float:
        return self.mle.lambda_M_fn(level, n)


def fit_model1_mle_from_lobster(
    msg: pd.DataFrame,
    ob: pd.DataFrame,
    *,
    K: int,
    n_max: int = 50,
    drop_unsupported: bool = True,
) -> Model1IntensityMLE:
    """
    MLE of Model I intensities from aligned LOBSTER DataFrames.

    Estimator:
        lambda^T_i(n) = N^T_i(n) / time_in_state_i(n)

    Alignment:
        event at row k uses pre-book at row k-1
        dt_k = time[k] - time[k-1]

    Symmetry:
        estimates for ask levels (+i) and bid levels (-i) are computed separately,
        then averaged: (ask + bid)/2.
    """
    if len(msg) != len(ob):
        raise ValueError("msg and ob must have same length and be row-aligned")

    if len(msg) < 2:
        raise ValueError("Need at least 2 rows to form dt and pre-book alignment")

    ask_px_cols, ask_sz_cols, bid_px_cols, bid_sz_cols = make_ob_cols(K)

    # Align events with pre-book
    events = msg.iloc[1:].reset_index(drop=True)
    pre_ob = ob.iloc[:-1].reset_index(drop=True)

    t = msg["time"].to_numpy(dtype=float)
    dt = np.diff(t)  # length N-1

    kinds = events["type"].map(event_kind_from_type).to_numpy()

    if drop_unsupported:
        mask = np.array([k is not None for k in kinds], dtype=bool)
        events = events.loc[mask].reset_index(drop=True)
        pre_ob = pre_ob.loc[mask].reset_index(drop=True)
        dt = dt[mask]
        kinds = kinds[mask]

    # Pre arrays for speed
    pre_ask_px = pre_ob[ask_px_cols].to_numpy(dtype=np.int64)  # (N', K)
    pre_bid_px = pre_ob[bid_px_cols].to_numpy(dtype=np.int64)
    pre_ask_sz = pre_ob[ask_sz_cols].to_numpy(dtype=np.int64)
    pre_bid_sz = pre_ob[bid_sz_cols].to_numpy(dtype=np.int64)

    prices = events["price"].to_numpy(dtype=np.int64)
    directions = events["direction"].to_numpy(dtype=np.int64)

    # Occupancy (time spent at queue size n) and event counts per n
    occ_ask = np.zeros((K, n_max + 1), dtype=float)
    occ_bid = np.zeros((K, n_max + 1), dtype=float)

    cntL_ask = np.zeros((K, n_max + 1), dtype=np.int64)
    cntC_ask = np.zeros((K, n_max + 1), dtype=np.int64)
    cntM_ask = np.zeros((K, n_max + 1), dtype=np.int64)

    cntL_bid = np.zeros((K, n_max + 1), dtype=np.int64)
    cntC_bid = np.zeros((K, n_max + 1), dtype=np.int64)
    cntM_bid = np.zeros((K, n_max + 1), dtype=np.int64)

    def clip_n(x: int) -> int:
        x = int(x)
        if x < 0:
            return 0
        if x > n_max:
            return n_max
        return x

    N = len(events)
    for k in range(N):
        dtk = float(dt[k])

        # Occupancy: all levels at once from pre-book sizes
        for j in range(K):
            occ_ask[j, clip_n(pre_ask_sz[k, j])] += dtk
            occ_bid[j, clip_n(pre_bid_sz[k, j])] += dtk

        kind = kinds[k]
        if kind is None:
            continue

        side = affected_side(kind, int(directions[k]))
        px = int(prices[k])

        if side == "ask":
            i = find_level_index(px, pre_ask_px[k, :])
            if i == 0:
                continue
            j = i - 1
            n0 = clip_n(pre_ask_sz[k, j])
            if kind == "L":
                cntL_ask[j, n0] += 1
            elif kind == "C":
                cntC_ask[j, n0] += 1
            else:
                cntM_ask[j, n0] += 1

        else:  # bid
            i = find_level_index(px, pre_bid_px[k, :])
            if i == 0:
                continue
            j = i - 1
            n0 = clip_n(pre_bid_sz[k, j])
            if kind == "L":
                cntL_bid[j, n0] += 1
            elif kind == "C":
                cntC_bid[j, n0] += 1
            else:
                cntM_bid[j, n0] += 1

    def safe_div(cnt: np.ndarray, occ: np.ndarray) -> np.ndarray:
        out = np.zeros_like(occ, dtype=float)
        mask = occ > 0
        out[mask] = cnt[mask] / occ[mask]
        return out

    lamL_ask = safe_div(cntL_ask, occ_ask)
    lamC_ask = safe_div(cntC_ask, occ_ask)
    lamM_ask = safe_div(cntM_ask, occ_ask)

    lamL_bid = safe_div(cntL_bid, occ_bid)
    lamC_bid = safe_div(cntC_bid, occ_bid)
    lamM_bid = safe_div(cntM_bid, occ_bid)

    # Symmetry (simple average). Optionally: occupancy-weighted average later.
    lambda_L = 0.5 * (lamL_ask + lamL_bid)
    lambda_C = 0.5 * (lamC_ask + lamC_bid)
    lambda_M = 0.5 * (lamM_ask + lamM_bid)

    return Model1IntensityMLE(
        K=K,
        n_max=n_max,
        lambda_L=lambda_L,
        lambda_C=lambda_C,
        lambda_M=lambda_M,
    )