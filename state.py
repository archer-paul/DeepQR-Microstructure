from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np


def _default_levels(K:int) -> np.ndarray:
    if K<=0:
        raise ValueError(f"K must be >= 1, got {K}")
    # [-K,...,-1,1,...,K]
    return np.array(list(range(-K,0))+list(range(1,K+1)), dtype=int)


@dataclass
class LOBState:
    """
    LOB state in the reference-price frame with fixed K and fixed p_ref.
    Stores queue sizes q_i for i in [-K..-1, 1..K].

    Notes:
    - q is stored as a flat vector aligned with `levels`.
    - No price dynamics here (p_ref changes belong to higher-level models).
    """
    K: int
    q: np.ndarray
    levels: np.ndarray
    level_to_index: Dict[int, int]
    p_ref: Optional[float] = None
    
    def __init__(
        self,
        K       : int,
        q0      : Optional[Sequence[int]] = None,
        *,
        p_ref   : Optional[float] = None,
        levels  : Optional[Sequence[int]] = None,
        dtype=np.int64,
    ):
        self.K = int(K)
        self.levels = _default_levels(self.K) if levels is None else np.array(levels, dtype=int)
        
        expected = 2 * self.K
        if self.levels.shape != (expected, ):
            raise ValueError(f"levels must have shape ({expected}, ), got {self.levels.shape}")
        if 0 in set(self.levels.tolist()):
            raise ValueError("levels must not include 0 (reference price level is excluded)")
        
        self.level_to_index = {lvl: i for i, lvl in enumerate(self.levels)}
        if len(self.level_to_index) != len(self.levels):
            raise ValueError("levels must be unique")
        
        if q0 is None:
            self.q = np.zeros(expected, dtype=dtype)
        else:
            q_arr = np.asarray(q0, dtype=dtype)
            if q_arr.shape != (expected, ):
                raise ValueError(f"q0 must have shape ({expected,}), got {q_arr.shape}")
            if np.any(q_arr < 0):
                raise ValueError("queue sizes must be nonnegative")
            self.q = q_arr.copy()
        
        self.pref = p_ref
        
    # ---------- indexing helpers ----------
    
    def idx(self, level:int) -> int:
        """ Map a signed level (e.g. -1, +2) to the internal index."""
        try:
            return self.level_to_index[int(level)]
        except KeyError as e:
            raise KeyError(f"Unknown level {level}. Valid: {self.levels.tolist()}") from e
    
    def get(self, level:int) -> int:
        return int(self.q[self.idx(level)])
    
    def set(self, level:int, value:int) -> None:
        v = int(value)
        if v<0:
            raise ValueError("queue sizes must be nonnegative")
        self.q[self.idx(level)] = v
        
    # ---------- mutation primitives ----------
    
    def incr(self, level:int, amount:int=1) -> None:
        """ Increase the queue at level `level` by amount `a` """
        a = int(amount)
        if a<0:
            raise ValueError("increasing amount must be >= 0")
        self.q[self.idx(level)] += a
        
    def decr(self, level:int, amount:int=1) -> None:
        """ Decrease the queue at level `level` by amount `a` """
        a = int(amount)
        if a<0:
            raise ValueError("decreasing amount must be >=0")
        j = self.idx(level)
        if self.q[j] < a:
            raise ValueError(f"Cannot decrement level {level} by {a}; current={int(self.q[j])}")
        self.q[j] -= a
        
    # ---------- convenience ----------
    
    def copy(self) -> "LOBState":
        return LOBState(self.K, self.q.copy(), p_ref=self.p_ref)
    
    @property
    def n_levels(self) -> int:
        return int(self.q.shape[0])
    
    def as_dict(self) -> Dict[int, int]:
        """ Return {level: queue_size} """
        return {int(lvl) : int(self.q[i]) for i, lvl in enumerate(self.levels)}
    
    def validate(self) -> None:
        """ Consistency check (useful for tests) """
        if self.q.shape != (2*self.K,):
            raise ValueError("q shape mismatch")
        if np.any(self.q<0):
            raise ValueError("negative queue size detected")
        
    def best_bid_level(self) -> Optional[int]:
        """ Closest bid level with positive size (highest price) """
        for lvl in (-1, *range(-2, -self.K -1, -1)):
            if self.get(lvl)>0:
                return lvl
        return None
    
    def best_ask_level(self) -> Optional[int]:
        """ Closest ask level with positive size (highest price) """
        for lvl in (1, *range(2, self.K + 1)):
            if self.get(lvl) > 0:
                return lvl
        return None
    
    def __repr__(self) -> str:
        return f"LOBState(K={self.K}, q={self.q.tolist()}, p_ref={self.p_ref})"