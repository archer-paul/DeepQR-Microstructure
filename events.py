from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence

import numpy as np


class EventType(str, Enum):
    """ Atomic event kinds for Model I """
    L = "L"     # Limit order (+1)
    C = "C"     # Cancellation order (-1)
    M = "M"     # Market order (-1)


@dataclass(frozen=True, slots=True)
class Event:
    """ Atomic event: occurs at a signed level with a certain EventType """
    level: int
    kind: EventType


@dataclass(frozen=True, slots=True)
class EventRates:
    """
    Container returned by model.rates(state).
    - events[k] has intensity rates[k]
    """
    events: Sequence[Event]
    rates: np.ndarray # shape (n_events, )
    
    def __post_init__(self) -> None:
        r = np.asarray(self.rates, dtype=float)
        if r.ndim != 1:
            raise ValueError("rates must be 1D")
        if np.any(r<0.0):
            raise ValueError("rates must be nonnegative")
        if len(self.events) != r.shape[0]:
            raise ValueError("events and rates must have same length")
        object.__setattr__(self, "rates", r)
    
    @property
    def total_rate(self) -> float:
        return float(self.rates.sum())


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """
    Event-driven simulation output.

    Arrays are aligned by event index k:
    - times[k]        : event time (strictly increasing)
    - levels[k]       : signed level where event occurred
    - kinds[k]        : event kind encoded as small int (0=L,1=C,2=M) for speed
    - q_before[k]     : queue size at that level before the event
    - q_after[k]      : queue size at that level after the event
    """
    times: np.ndarray
    levels: np.ndarray
    kinds: np.ndarray
    q_before: np.ndarray
    q_after: np.ndarray
    seed: Optional[int] = None
    meta: Optional[dict] = None
    
    def __post_init__(self) -> None:
        self._check_1d_int(self.levels, "levels")
        self._check_1d_int(self.kinds, "kinds")
        self._check_1d_int(self.q_before, "q_before")
        self._check_1d_int(self.q_after, "q_after")
        
        t = np.asarray(self.times, dtype=float)
        if t.ndim != 1:
            raise ValueError("times must be 1D")
        if len(t) != len(self.levels):
            raise ValueError("all arrays must have same length")
        if np.any(np.diff(t) <= 0.0) and len(t) > 1:
            raise ValueError("times must be strictly increasing")
        if np.any(self.q_before < 0) or np.any(self.q_after < 0):
            raise ValueError("queue sizes must be nonnegative")
        if np.any((self.kinds < 0) | (self.kinds > 2)):
            raise ValueError("kinds must be in {0, 1, 2}")
        object.__setattr__(self, "times", t)
    
    @staticmethod
    def _check_1d_int(arr: np.ndarray, name:str) -> None:
        a = np.asarray(arr)
        if a.ndim != 1:
            raise ValueError(f"{name} must be 1D")
        # don't force dtype exactly; just ensure integer-like
        if not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f"{name} must be integer dtype")
    
    def n_events(self) -> int:
        return int(self.times.shape[0])
    
    def to_event_types(self) -> List[EventType]:
        mapping = {0: EventType.L, 1: EventType.C, 2: EventType.M}
        return[mapping[int(k)] for k in self.kinds.tolist()]