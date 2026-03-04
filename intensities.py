from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import numpy as np


class IntensityModel(Protocol):
    """
    Interface for Model-I style intensities:
        lambda_L(level, n), lambda_C(level, n), lambda_M(level, n)
    where:
        - level is signed: -K..-1, +1..+K
        - n is current queue size (nonnegative int)
    """
    def lambda_L(self, level: int, n: int) -> float: ...
    def lambda_C(self, level: int, n: int) -> float: ...
    def lambda_M(self, level: int, n: int) -> float: ...
    

def _sym_level(level: int) -> int:
    """Map level to its symmetric representative (absolute value)."""
    l = int(level)
    if l == 0:
        raise ValueError("level cannot be 0")
    return abs(l)


@dataclass(frozen=True, slots=True)
class DummyIntensityModel:
    """
    Simple, stable dummy parametrization with bid/ask symmetry:
        lambda^L_i(n) = a_i
        lambda^C_i(n) = b_i * n
        lambda^M_i(n) = m_i  (often nonzero mostly at i=1)

    Parameters are keyed by distance i = abs(level) in {1..K}.

    This is a good default because:
    - cancellations proportional to n prevents explosive growth
    - you can emphasize market orders at the best (i=1)
    """
    K: int
    
    # a_i >= 0
    a_L: Tuple[float, ...] # length K
    
    # b_i >= 0
    b_C: Tuple[float, ...] # length K
    
    # m_i >= 0
    m_M: Tuple[float, ...] # length K
    
    # small floor to avoid total_rate==0 edge cases
    eps: float = 0.0
    
    def __post_init__(self) -> None:
        K = int(self.K)
        if K <= 0:
            raise ValueError("K must be >= 1")
        if len(self.a_L) != K or len(self.b_C) != K or len(self.m_M) != K:
            raise ValueError("a_L, b_C, m_M must all have length K")
        if any(x < 0.0 for x in (*self.a_L, *self.b_C, *self.m_M)):
            raise ValueError("all parameters must be nonnegative")
        if self.eps < 0.0:
            raise ValueError("epsilon must be nonnegative")
        
    def _check_n(self, n: int) -> int:
        nn = int(n)
        if nn < 0:
            raise ValueError("n must be nonnegative")
        return nn
    
    def _params(self, level: int) -> Tuple[float, float, float]:
        i = _sym_level(level)
        if not (1 <= i <= self.K):
            raise ValueError(f"level {level} out of range for K={self.K}")
        idx = i-1
        return float(self.a_L[idx]), float(self.b_C[idx]), float(self.m_M[idx])
    
    def lambda_L(self, level: int, n: int) -> float:
        self._check_n(n)
        a, _, _ = self._params(level)
        return max(0.0, a + self.eps)
    
    def lambda_C(self, level: int, n: int) -> float:
        nn = self._check_n(n)
        _, b, _ = self._params(level)
        # cancellation proportional to queue size
        return max(0.0, b * nn + self.eps)
    
    def lambda_M(self, level: int, n: int) -> float:
        self._check_n(n)
        _, _, m = self._params(level)
        return max(0.0, m + self.eps)


def make_default_dummy_intensity(K: int = 3) -> DummyIntensityModel:
    """
    Defaults: higher activity at best (i=1), decreasing outward.
    You can tune these later.
    """
    K = int(K)
    a_L = tuple(np.linspace(6.0, 2.0, K)) # limit orders (insertion)
    b_C = tuple(np.linspace(0.6, 0.3, K)) # cancellation orders
    m_M = tuple([3.0] + [0.2] * (K-1))    # market orders mostly at best
    return DummyIntensityModel(K=K, a_L=a_L, b_C=b_C, m_M=m_M, eps=0.0)