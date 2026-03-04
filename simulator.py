from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Tuple

import numpy as np

from events import Event, EventRates, EventType, SimulationResult
from state import LOBState


class ModelProtocol(Protocol):
    def rates(self, state: LOBState) -> EventRates: ...
    def apply(self, state: LOBState, event: Event) -> None: ...
    
    
def _kind_to_code(kind: EventType) -> int:
    if kind == EventType.L:
        return 0
    if kind == EventType.C:
        return 1
    if kind == EventType.M:
        return 2
    raise ValueError(f"Unknown kind {kind}")


@dataclass
class CTMCSimulator:
    """
    Generic CTMC simulator using Gillespie's direct method.

    The simulator is model-agnostic:
        - it queries model.rates(state) to get events + intensities
        - it samples next event time and event index
        - it calls model.apply(state, event)
    """
    model : ModelProtocol
    seed: Optional[int] = None
    
    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        
    def run(
        self,
        state0: LOBState,
        *,
        T: Optional[float] = None,
        n_events: Optional[int] = None,
        max_events: int = 10_000_000,
        log_capacity: int = 100_000,
        copy_state0: bool = True,
    ) -> Tuple[LOBState, SimulationResult]:
        """
        Generic CTMC simulator using Gillespie's direct method.

        The simulator is model-agnostic:
            - it queries model.rates(state) to get events + intensities
            - it samples next event time and event index
            - it calls model.apply(state, event)
        """
        if T is None and n_events is None:
            raise ValueError("Provide at least one stopping criterion: T or n_events")
        
        if T is not None and T <= 0:
            raise ValueError("T must be > 0")
        if n_events is not None and n_events<=0:
            raise ValueError("n_events must be > 0")
        
        state = state0.copy() if copy_state0 else state0
        
        # Pre-allocate logs (grow dynamically)
        cap = int(log_capacity)
        times    = np.empty(cap, dtype=float)
        levels   = np.empty(cap, dtype=np.int32)
        kinds    = np.empty(cap, dtype=np.int8)
        q_before = np.empty(cap, dtype=np.int32)
        q_after  = np.empty(cap, dtype=np.int32)
        
        t = 0.0
        k = 0 # number of looged events
        
        
        while True:
            if k > max_events:
                raise RuntimeError(f"Reached max_events={max_events} without stopping")
            
            if T is not None and t >= T:
                break
            if n_events is not None and k >= n_events:
                break
            
            er = self.model.rates(state)
            R = er.total_rate
            if R <= 0.0:
                # No admissible events => absorbing / stuck
                break
            
            # Sample next event time
            dt = self.rng.exponential(1.0/R)
            t_next = t+dt
            
            # If time horizon would be exceed, stop
            if T is not None and t_next > T:
                t = T
                break
            
            # Choose event index
            # Prob(event=j) = rate_j / R
            probs = er.rates / R
            j = int(self.rng.choice(len(er.events), p=probs))
            ev = er.events[j]       
            
            lvl = int(ev.level)
            qb = state.get(lvl)
            
            # Apply event
            self.model.apply(state, ev)
            
            qa = state.get(lvl)
            
            # Grow array if needed
            if k >= cap:
                cap      = int(cap * 1.5) + 1
                times    = np.resize(times, cap)
                levels   = np.resize(levels, cap)
                kinds    = np.resize(kinds, cap)
                q_before = np.resize(q_before, cap)
                q_after  = np.resize(q_after, cap)

            times[k]    = t_next
            levels[k]   = lvl
            kinds[k]    = _kind_to_code(ev.kind)
            q_before[k] = qb
            q_after[k]  = qa
            
            t = t_next
            k += 1
        
        # Trim logs
        times    = times[:k]
        levels   = levels[:k]
        kinds    = kinds[:k]
        q_before = q_before[:k]
        q_after  = q_after[:k]
        
        result = SimulationResult(
            times=times,
            levels=levels,
            kinds=kinds,
            q_before=q_before,
            q_after=q_after,
            seed=self.seed,
            meta={"stopped_at_time": t, "n_events": int(k), "K":int(state.K)},
        )
        return state, result
