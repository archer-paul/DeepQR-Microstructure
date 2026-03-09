from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from events import Event, EventRates, EventType
from intensities import IntensityModel
from state import LOBState


@dataclass(frozen=True, slots=True)
class Model1IndependentQueues:
    """
    Model I: independent queues, local state-dependent intensities.

    Events at each signed level `lvl`:
        - Limit insertion:   q(lvl) += 1, rate lambda_L(lvl, q)
        - Cancellation:      q(lvl) -= 1, rate lambda_C(lvl, q)  (only if q>0)
        - Market order:      q(lvl) -= 1, rate lambda_M(lvl, q)  (only if q>0)

    Notes:
    - We keep 'market order at level i' as in the paper's Model I approximation.
    - Symmetry is handled inside the IntensityModel (recommended).
    """
    intensity: IntensityModel
    
    def rates(self, state: LOBState) -> EventRates:
        events: List[Event] = []
        rates: List[float] = []
        
        # Iterate in the order : [-K,...-1,+1,...,+K]
        for lvl in state.levels.tolist():
            n = int(state.get(lvl))
            
            # Limit insertion always admissible
            rL = float(self.intensity.lambda_L(lvl, n))
            if rL < 0:
                raise ValueError("Negative rate produced by intensity model")
            if rL > 0:
                events.append(Event(level=int(lvl), kind=EventType.L))
                rates.append(rL)
                
            # Death events only if queue nonempty
            if n > 0:
                rC = float(self.intensity.lambda_C(lvl, n))
                rM = float(self.intensity.lambda_M(lvl, n))
                if rC < 0 or rM < 0:
                    raise ValueError("Negative rate produced by intensity model")

                if rC > 0:
                    events.append(Event(level=int(lvl), kind=EventType.C))
                    rates.append(rC)
                if rM > 0:
                    events.append(Event(level=int(lvl), kind=EventType.M))
                    rates.append(rM)
        
        return EventRates(events=events, rates=np.asarray(rates, dtype=float))
    
    def apply(self, state: LOBState, event: Event) -> None:
        lvl = int(event.level)
        if event.kind == EventType.L:
            state.incr(lvl, 1)
        elif event.kind == EventType.C or event.kind == EventType.M:
            state.decr(lvl, 1)
        else:
            raise ValueError(f"Unknown event kind: {event.kind}")
        