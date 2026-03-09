from __future__ import annotations

from abc import ABC, abstractmethod

from events import EventRates
from state import LOBState


class BaseLOBModel(ABC):
    """
    Model interface for the CTMC simulator.
    The simulator is generic: it only need rates(state) and apply(state, event).
    """
    
    @abstractmethod
    def rates(self, state: LOBState) -> EventRates:
        """ Return admissible events and their intensities at current state. """
        raise NotImplementedError
    
    @abstractmethod
    def apply(self, state: LOBState, event) -> None:
        """ Mutate the state according to the chosen event. """
        raise NotImplementedError