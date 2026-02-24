from __future__ import annotations
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from env import Item

@dataclass
class BidderState:
    """Mutable per-episode state tracked for each bidder."""

    budget: float
    remaining_budget: float
    spent: float = 0.0
    items_won: List[Item] = field(default_factory=list)
    prices_paid: List[float] = field(default_factory=list)
    total_score: float = 0.0

    @property
    def n_won(self) -> int:
        return len(self.items_won)


class BaseBidder(ABC):
    """Abstract base for every bidder (heuristic, RL, LLM, etc.)."""

    def __init__(self, bidder_id: int, budget: float):
        self.bidder_id = bidder_id
        self.budget = budget

    @abstractmethod
    def place_bid(
        self,
        item: Item,
        state: BidderState,
        items_remaining: int,
        value_mode: str = "linear",
    ) -> float:
        """Return a non-negative bid for *item* given current *state*."""
        ...

    def new_state(self) -> BidderState:
        """Create a fresh BidderState for the start of an episode."""
        return BidderState(budget=self.budget, remaining_budget=self.budget)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.bidder_id}, budget={self.budget:.1f})"


