from __future__ import annotations
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from env import Item
from scoring import rank_to_weight


@dataclass
class BidderState:
    """Mutable per-episode state tracked for each bidder."""

    budget: float
    remaining_budget: float
    n_items: int = 0
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

    def _priority_value(
        self,
        item: Item,
        state: BidderState,
        weight_scheme: str = "linear",
    ) -> float:
        """
        Convert an item's priority rank into a dollar-denominated value
        so existing bid-ceiling math (value / beta, etc.) works unchanged.

        priority_value = w_i * B, where w_i ∈ [0, 1] comes from the rank
        and B is the starting budget.  Break-even bid = w_i * B / β, which
        matches the scoring formula score = w_i − β * (p_i / B).
        """
        w = rank_to_weight(item.rank, state.n_items, weight_scheme)
        return w * state.budget

    @abstractmethod
    def place_bid(
        self,
        item: Item,
        state: BidderState,
        items_remaining: int,
        weight_scheme: str = "linear",
    ) -> float:
        """Return a non-negative bid for *item* given current *state*."""
        ...

    def new_state(self, n_items: int = 0) -> BidderState:
        """Create a fresh BidderState for the start of an episode."""
        return BidderState(
            budget=self.budget, remaining_budget=self.budget, n_items=n_items,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.bidder_id}, budget={self.budget:.1f})"


class PositiveMarginBidder(BaseBidder):
    """
    HEURISTIC 1: Positive Margin Bidder

    Bids up to the break-even price p ≤ (w_i * B) / β, then samples
    a bid uniformly from [min_bid, max_bid].

    Parameters
    ----------
    beta    : cost sensitivity — higher values make the bidder more
              conservative by shrinking the max affordable bid
    min_bid : minimum bid submitted if bidding at all; prevents
              zero or negligible bids that win nothing
    """

    def __init__(
        self,
        bidder_id: int,
        budget: float,
        beta: float = 1.0,
        min_bid: float = 0.01,
    ):
        super().__init__(bidder_id, budget)
        self.beta = beta
        self.min_bid = min_bid

    def place_bid(
        self,
        item: Item,
        state: BidderState,
        items_remaining: int,
        weight_scheme: str = "linear",
    ) -> float:
        if state.remaining_budget <= 0:
            return 0.0

        value = self._priority_value(item, state, weight_scheme)
        max_bid = value / self.beta
        max_bid = min(max_bid, state.remaining_budget)

        if max_bid < self.min_bid:
            return 0.0

        return random.uniform(self.min_bid, max_bid)


class MarginPlusSafetyBidder(BaseBidder):
    """
    HEURISTIC 2: Margin + Safety Buffer Bidder

    Only bids when the item clears a minimum profit threshold:
        w_i*B - β*p ≥ m  →  p ≤ (w_i*B - m) / β

    Bids in the upper half of the affordable range.

    Parameters
    ----------
    beta   : cost sensitivity weight
    margin : required minimum profit m; higher values mean the bidder
             only enters auctions where winning is clearly worthwhile
    """

    def __init__(
        self,
        bidder_id: int,
        budget: float,
        beta: float = 1.0,
        margin: float = 1.0,
    ):
        super().__init__(bidder_id, budget)
        self.beta = beta
        self.margin = margin

    def place_bid(
        self,
        item: Item,
        state: BidderState,
        items_remaining: int,
        weight_scheme: str = "linear",
    ) -> float:
        if state.remaining_budget <= 0:
            return 0.0

        value = self._priority_value(item, state, weight_scheme)
        max_bid = (value - self.margin) / self.beta
        max_bid = min(max_bid, state.remaining_budget)

        if max_bid <= 0:
            return 0.0

        return random.uniform(max_bid * 0.5, max_bid)


class BudgetPacedMarginBidder(BaseBidder):
    """
    HEURISTIC 3: Budget-Paced Margin Bidder

    Applies two simultaneous bid caps:
        1. Margin cap  : p ≤ (w_i * B) / β
        2. Pace cap    : p ≤ c * (remaining_budget / items_remaining)

    For items ranked within the top-K, the pace cap is removed and
    only the margin cap applies (priority override).

    Parameters
    ----------
    beta  : cost sensitivity weight
    c     : pace multiplier; c > 1 allows spending above even-pace,
            c < 1 forces under-pacing to save budget for later
    top_k : items with rank ≤ top_k bypass the pace cap entirely
    """

    def __init__(
        self,
        bidder_id: int,
        budget: float,
        beta: float = 1.0,
        c: float = 1.2,
        top_k: int = 3,
    ):
        super().__init__(bidder_id, budget)
        self.beta = beta
        self.c = c
        self.top_k = top_k

    def place_bid(
        self,
        item: Item,
        state: BidderState,
        items_remaining: int,
        weight_scheme: str = "linear",
    ) -> float:
        if state.remaining_budget <= 0 or items_remaining <= 0:
            return 0.0

        value = self._priority_value(item, state, weight_scheme)
        margin_limit = value / self.beta
        pace_limit = self.c * (state.remaining_budget / items_remaining)
        is_top_k = item.rank <= self.top_k

        if is_top_k:
            max_bid = min(margin_limit, state.remaining_budget)
        else:
            max_bid = min(margin_limit, pace_limit, state.remaining_budget)

        if max_bid <= 0:
            return 0.0

        low = max_bid * (0.7 if is_top_k else 0.4)
        return random.uniform(low, max_bid)


class TopKSpecialistBidder(BaseBidder):
    """
    HEURISTIC 4: Top-K Specialist Bidder

    Abstains on all items with rank > top_k. On top-K items, bids
    near the margin ceiling (w_i*B - m) / β.

    Parameters
    ----------
    beta  : cost sensitivity weight
    top_k : only participate in auctions for items ranked ≤ top_k
    margin: minimum profit buffer required before bidding
    """

    def __init__(
        self,
        bidder_id: int,
        budget: float,
        beta: float = 1.0,
        top_k: int = 3,
        margin: float = 0.0,
    ):
        super().__init__(bidder_id, budget)
        self.beta = beta
        self.top_k = top_k
        self.margin = margin

    def place_bid(
        self,
        item: Item,
        state: BidderState,
        items_remaining: int,
        weight_scheme: str = "linear",
    ) -> float:
        if state.remaining_budget <= 0:
            return 0.0

        if item.rank > self.top_k:
            return 0.0

        value = self._priority_value(item, state, weight_scheme)
        max_bid = (value - self.margin) / self.beta
        max_bid = min(max_bid, state.remaining_budget)

        if max_bid <= 0:
            return 0.0

        return random.uniform(max_bid * 0.75, max_bid)


class FlatFractionBidder(BaseBidder):
    """
    HEURISTIC 5: Flat Fraction Bidder

    Bids a fixed fraction f of priority value on every item with no
    adaptation. bid = f * w_i * B, capped at remaining budget.

    Parameters
    ----------
    f : bid as a fraction of priority value; f > 1 produces overbids
        relative to priority, f < 1 produces conservative bids
    """

    def __init__(
        self,
        bidder_id: int,
        budget: float,
        f: float = 0.8,
    ):
        super().__init__(bidder_id, budget)
        self.f = f

    def place_bid(
        self,
        item: Item,
        state: BidderState,
        items_remaining: int,
        weight_scheme: str = "linear",
    ) -> float:
        if state.remaining_budget <= 0:
            return 0.0

        value = self._priority_value(item, state, weight_scheme)
        bid = self.f * value
        return min(bid, state.remaining_budget)


class DescendingAggressionBidder(BaseBidder):
    """
    HEURISTIC 6: Descending Aggression Bidder

    Linearly interpolates bid aggressiveness between f_start and f_end
    based on how much of the budget has been spent:

        aggression = f_start - (f_start - f_end) * (spent / budget)
        bid = aggression * (w_i * B) / β

    Parameters
    ----------
    beta    : cost sensitivity weight
    f_start : aggression fraction at episode start (spend_ratio = 0)
    f_end   : aggression fraction when budget is fully spent (spend_ratio = 1)
    """

    def __init__(
        self,
        bidder_id: int,
        budget: float,
        beta: float = 1.0,
        f_start: float = 0.95,
        f_end: float = 0.2,
    ):
        super().__init__(bidder_id, budget)
        self.beta = beta
        self.f_start = f_start
        self.f_end = f_end

    def place_bid(
        self,
        item: Item,
        state: BidderState,
        items_remaining: int,
        weight_scheme: str = "linear",
    ) -> float:
        if state.remaining_budget <= 0:
            return 0.0

        spend_ratio = state.spent / state.budget if state.budget > 0 else 0.0
        aggression = self.f_start - (self.f_start - self.f_end) * spend_ratio

        value = self._priority_value(item, state, weight_scheme)
        max_bid = aggression * value / self.beta
        max_bid = min(max_bid, state.remaining_budget)

        if max_bid <= 0:
            return 0.0

        noise = random.uniform(0.9, 1.0)
        return max_bid * noise


class SnipeBidder(BaseBidder):
    """
    HEURISTIC 7: Snipe Bidder

    Abstains on items with rank < snipe_from_rank (early/high-priority
    items), conserving the full budget. Once rank >= snipe_from_rank,
    bids aggressively at a multiple of the per-item pace allocation:

        snipe_limit = aggression * (remaining_budget / items_remaining)

    Final bid = min(margin_limit, snipe_limit, remaining_budget).

    Parameters
    ----------
    beta            : cost sensitivity weight
    snipe_from_rank : rank threshold; abstain below this, snipe at or above
    aggression      : multiplier on the per-item budget pace when sniping;
                      values > 1.0 mean spending above sustainable pace
    """

    def __init__(
        self,
        bidder_id: int,
        budget: float,
        beta: float = 1.0,
        snipe_from_rank: int = 6,
        aggression: float = 1.5,
    ):
        super().__init__(bidder_id, budget)
        self.beta = beta
        self.snipe_from_rank = snipe_from_rank
        self.aggression = aggression

    def place_bid(
        self,
        item: Item,
        state: BidderState,
        items_remaining: int,
        weight_scheme: str = "linear",
    ) -> float:
        if state.remaining_budget <= 0 or items_remaining <= 0:
            return 0.0

        if item.rank < self.snipe_from_rank:
            return 0.0

        value = self._priority_value(item, state, weight_scheme)
        margin_limit = value / self.beta
        snipe_limit = self.aggression * (state.remaining_budget / items_remaining)
        max_bid = min(margin_limit, snipe_limit, state.remaining_budget)

        if max_bid <= 0:
            return 0.0

        return random.uniform(max_bid * 0.8, max_bid)


class RandomBidder(BaseBidder):
    """
    HEURISTIC 8: Random Bidder

    Bids a uniformly random amount between 0 and
    max_fraction * remaining_budget with no regard for item priority,
    budget state, or episode position.

    Parameters
    ----------
    max_fraction : upper bound on bids as a fraction of remaining budget;
                   caps runaway spending while still allowing large bids
    """

    def __init__(
        self,
        bidder_id: int,
        budget: float,
        max_fraction: float = 0.5,
    ):
        super().__init__(bidder_id, budget)
        self.max_fraction = max_fraction

    def place_bid(
        self,
        item: Item,
        state: BidderState,
        items_remaining: int,
        weight_scheme: str = "linear",
    ) -> float:
        if state.remaining_budget <= 0:
            return 0.0

        cap = self.max_fraction * state.remaining_budget
        return random.uniform(0.0, cap)


_PARAM_RANGES = {
    "PositiveMarginBidder": {
        "beta": (0.5, 2.5),
        "min_bid": (0.01, 1.0),
    },
    "MarginPlusSafetyBidder": {
        "beta": (0.5, 2.5),
        "margin": (0.0, 4.0),
    },
    "BudgetPacedMarginBidder": {
        "beta": (0.5, 2.5),
        "c": (0.8, 2.0),
        "top_k": (1, 5),
    },
    "TopKSpecialistBidder": {
        "beta": (0.5, 2.0),
        "top_k": (1, 5),
        "margin": (0.0, 2.0),
    },
    "FlatFractionBidder": {
        "f": (0.3, 1.5),
    },
    "DescendingAggressionBidder": {
        "beta": (0.5, 2.0),
        "f_start": (0.7, 1.0),
        "f_end": (0.1, 0.4),
    },
    "SnipeBidder": {
        "beta": (0.5, 2.0),
        "snipe_from_rank": (4, 8),
        "aggression": (1.0, 2.5),
    },
    "RandomBidder": {
        "max_fraction": (0.2, 0.8),
    },
}


def build_opponent_pool(
    n_opponents: int,
    budget: float,
    seed: Optional[int] = None,
    budget_noise: float = 0.2,
) -> List[BaseBidder]:
    """
    Builds a diverse pool of randomized heuristic opponents.

    Cycles through all 8 bidder types in round-robin order so every
    type is always represented. Parameters are sampled from _PARAM_RANGES.

    Parameters
    ----------
    n_opponents  : number of opponent bidders to create
    budget       : base budget (each bidder gets budget * U(1 ± noise))
    seed         : random seed for reproducibility
    budget_noise : fractional variation on per-bidder budget
    """
    rng = random.Random(seed)

    bidder_classes = [
        PositiveMarginBidder,
        MarginPlusSafetyBidder,
        BudgetPacedMarginBidder,
        TopKSpecialistBidder,
        FlatFractionBidder,
        DescendingAggressionBidder,
        SnipeBidder,
        RandomBidder,
    ]

    bidders: List[BaseBidder] = []

    for i in range(n_opponents):
        bidder_id = i + 1
        cls = bidder_classes[i % len(bidder_classes)]
        ranges = _PARAM_RANGES[cls.__name__]

        bgt = budget * rng.uniform(1.0 - budget_noise, 1.0 + budget_noise)

        params: Dict = {}
        for param, bounds in ranges.items():
            lo, hi = bounds
            if isinstance(lo, int) and isinstance(hi, int):
                params[param] = rng.randint(lo, hi)
            else:
                params[param] = rng.uniform(lo, hi)

        bidders.append(cls(bidder_id=bidder_id, budget=bgt, **params))

    return bidders
