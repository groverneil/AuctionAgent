from __future__ import annotations
import random
import re
import os
import importlib
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


class PositiveMarginBidder(BaseBidder):
    """
    HEURISTIC 1: Positive Margin Bidder

    Bids up to the break-even price p ≤ Q_i / β, then samples
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
        value_mode: str = "linear",
    ) -> float:
        if state.remaining_budget <= 0:
            return 0.0

        value = item.get_value(value_mode)
        max_bid = value / self.beta
        max_bid = min(max_bid, state.remaining_budget)

        if max_bid < self.min_bid:
            return 0.0

        return random.uniform(self.min_bid, max_bid)


class MarginPlusSafetyBidder(BaseBidder):
    """
    HEURISTIC 2: Margin + Safety Buffer Bidder

    Only bids when the item clears a minimum profit threshold:
        Q_i - β * p ≥ m  →  p ≤ (Q_i - m) / β

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
        value_mode: str = "linear",
    ) -> float:
        if state.remaining_budget <= 0:
            return 0.0

        value = item.get_value(value_mode)
        max_bid = (value - self.margin) / self.beta
        max_bid = min(max_bid, state.remaining_budget)

        if max_bid <= 0:
            return 0.0

        return random.uniform(max_bid * 0.5, max_bid)


class BudgetPacedMarginBidder(BaseBidder):
    """
    HEURISTIC 3: Budget-Paced Margin Bidder

    Applies two simultaneous bid caps:
        1. Margin cap  : p ≤ Q_i / β
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
        value_mode: str = "linear",
    ) -> float:
        if state.remaining_budget <= 0 or items_remaining <= 0:
            return 0.0

        value = item.get_value(value_mode)
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
    near the margin ceiling (Q_i - m) / β.

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
        value_mode: str = "linear",
    ) -> float:
        if state.remaining_budget <= 0:
            return 0.0

        if item.rank > self.top_k:
            return 0.0

        value = item.get_value(value_mode)
        max_bid = (value - self.margin) / self.beta
        max_bid = min(max_bid, state.remaining_budget)

        if max_bid <= 0:
            return 0.0

        return random.uniform(max_bid * 0.75, max_bid)


class FlatFractionBidder(BaseBidder):
    """
    HEURISTIC 5: Flat Fraction Bidder

    Bids a fixed fraction f of item quality on every item with no
    adaptation. bid = f * Q_i, capped at remaining budget.

    Parameters
    ----------
    f : bid as a fraction of item quality; f > 1 produces overbids
        relative to quality, f < 1 produces conservative bids
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
        value_mode: str = "linear",
    ) -> float:
        if state.remaining_budget <= 0:
            return 0.0

        value = item.get_value(value_mode)
        bid = self.f * value
        return min(bid, state.remaining_budget)


class DescendingAggressionBidder(BaseBidder):
    """
    HEURISTIC 6: Descending Aggression Bidder

    Linearly interpolates bid aggressiveness between f_start and f_end
    based on how much of the budget has been spent:

        aggression = f_start - (f_start - f_end) * (spent / budget)
        bid = aggression * Q_i / β

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
        value_mode: str = "linear",
    ) -> float:
        if state.remaining_budget <= 0:
            return 0.0

        spend_ratio = state.spent / state.budget if state.budget > 0 else 0.0
        aggression = self.f_start - (self.f_start - self.f_end) * spend_ratio

        value = item.get_value(value_mode)
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
        value_mode: str = "linear",
    ) -> float:
        if state.remaining_budget <= 0 or items_remaining <= 0:
            return 0.0

        if item.rank < self.snipe_from_rank:
            return 0.0

        value = item.get_value(value_mode)
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
    max_fraction * remaining_budget with no regard for item quality,
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
        value_mode: str = "linear",
    ) -> float:
        if state.remaining_budget <= 0:
            return 0.0

        cap = self.max_fraction * state.remaining_budget
        return random.uniform(0.0, cap)


class LLMBidder(BaseBidder):
    """
    LLM-driven bidder.

    Uses a text prompt to ask an LLM for a single numeric bid amount, then
    clamps that bid to valid bounds [0, remaining_budget].

    - If parsing/API fails, this bidder returns 0.0 (drop out).
    """

    BASE_URL = "https://tritonai-api.ucsd.edu"
    MODEL = "api-gpt-oss-120b"

    def __init__(
        self,
        bidder_id: int,
        budget: float,
        beta: float = 1.0,
        temperature: float = 0.1,
        debug: bool = False,
    ):
        super().__init__(bidder_id, budget)
        self.beta = beta
        self.temperature = temperature
        self.debug = debug
        self.api_key = os.getenv("API_KEY", "")
        self._client = None

    def _get_client(self):
        if not self.api_key:
            raise ValueError("Missing API_KEY environment variable.")
        if self._client is None:
            openai_mod = importlib.import_module("openai")
            OpenAI = getattr(openai_mod, "OpenAI")
            self._client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)
        return self._client

    @staticmethod
    def _extract_first_number(text: str) -> Optional[float]:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def place_bid(
        self,
        item: Item,
        state: BidderState,
        items_remaining: int,
        value_mode: str = "linear",
        current_price: Optional[float] = None,
        min_required: Optional[float] = None,
        reserve_price: Optional[float] = None,
        has_current_winner: Optional[bool] = None,
    ) -> float:
        if state.remaining_budget <= 0:
            return 0.0

        value = float(item.get_value(value_mode))
        bid_history = [float(x) for x in item.bids]
        observed_top_bid = float(max(bid_history) if bid_history else 0.0)
        if current_price is None:
            current_price = observed_top_bid
        if min_required is None:
            # Fallback when caller does not provide current auction constraints.
            if has_current_winner is None:
                has_current_winner = len(bid_history) > 0
            min_required = float(current_price + 1.0) if has_current_winner else float(current_price)
        if reserve_price is None:
            reserve_price = max(1.0, round(value * 0.2, 2))
        affordable_cap = float(state.remaining_budget)
        suggested_cap = min(affordable_cap, value / max(self.beta, 1e-6))

        prompt = (
            "You are bidding in a sequential auction.\n"
            "Return ONLY one number (no words), the bid amount.\n"
            "Return 0 to skip/drop out.\n"
            "If bidding, choose a legal bid >= min_required and <= remaining budget.\n\n"
            f"Item value: {value:.4f}\n"
            f"Item rank: {item.rank}\n"
            f"Reserve price: {float(reserve_price):.4f}\n"
            f"Current price: {float(current_price):.4f}\n"
            f"Observed top bid in history: {observed_top_bid:.4f}\n"
            f"Bid history this item: {bid_history[-6:]}\n"
            f"Current winner exists: {bool(has_current_winner) if has_current_winner is not None else (len(bid_history) > 0)}\n"
            f"Minimum legal next bid (min_required): {float(min_required):.4f}\n"
            f"Your remaining budget: {state.remaining_budget:.4f}\n"
            f"Items remaining including this one: {items_remaining}\n"
            "Objective: maximize long-run utility (value - price), avoid overpaying.\n"
            "Output one numeric bid now (0 or >= min_required):"
        )

        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.MODEL,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": "Output only a numeric bid."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content if resp.choices else ""
            bid = self._extract_first_number(raw or "")
            if bid is None:
                if self.debug:
                    print(
                        f"[LLMBidder {self.bidder_id}] Could not parse numeric bid "
                        f"from model output: {raw!r}"
                    )
                return 0.0
        except Exception as exc:
            if self.debug:
                print(f"[LLMBidder {self.bidder_id}] API error: {exc!r}")
            return 0.0

        if bid < 0:
            if self.debug:
                print(f"[LLMBidder {self.bidder_id}] Negative bid {bid:.4f}, dropping out.")
            return 0.0
        if 0 < bid < float(min_required):
            if self.debug:
                print(
                    f"[LLMBidder {self.bidder_id}] Bid {bid:.4f} below min_required "
                    f"{float(min_required):.4f}, raising to min_required."
                )
            bid = float(min_required)
        if bid > affordable_cap:
            if self.debug:
                print(
                    f"[LLMBidder {self.bidder_id}] Bid {bid:.4f} above budget "
                    f"{affordable_cap:.4f}, clamping."
                )
            bid = affordable_cap
        return float(bid)


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