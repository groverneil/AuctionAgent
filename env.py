"""
Auction environment for multi-agent bidding with RL and LLM opponents.

Items are presented in random order. Turn-based: each step = one agent places
a bid or drops out. Agents accumulate reward (weight/priority) for items won.
Auction ends when all items have been bid on.
"""
from typing import List, Dict, Any, Optional, Iterable, Tuple
import numpy as np


class Item:
    """Auctionable item with base market value, bid history, and coalition value uplift."""

    def __init__(self, name: str, value: int):
        self.name = name
        self.base_value = int(value)              # Original market value (fixed reference)
        self._coalition_uplift = 0.0              # Max coalition bid seen so far (added into value)
        self.bids: List[float] = []               # Scalar bid history (single bids or coalition-max bids)
        self.bid_meta: List[Dict[str, Any]] = []  # Optional: richer bid logs (who/coalition/etc.)

    @property
    def value(self) -> float:
        """Effective value used by the environment."""
        return float(self.base_value) + float(self._coalition_uplift)

    def clear_history(self) -> None:
        self.bids.clear()
        self.bid_meta.clear()
        self._coalition_uplift = 0.0

    def record_bid(
        self,
        amount: float,
        *,
        bidders: Optional[List[str]] = None,
        is_coalition: bool = False,
        raw_amounts: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record a bid.

        If is_coalition=True, the item value is uplifted by the max coalition bid
        seen so far (incorporated into effective value).
        """
        amount = float(amount)
        self.bids.append(amount)
        self.bid_meta.append({
            "amount": amount,
            "bidders": bidders or [],
            "is_coalition": bool(is_coalition),
            "raw_amounts": raw_amounts or {},
        })

        if is_coalition:
            # Incorporate the *max* coalition bid into the item value (monotonic uplift).
            # If you instead want to *add each time*, replace max(...) with += amount.
            self._coalition_uplift = max(self._coalition_uplift, amount)


class Agent:
    """
    Bidder in the auction. Has item preferences (weights) and accumulates
    reward when winning items. Supports RL or LLM decision models.
    """

    def __init__(self, name: str, priority: List[Item], valuations: Optional[Dict[str, float]] = None):
        """
        Args:
            name: Agent identifier.
            priority: Ordered list of items (preference order, first = most wanted).
            valuations: Optional dict mapping item.name -> agent's weight for that item.
                If None, derived from priority index (first item = highest weight).
        """
        self.name = name
        self.priority = priority
        self.valuations = valuations or {
            item.name: len(priority) - i for i, item in enumerate(priority)
        }
        self.type: Optional[str] = None  # "rl" or "llm" after bind_model
        self.accumulated_reward = 0.0

    def get_value(self, item: Item) -> float:
        """Return this agent's weight/valuation for the given item."""
        return self.valuations.get(item.name, 0.0)

    def bind_model(self, model: Any, params: Dict[str, Any]) -> None:
        """Set the agent's decision model type (rl or llm). Model params stored for future use."""
        if model["type"] == "rl":
            self.type = "rl"
        else:
            self.type = "llm"


class AuctionEnvironment:
    """
    Turn-based auction environment. Items are presented in random order.
    Each round auctions one item; agents bid or drop out sequentially.
    """

    def __init__(self, num_agents: int, items: List[Item], rng: Optional[np.random.Generator] = None):
        self.num_agents = num_agents
        self.items = items
        self.agents: List[Agent] = []
        self.rng = rng or np.random.default_rng()
        self.item_order: List[Item] = []
        self.current_round: int = 0

    def add_agent(self, agent: Agent) -> None:
        self.agents.append(agent)

    def reset(self) -> None:
        """Shuffle item order, clear bid history, and reset agent rewards."""
        self.item_order = self.rng.permutation(self.items).tolist()
        self.current_round = 0
        for item in self.items:
            item.clear_history()
        for agent in self.agents:
            agent.accumulated_reward = 0.0

    # ---------------------------
    # NEW: bidding helpers
    # ---------------------------
    def _current_item(self) -> Optional[Item]:
        if self.current_round >= len(self.items):
            return None
        return self.item_order[self.current_round]

    def place_bid(self, agent: Agent, amount: float) -> None:
        """
        Single-agent bid on the current item.
        Keeps backwards compatibility: a single bid does NOT change item value uplift.
        """
        item = self._current_item()
        if item is None:
            raise RuntimeError("Auction is done; no current item to bid on.")
        item.record_bid(float(amount), bidders=[agent.name], is_coalition=False)

    def place_coalition_bid(self, agents: Iterable[Agent], amounts: Dict[str, float]) -> float:
        """
        Coalition bid on the current item.

        amounts: dict agent.name -> bid_amount for members participating.
        The coalition submits one effective bid equal to max(member_bid_amounts).
        That max is also incorporated into the item's effective value (uplift).
        """
        item = self._current_item()
        if item is None:
            raise RuntimeError("Auction is done; no current item to bid on.")

        coalition_agents = list(agents)
        coalition_names = [a.name for a in coalition_agents]

        # Filter to coalition members if amounts includes extra keys
        member_amounts = {k: float(v) for k, v in amounts.items() if k in set(coalition_names)}
        if not member_amounts:
            raise ValueError("Coalition bid must include at least one participating agent with an amount.")

        max_bid = max(member_amounts.values())

        # Record as a single bid + uplift item value based on max coalition bid
        item.record_bid(
            max_bid,
            bidders=coalition_names,
            is_coalition=True,
            raw_amounts=member_amounts,
        )
        return max_bid

    # ---------------------------
    # State
    # ---------------------------
    def get_state(self, agent: Agent) -> np.ndarray:
        """
        Encode the current auction state as a fixed-size vector for the RL agent.

        State vector layout (size = 4*n_items + 5):
        - items_done [n_items]
        - current_item_onehot [n_items]
        - market_value_norm [1]: CURRENT item's effective value / max effective value
        - current_bid_norm [1]: Highest bid on current item / max effective value
        - my_val_current_norm [1]
        - vals_remaining [n_items]
        - reward_norm [1]
        - progress [1]
        """
        n_items = len(self.items)

        # 1) Which items have been auctioned
        items_done = np.zeros(n_items, dtype=np.float32)
        for i, item in enumerate(self.items):
            if item in self.item_order[: self.current_round]:
                items_done[i] = 1.0

        # 2) Current item one-hot
        current_item_onehot = np.zeros(n_items, dtype=np.float32)
        current_item = None
        if self.current_round < n_items:
            current_item = self.item_order[self.current_round]
            idx = self.items.index(current_item)
            current_item_onehot[idx] = 1.0

        # 3) Effective market value + current bid (normalize by max effective value across items)
        market_value = 0.0
        current_bid = 0.0
        if current_item is not None:
            market_value = float(current_item.value)  # <-- effective value (base + coalition uplift)
            current_bid = max(current_item.bids) if current_item.bids else 0.0

        max_market = max((float(i.value) for i in self.items), default=1.0)  # <-- max effective value
        denom = max_market or 1.0
        market_value_norm = market_value / denom
        current_bid_norm = current_bid / denom

        # 4) Agent valuation for current item (normalized)
        my_val_current = 0.0
        if current_item is not None:
            my_val_current = agent.get_value(current_item)
        max_val = max(agent.valuations.values()) or 1.0
        my_val_current_norm = my_val_current / max_val

        # 5) Valuations remaining
        vals_remaining = np.zeros(n_items, dtype=np.float32)
        for i, item in enumerate(self.items):
            if items_done[i] == 0:
                vals_remaining[i] = agent.get_value(item) / max_val

        # 6) Accumulated reward normalized
        total_possible_weight = sum(agent.get_value(item) for item in self.items)
        reward_norm = agent.accumulated_reward / (total_possible_weight or 1.0)

        # 7) Progress
        progress = self.current_round / (n_items or 1)

        return np.concatenate([
            items_done,
            current_item_onehot,
            [market_value_norm, current_bid_norm, my_val_current_norm],
            vals_remaining,
            [reward_norm, progress],
        ])

    def is_done(self) -> bool:
        return self.current_round >= len(self.items)
