"""
Auction environment for multi-agent bidding with RL and LLM opponents.

Items are presented in random order. Turn-based: each step = one agent places
a bid or drops out. Agents accumulate reward (weight/priority) for items won.
Auction ends when all items have been bid on.
"""
from typing import List, Dict, Any, Optional
import numpy as np


class Item:
    """Auctionable item with market value and bid history."""

    def __init__(self, name: str, value: int, rank: int = 0):
        self.name = name
        self.value = value  # Market value (e.g., list price, estimated worth)
        self.rank = rank
        self.bids: List[float] = []  # Bid amounts placed so far for this item

    def get_value(self, mode: str = "linear") -> float:
        """Return item value under the given valuation *mode*."""
        return float(self.value)


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
        self.priority = priority  # Preference order for items
        # Weight/importance of each item to this agent (used for reward, not monetary value)
        self.valuations = valuations or {
            item.name: len(priority) - i for i, item in enumerate(priority)
        }
        self.type: Optional[str] = None  # "rl" or "llm" after bind_model
        self.accumulated_reward = 0.0  # Sum of item weights won (not monetary value)

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
        """
        Args:
            num_agents: Expected number of agents.
            items: All items to be auctioned.
            rng: Random generator for reproducible item order. If None, uses default.
        """
        self.num_agents = num_agents
        self.items = items
        self.agents: List[Agent] = []
        self.rng = rng or np.random.default_rng()
        self.item_order: List[Item] = []  # Shuffled at reset; order items are presented
        self.current_round: int = 0  # Index of item currently being auctioned

    def add_agent(self, agent: Agent) -> None:
        """Register an agent to participate in the auction."""
        self.agents.append(agent)

    def reset(self) -> None:
        """Shuffle item order, clear bid history, and reset agent rewards."""
        self.item_order = self.rng.permutation(self.items).tolist()
        self.current_round = 0
        for item in self.items:
            item.bids.clear()
        for agent in self.agents:
            agent.accumulated_reward = 0.0

    def get_state(self, agent: Agent) -> np.ndarray:
        """
        Encode the current auction state as a fixed-size vector for the RL agent.

        State vector layout (size = 4*n_items + 5):
        - items_done [n_items]: 1 if item auctioned, 0 otherwise
        - current_item_onehot [n_items]: One-hot encoding of item being auctioned
        - market_value_norm [1]: Current item's market value / max market value
        - current_bid_norm [1]: Highest bid on current item / max market value
        - my_val_current_norm [1]: Agent's weight for current item (normalized)
        - vals_remaining [n_items]: Agent's weights for remaining items, 0 if done
        - reward_norm [1]: Accumulated weight won / total possible weight
        - progress [1]: current_round / n_items

        All scalar values are normalized to [0, 1] for stability.
        """
        n_items = len(self.items)

        # 1. Which items have been auctioned (done) vs remaining
        items_done = np.zeros(n_items, dtype=np.float32)
        for i, item in enumerate(self.items):
            if item in self.item_order[: self.current_round]:
                items_done[i] = 1.0

        # 2. Current item being auctioned (one-hot)
        current_item_onehot = np.zeros(n_items, dtype=np.float32)
        if self.current_round < n_items:
            current_item = self.item_order[self.current_round]
            idx = self.items.index(current_item)
            current_item_onehot[idx] = 1.0

        # 3. Market value and current bid (both normalized by max market value)
        #    Same denominator preserves ratio: e.g. bid=300, value=400 -> 0.75
        market_value = 0.0
        current_bid = 0.0
        if self.current_round < n_items:
            current_item = self.item_order[self.current_round]
            market_value = float(current_item.value)
            current_bid = max(current_item.bids) if current_item.bids else 0.0
        max_market = max((i.value for i in self.items), default=1)
        market_value_norm = market_value / (max_market or 1.0)
        current_bid_norm = current_bid / (max_market or 1.0)

        # 4. Agent's valuation for current item (normalized)
        my_val_current = 0.0
        if self.current_round < n_items:
            current_item = self.item_order[self.current_round]
            my_val_current = agent.get_value(current_item)
        max_val = max(agent.valuations.values()) or 1.0
        my_val_current_norm = my_val_current / max_val

        # 5. Agent's valuations for remaining items (fixed-size: one per item, 0 if done)
        vals_remaining = np.zeros(n_items, dtype=np.float32)
        for i, item in enumerate(self.items):
            if items_done[i] == 0:
                vals_remaining[i] = agent.get_value(item) / max_val

        # 6. Accumulated reward so far (weight-based, normalized by total possible weight)
        total_possible_weight = sum(agent.get_value(item) for item in self.items)
        reward_norm = agent.accumulated_reward / (total_possible_weight or 1.0)

        # 7. Progress: fraction of rounds completed
        progress = self.current_round / (n_items or 1)

        return np.concatenate([
            items_done,
            current_item_onehot,
            [market_value_norm, current_bid_norm, my_val_current_norm],
            vals_remaining,
            [reward_norm, progress],
        ])

    def is_done(self) -> bool:
        """True if all items have been auctioned (auction complete)."""
        return self.current_round >= len(self.items)

