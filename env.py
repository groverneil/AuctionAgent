from typing import List, Dict, Any, Optional
import numpy as np

class Item:
    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value
        self.bids = []

class Agent:
    def __init__(self, name: str, priority: List[Item], valuations: Optional[Dict[str, float]] = None):
        """
        priority: ordered list of items (preference order)
        valuations: optional dict mapping item.name -> agent's value for that item.
                    If None, derived from priority index (first = highest value).
        """
        self.name = name
        self.priority = priority
        self.valuations = valuations or {
            item.name: len(priority) - i for i, item in enumerate(priority)
        }
        self.type = None
        self.accumulated_reward = 0.0  # sum of item weights won (not monetary value)

    def get_value(self, item: Item) -> float:
        return self.valuations.get(item.name, 0.0)

    def bind_model(self, model: Any, params: Dict[str, Any]):
        if model["type"] == "rl":
            self.type = "rl"
        else:
            self.type = "llm"

class AuctionEnvironment:
    def __init__(self, num_agents: int, items: List[Item], rng: Optional[np.random.Generator] = None):
        self.num_agents = num_agents
        self.items = items
        self.agents = []
        self.rng = rng or np.random.default_rng()
        # Item order is randomized when auction starts
        self.item_order: List[Item] = []
        self.current_round: int = 0

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def reset(self) -> None:
        """Shuffle item order and reset agent rewards (accumulated weight)."""
        self.item_order = self.rng.permutation(self.items).tolist()
        self.current_round = 0
        for agent in self.agents:
            agent.accumulated_reward = 0.0

    def get_state(self, agent: Agent) -> np.ndarray:
        """
        Encode state for RL agent. Items presented at random; agent accumulates
        reward = item weight (priority) when round ends; auction ends when all
        items have been bid on.
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

        # 3. Agent's valuation for current item (normalized)
        my_val_current = 0.0
        if self.current_round < n_items:
            current_item = self.item_order[self.current_round]
            my_val_current = agent.get_value(current_item)
        max_val = max(agent.valuations.values()) or 1.0
        my_val_current_norm = my_val_current / max_val

        # 4. Agent's valuations for remaining items (fixed-size: one per item, 0 if done)
        vals_remaining = np.zeros(n_items, dtype=np.float32)
        for i, item in enumerate(self.items):
            if items_done[i] == 0:
                vals_remaining[i] = agent.get_value(item) / max_val

        # 5. Accumulated reward so far (weight-based, normalized by total possible weight)
        total_possible_weight = sum(agent.get_value(item) for item in self.items)
        reward_norm = agent.accumulated_reward / (total_possible_weight or 1.0)

        # 6. Progress: fraction of rounds completed
        progress = self.current_round / (n_items or 1)

        return np.concatenate([
            items_done,
            current_item_onehot,
            [my_val_current_norm],
            vals_remaining,
            [reward_norm, progress],
        ])

    def is_done(self) -> bool:
        return self.current_round >= len(self.items)

