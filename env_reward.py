"""
Auction environment for multi-agent bidding with RL and LLM opponents.

Items are presented in random order. Turn-based: each step = one agent places
a bid or drops out. Agents accumulate reward (weight/priority) for items won.
Auction ends when all items have been bid on.
"""
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
from model import AuctionModel
from scoring import score_priority_weighted, rank_to_weight
from bidders import (
    BaseBidder,
    BidderState,
    build_opponent_pool,
)

class Item:
    """Auctionable item with market value and bid history."""

    def __init__(self, name: str, value: int, rank: int = 0):
        """
        Args:
            name: Item identifier.
            value: Market value (e.g., list price).
            rank: Priority rank (1 = most wanted). Used by bidders and scoring. 0 = unranked.
        """
        self.name = name
        self.value = value  # Market value (e.g., list price, estimated worth)
        self.rank = rank  # Priority rank: 1 = most wanted, matches bidders & scoring.py
        self.bids: List[float] = []  # Bid amounts placed so far for this item


def _bid_increment(item: Item, ratio: float) -> float:
    """Increment = ratio * item's market value."""
    return ratio * float(item.value)


def round_bid_to_increment(bid: float, item: Item, ratio: float) -> float:
    """Round bid to nearest multiple of increment (ratio * item value)."""
    inc = _bid_increment(item, ratio)
    if inc <= 0:
        return bid
    return round(bid / inc) * inc


def min_bid_to_beat(high_bid: float, item: Item, ratio: float) -> float:
    """Minimum valid bid to beat high_bid (one increment above)."""
    return high_bid + _bid_increment(item, ratio)


class Agent:
    """
    Bidder in the auction. Has item preferences (weights) and accumulates
    reward when winning items. Supports RL or LLM decision models.
    """

    def __init__(
        self,
        name: str,
        priority: List[Item],
        valuations: Optional[Dict[str, float]] = None,
        budget: Optional[float] = None,
        beta: float = 1.0,
        weight_scheme: str = "linear",
    ):
        """
        Args:
            name: Agent identifier.
            priority: Ordered list of items (preference order, first = most wanted).
            valuations: Optional dict mapping item.name -> agent's weight for that item.
                If None, derived from priority index (first item = highest weight).
            budget: Optional budget. If None, reward uses w_i only (no cost penalty).
            beta: Cost sensitivity for heuristic scoring. Higher = more conservative.
            weight_scheme: "linear" or "exponential" for rank_to_weight (matches scoring.py).
        """
        self.name = name
        self.priority = priority  # Preference order for items
        # Weight/importance of each item to this agent (used for reward, not monetary value)
        self.valuations = valuations or {
            item.name: len(priority) - i for i, item in enumerate(priority)
        }
        self.type: Optional[str] = None  # "rl" or "llm" after bind_model
        self.model_params: Dict[str, Any] = {}  # Set by bind_model
        self.accumulated_reward = 0.0  # Sum of scores (w_i - β*p_i/B) for items won
        self.budget = budget
        self.beta = beta
        self.weight_scheme = weight_scheme
        self.remaining_budget: float = budget if budget is not None else float("inf")
        self.items_won: List[Item] = []
        self.prices_paid: List[float] = []

    def get_value(self, item: Item) -> float:
        """Return this agent's weight/valuation for the given item."""
        return self.valuations.get(item.name, 0.0)

    def get_rank(self, item: Item) -> int:
        """Return item's 1-based priority rank (1 = most wanted). 0 if not in priority."""
        for i, p in enumerate(self.priority):
            if p.name == item.name:
                return i + 1
        return 0

    def bind_model(self, model: AuctionModel, modelType: str) -> None:
        """Set the agent's decision model type (rl or llm). Model params stored for future use."""
        self.type = modelType
        self.model = model

class RLAgent(Agent):
    def __init__(self, name: str, priority: List[Item], valuations: Optional[Dict[str, float]] = None, budget: Optional[float] = None, beta: float = 1.0, weight_scheme: str = "linear"):
        super().__init__(name, priority, valuations, budget, beta, weight_scheme)
        model = AuctionModel(input_size=2*len(priority)+3, hidden_size=128, action_size=2)
        self.bind_model(model, "rl")

    def get_action(self, state: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
        state_batch = np.asarray(state, dtype=np.float32)
        if state_batch.ndim == 1:
            state_batch = state_batch.reshape(1, -1)
        mask_t = None
        if mask is not None:
            mask_t = torch.from_numpy(mask).float()
            if mask_t.dim() == 1:
                mask_t = mask_t.unsqueeze(0)
        with torch.no_grad():
            logits = self.model(
                torch.from_numpy(state_batch),
                mask_t,
            )
        action = int(logits.argmax(dim=1).item())
        return action


class OpponentAgent(Agent):
    """
    Wraps a heuristic BaseBidder from bidders.py so it can participate
    in AuctionEnvironment. Uses place_bid(item, state, ...) to choose actions.
    """

    def __init__(
        self,
        name: str,
        priority: List[Item],
        bidder: BaseBidder,
        valuations: Optional[Dict[str, float]] = None,
        beta: float = 1.0,
        weight_scheme: str = "linear",
    ):
        budget = float(bidder.budget)
        super().__init__(
            name=name,
            priority=priority,
            valuations=valuations,
            budget=budget,
            beta=beta,
            weight_scheme=weight_scheme,
        )
        self.bidder = bidder
        self.type = "opponent"

    def get_action(
        self,
        env: "AuctionEnvironment",
        state: Optional[np.ndarray] = None,
    ) -> float:
        """
        Use the wrapped bidder's place_bid to choose a bid for the current item.
        Returns bid amount (>= 0) or -1 to drop out. Called by run_auction when
        it is this agent's turn.
        """
        if env.is_done():
            return -1.0
        current_item = env.item_order[env.current_round]
        items_remaining = len(env.items) - env.current_round
        bidder_state = BidderState(
            budget=self.budget,
            remaining_budget=self.remaining_budget,
            n_items=len(env.items),
            spent=sum(self.prices_paid),
            items_won=list(self.items_won),
            prices_paid=list(self.prices_paid),
            total_score=self.accumulated_reward,
        )
        bid = self.bidder.place_bid(
            current_item,
            bidder_state,
            items_remaining=items_remaining,
            weight_scheme=self.weight_scheme,
        )
        if bid <= 0:
            return -1.0
        ratio = env.bid_increment_ratio
        bid = round_bid_to_increment(float(bid), current_item, ratio)
        high_bid = max(current_item.bids) if current_item.bids else 0.0
        min_valid = min_bid_to_beat(high_bid, current_item, ratio)
        if bid < min_valid:
            bid = min_valid
        return float(bid)


def add_opponents_from_pool(
    env: "AuctionEnvironment",
    items: List[Item],
    n_opponents: int,
    budget: float,
    seed: Optional[int] = None,
    budget_noise: float = 0.2,
) -> None:
    """
    Create heuristic opponents via build_opponent_pool and add them as
    OpponentAgents to the environment. Each opponent uses items as priority
    order (item rank from items is used by the heuristic bidders).
    """
    pool = build_opponent_pool(
        n_opponents=n_opponents,
        budget=budget,
        seed=seed,
        budget_noise=budget_noise,
    )
    for b in pool:
        agent = OpponentAgent(
            name=f"Opponent_{b.bidder_id}",
            priority=items,
            bidder=b,
        )
        env.add_agent(agent)


class AuctionEnvironment:
    """
    Turn-based auction environment. Items are presented in random order.
    Each round auctions one item; agents bid or drop out sequentially.
    """

    def __init__(self, num_agents: int, bid_increment_ratio: float, items: List[Item], rng: Optional[np.random.Generator] = None):
        """
        Args:
            num_agents: Expected number of agents.
            bid_increment_ratio: Bids in increments of this fraction of item value (e.g. 0.1 = 10%).
            items: All items to be auctioned.
            rng: Random generator for reproducible item order. If None, uses default.
        """
        self.num_agents = num_agents
        self.items = items
        self.agents: List[Agent] = []
        self.rng = rng or np.random.default_rng()
        self.item_order: List[Item] = []  # Shuffled at reset; order items are presented
        self.current_round: int = 0  # Index of item currently being auctioned
        self.current_bidder_idx: int = 0  # Which agent is to act (within current round)
        self.dropped_this_round: set = set()  # Agent indices who dropped for current item (reset each round)
        self.bid_increment_ratio = bid_increment_ratio

    def add_agent(self, agent: Agent) -> None:
        """Register an agent to participate in the auction."""
        if len(self.agents) >= self.num_agents:
            raise ValueError(
                f"Cannot add more than {self.num_agents} agents "
                f"(already have {len(self.agents)})."
            )
        self.agents.append(agent)

    def reset(self) -> None:
        """Shuffle item order, clear bid history, and reset agent rewards."""
        self.item_order = self.rng.permutation(self.items).tolist()
        self.current_round = 0
        self.current_bidder_idx = 0
        self.dropped_this_round = set()
        for item in self.items:
            item.bids.clear()
        for agent in self.agents:
            agent.accumulated_reward = 0.0
            agent.remaining_budget = agent.budget if agent.budget is not None else float("inf")
            agent.items_won = []
            agent.prices_paid = []

    def compute_reward(
        self, agent: Agent, item: Item, won: bool, price_paid: float = 0.0
    ) -> float:
        """
        Compute step reward using scoring.py: score = w_i − β * (p_i / B).

        Uses score_priority_weighted from scoring.py when budget is set, so
        weights match rank_to_weight (linear or exponential scheme).

        Args:
            agent: The agent to compute reward for.
            item: The item that was auctioned.
            won: True if this agent won the item.
            price_paid: Price the winner paid (highest bid). Used when budget is set.

        Returns:
            Step reward. With budget: score_priority_weighted. Without budget: w_i (normalized).
        """
        if won:
            n_items = len(self.items)
            max_val = max(agent.valuations.values()) or 1.0
            item_ranks = {
                i: agent.get_rank(i) or getattr(i, "rank", 0)
                for i in self.items
            }
            rank = item_ranks.get(item, 0)
            if agent.budget is not None and agent.budget > 0:
                if rank > 0:
                    r = score_priority_weighted(
                        rank=rank,
                        price=price_paid,
                        n_items=n_items,
                        budget=agent.budget,
                        beta=agent.beta,
                        weight_scheme=agent.weight_scheme,
                    )
                else:
                    # Item not in priority: fallback to w_i - β*(p/B)
                    w_i = agent.get_value(item) / max_val
                    r = w_i - agent.beta * (price_paid / agent.budget)
            else:
                w_i = agent.get_value(item) / max_val
                r = w_i
            agent.accumulated_reward += r
            # Normalize for RL stability (scale to roughly [0, 1] range)
            total_possible = sum(
                rank_to_weight(rk, n_items, agent.weight_scheme)
                for rk in item_ranks.values()
                if rk > 0
            )
            return r / (total_possible or 1.0)
        return 0.0

    def get_current_bidder(self) -> Optional[Agent]:
        """Return the agent whose turn it is to bid, or None if round is over."""
        if self.is_done():
            return None
        return self.agents[self.current_bidder_idx]

    def step(
        self, agent: Agent, action: Union[float, int]
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Process one agent's action (bid or drop out).

        Args:
            agent: The agent taking the action.
            action: Bid amount (float >= 0) or -1 to drop out.

        Returns:
            (reward, done, info) for the acting agent.
            reward: Step reward (from compute_reward if agent won; 0 otherwise).
            done: True if auction is complete.
            info: Dict with 'winner', 'item', etc.
        """
        info: Dict[str, Any] = {}
        reward = 0.0

        if self.is_done():
            return 0.0, True, {"msg": "auction already done"}

        agent_idx = self.agents.index(agent)
        if agent_idx != self.current_bidder_idx:
            return 0.0, False, {"msg": "not this agent's turn"}

        current_item = self.item_order[self.current_round]
        dropped = action < 0  # -1 or negative = drop out

        if dropped:
            self.dropped_this_round.add(agent_idx)
        else:
            ratio = self.bid_increment_ratio
            bid_amount = round_bid_to_increment(float(action), current_item, ratio)
            high_bid = max(current_item.bids) if current_item.bids else 0.0
            min_valid = min_bid_to_beat(high_bid, current_item, ratio)
            # Reject if below min, exceeds budget, or below one increment
            if bid_amount < min_valid or (agent.budget is not None and bid_amount > agent.remaining_budget):
                self.dropped_this_round.add(agent_idx)
            else:
                current_item.bids.append(bid_amount)

        n = len(self.agents)
        active = [i for i in range(n) if i not in self.dropped_this_round]

        # Advance to next bidder (round-robin, skip dropped agents)
        if len(active) > 1:
            for _ in range(n):
                self.current_bidder_idx = (self.current_bidder_idx + 1) % n
                if self.current_bidder_idx not in self.dropped_this_round:
                    break
            info["current_bidder"] = self.agents[self.current_bidder_idx]
            info["item"] = current_item
            info["active_count"] = len(active)

        # Check if only one agent remains -> they win; if all dropped -> no winner
        if len(active) == 1:
            winner_idx = active[0]
            winner = self.agents[winner_idx]
            price_paid = max(current_item.bids) if current_item.bids else 0.0
            winner.remaining_budget -= price_paid
            winner.items_won.append(current_item)
            winner.prices_paid.append(price_paid)
            winner_reward = self.compute_reward(
                winner, current_item, won=True, price_paid=price_paid
            )
            if agent == winner:
                reward = winner_reward
            info["winner"] = winner
            info["item"] = current_item
            info["price_paid"] = price_paid
        elif len(active) == 0:
            info["winner"] = None
            info["item"] = current_item

        if len(active) <= 1:
            # Advance to next round
            self.current_round += 1
            self.current_bidder_idx = 0
            self.dropped_this_round = set()

        done = self.is_done()
        return reward, done, info

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
        - reward_norm [1]: Accumulated score / total possible (can be negative with heuristic scoring)
        - progress [1]: current_round / n_items

        Scalar values are normalized; reward_norm may be negative when using budget-based scoring.
        """
        n_items = len(self.items)
        item_ranks = {
            item: agent.get_rank(item) or getattr(item, "rank", 0)
            for item in self.items
        }

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

        # 4. Agent's valuation for current item (normalized via rank_to_weight)
        max_val = max(agent.valuations.values()) or 1.0
        my_val_current_norm = 0.0
        if self.current_round < n_items:
            current_item = self.item_order[self.current_round]
            rank = item_ranks.get(current_item, 0)
            if rank > 0:
                my_val_current_norm = rank_to_weight(
                    rank, n_items, agent.weight_scheme
                )
            else:
                my_val_current_norm = agent.get_value(current_item) / max_val

        # 5. Agent's valuations for remaining items (fixed-size: one per item, 0 if done)
        # Use rank_to_weight when available for consistency with scoring.py
        vals_remaining = np.zeros(n_items, dtype=np.float32)
        for i, item in enumerate(self.items):
            if items_done[i] == 0:
                rank = item_ranks.get(item, 0)
                if rank > 0:
                    vals_remaining[i] = rank_to_weight(
                        rank, n_items, agent.weight_scheme
                    )
                else:
                    vals_remaining[i] = agent.get_value(item) / max_val

        # 6. Accumulated reward so far (heuristic score or weight-based)
        total_possible_weight = sum(
            rank_to_weight(rk, n_items, agent.weight_scheme)
            for rk in item_ranks.values()
            if rk > 0
        ) or sum(agent.get_value(item) / max_val for item in self.items)
        reward_norm = agent.accumulated_reward / (total_possible_weight or 1.0)

        # 7. Progress: fraction of rounds completed
        progress = self.current_round / (n_items or 1)

        return np.concatenate([
            items_done, # completion status of item[i]
            current_item_onehot, # one-hot encoding of item being auctioned
            [market_value_norm, current_bid_norm, my_val_current_norm], # size 3 -> market, current bid, agent priority for current item
            # vals_remaining, # size n_items
            # [reward_norm, progress], # size 2
        ])

    def is_done(self) -> bool:
        """True if all items have been auctioned (auction complete)."""
        return self.current_round >= len(self.items)

    def get_mask(self, agent: Agent) -> np.ndarray:
        """
        Return action mask for the RL agent (shape (1, 2)): [drop_ok, bid_ok].
        Masks out bid when the agent cannot afford to bid: e.g. $100 remaining
        but current high bid is $500 — can't outbid, so bid is illegal.
        """
        mask = np.ones((1, 2), dtype=np.float32)
        if agent.budget is None:
            return mask
        if agent.remaining_budget <= 0:
            mask[0, 1] = 0.0  # No budget left
            return mask
        # Can't bid if you can't afford to beat the current high bid (one increment)
        if self.current_round < len(self.item_order):
            current_item = self.item_order[self.current_round]
            high_bid = max(current_item.bids) if current_item.bids else 0.0
            min_bid = min_bid_to_beat(high_bid, current_item, self.bid_increment_ratio)
            if agent.remaining_budget < min_bid:
                mask[0, 1] = 0.0  # Can't afford to outbid (e.g. $100 vs $500 high)
        return mask

    def run_auction(self) -> None:
        """Run the auction until it is complete."""
        while not self.is_done():
            agent = self.get_current_bidder()
            if agent is None:
                break
            if agent.type == "rl":
                state = self.get_state(agent)
                mask = self.get_mask(agent)
                raw_action = agent.get_action(state, mask)
                # RL model outputs 0=drop, 1=bid; step() expects float bid or -1
                if raw_action == 0:
                    action = -1.0
                else:
                    current_item = self.item_order[self.current_round]
                    high_bid = max(current_item.bids) if current_item.bids else 0.0
                    ratio = self.bid_increment_ratio
                    min_valid = min_bid_to_beat(high_bid, current_item, ratio)
                    if agent.remaining_budget < min_valid:
                        action = -1.0  # Can't afford; drop
                    else:
                        action = round_bid_to_increment(min_valid, current_item, ratio)
            elif agent.type == "opponent":
                action = agent.get_action(env=self)
            else:
                action = agent.get_action(self.get_state(agent))
            self.step(agent, action)

