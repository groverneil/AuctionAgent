"""
Evaluate a saved RL model against heuristic bidders (plus an LLM opponent).

Usage:
    source venv/bin/activate
    python eval_LLM.py
"""
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import torch
from typing import Dict, List
from tqdm import tqdm
from env_reward import (
    Item,
    AuctionEnvironment,
    RLAgent,
    add_opponents_from_pool,
    OpponentAgent,
)
from bidders import LLMBidder

LLM_CALL_COUNT = 0

_ORIG_CALL_LLM = LLMBidder._call_llm


def _call_llm_with_count(self, prompt: str):
    global LLM_CALL_COUNT
    LLM_CALL_COUNT += 1
    return _ORIG_CALL_LLM(self, prompt)


LLMBidder._call_llm = _call_llm_with_count

# ---- Config (must match run_train.py) ----
MODEL_TYPE = "lstm"  # Must match model used when training
N_ITEMS = 20
BUDGET = 10000
N_OPPONENTS = 8
BID_INCREMENT_RATIO = 0.1
BETA = 0.5
WEIGHT_SCHEME = "linear"
N_LLM_OPPONENTS = 1
N_HEURISTIC_OPPONENTS = N_OPPONENTS - N_LLM_OPPONENTS
N_EVAL = 3
SEEDS = [42]
SEED = SEEDS[0]
SAVE_PATH = "auction_model.pt"

# ---- Create items with ranks (1 = most wanted) ----
rng = np.random.default_rng(SEED)
items = [
    Item(name=f"item_{i+1}", value=int(rng.integers(10, 100)), rank=i + 1)
    for i in range(N_ITEMS)
]

# ---- Create RL agent and load saved weights ----
rl_agent = RLAgent(
    name="RL_Agent",
    priority=items,
    budget=BUDGET,
    beta=BETA,
    weight_scheme=WEIGHT_SCHEME,
    model_type=MODEL_TYPE,
)
rl_agent.model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
print(f"Loaded {MODEL_TYPE} model from {SAVE_PATH}")

# ---- Build environment ----
env = AuctionEnvironment(
    num_agents=1 + N_OPPONENTS,
    bid_increment_ratio=BID_INCREMENT_RATIO,
    items=items,
    rng=np.random.default_rng(SEED),
)
env.add_agent(rl_agent)
for i in range(N_LLM_OPPONENTS):
    env.add_agent(
        OpponentAgent(
            name=f"LLM_{i+1}",
            priority=items,
            bidder=LLMBidder(
                bidder_id=N_HEURISTIC_OPPONENTS + i + 1,
                budget=BUDGET,
                debug=False,
            ),
        )
    )

add_opponents_from_pool(
    env,
    items,
    n_opponents=N_HEURISTIC_OPPONENTS,
    budget=BUDGET,
    seed=SEED,
)


# ---- Evaluation: N_EVAL auctions per seed (multi-seed for variance) ----
print(f"\n{'='*60}")
print(f"{'EVALUATION':^60}")
print(f"{'='*60}")
print(f"  Running {N_EVAL} eval auctions × {len(SEEDS)} seeds...")

all_wins: Dict[str, List[float]] = {}

total_eval = N_EVAL * len(SEEDS)
with tqdm(total=total_eval, desc="Evaluation", unit="auction") as pbar:
    for eval_seed in SEEDS:
        env.rng = np.random.default_rng(eval_seed)
        for i in range(N_EVAL):
            env.reset()
            env.run_auction(save_json=(eval_seed == SEEDS[0] and i == 0), json_path="auction_log_LLM.json")
            for agent in env.agents:
                name = agent.name
                if name not in all_wins:
                    all_wins[name] = []
                all_wins[name].append(len(agent.items_won))
            pbar.update(1)

n_rounds = len(env.items)
agent_heuristic = {a.name: a.bidder.__class__.__name__ for a in env.agents if hasattr(a, "bidder") and a.bidder is not None}

print(f"\n  Eval summary ({N_EVAL * len(SEEDS)} auctions, {n_rounds} rounds each):")
rankings = sorted(
    all_wins.keys(),
    key=lambda n: (-np.mean(all_wins[n]), n),
)
for rank, name in enumerate(rankings, 1):
    wins = all_wins[name]
    mean_w = np.mean(wins)
    std_w = np.std(wins)
    pct = 100 * mean_w / n_rounds
    suffix = f" ({agent_heuristic[name]})" if name in agent_heuristic else ""
    print(f"    {rank}. {name}{suffix}: {mean_w:.1f} ± {std_w:.1f} rounds/auction ({pct:.1f}%)")

print(f"\n  Sample run (last auction):")
print(f"    Items in auction order: {[i.name for i in env.item_order]}")
print(f"\n  RL Agent:")
print(f"    Items won:   {len(rl_agent.items_won)}")
print(f"    Items:       {[i.name for i in rl_agent.items_won]}")
print(f"    Ranks won:   {[i.rank for i in rl_agent.items_won]}")
print(f"    Prices paid: {[round(p, 2) for p in rl_agent.prices_paid]}")
print(f"    Budget left: {rl_agent.remaining_budget:.2f}")
print(f"    Total score: {rl_agent.accumulated_reward:.4f}")

for agent in env.agents[1:]:
    heuristic = agent.bidder.__class__.__name__
    print(f"\n  {agent.name} ({heuristic}):")
    print(f"    Items won:   {len(agent.items_won)}")
    print(f"    Ranks won:   {[i.rank for i in agent.items_won]}")
    print(f"    Prices paid: {[round(p, 2) for p in agent.prices_paid]}")
    print(f"    Budget left: {agent.remaining_budget:.2f}")
    print(f"    Total score: {agent.accumulated_reward:.4f}")

print(f"\nAuction log saved to auction_log_LLM.json")
print(f"Total LLM calls (LLMBidder._call_llm): {LLM_CALL_COUNT}")
