"""
Train RL agent against heuristic bidders, then run an evaluation auction.

Usage:
    source venv/bin/activate
    python run_train.py
"""
import numpy as np
from env_reward import (
    Item,
    AuctionEnvironment,
    RLAgent,
    add_opponents_from_pool,
    train_rl_against_heuristics,
)
from bidders import TopKSpecialistBidder

# ---- Config ----
N_ITEMS = 20
BUDGET = 100.0
N_OPPONENTS = 1
BID_INCREMENT_RATIO = 0.1
BETA = 0.3
WEIGHT_SCHEME = "linear"
TRAIN_EPISODES = 500
SEED = 42

# ---- Create items with ranks (1 = most wanted) ----
rng = np.random.default_rng(SEED)
items = [
    Item(name=f"item_{i+1}", value=int(rng.integers(10, 100)), rank=i + 1)
    for i in range(N_ITEMS)
]

# ---- Build environment ----
env = AuctionEnvironment(
    num_agents=1 + N_OPPONENTS,
    bid_increment_ratio=BID_INCREMENT_RATIO,
    items=items,
    rng=np.random.default_rng(SEED),
)

# ---- Create RL agent ----
rl_agent = RLAgent(
    name="RL_Agent",
    priority=items,
    budget=BUDGET,
    beta=BETA,
    weight_scheme=WEIGHT_SCHEME,
    lr=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
)

# ---- Create heuristic opponents (single TopK specialist) ----
opponent_bidders = [
    TopKSpecialistBidder(
        bidder_id=1,
        budget=BUDGET,
        beta=BETA,
        top_k=5,
        margin=0.0,
    ),
]

# ---- Train ----
print(f"Training for {TRAIN_EPISODES} episodes against {N_OPPONENTS} heuristic opponents...")
print(f"Items: {N_ITEMS}, Budget: {BUDGET}, Beta: {BETA}\n")

history = train_rl_against_heuristics(
    env=env,
    rl_agent=rl_agent,
    heuristic_bidders=opponent_bidders,
    episodes=TRAIN_EPISODES,
    seed=SEED,
)

# ---- Print training summary ----
rewards = history["episode_reward"]
window = 50
print(f"\n{'='*60}")
print(f"{'TRAINING RESULTS':^60}")
print(f"{'='*60}")
print(f"  Total episodes:   {len(rewards)}")
print(f"  Final epsilon:    {rl_agent.epsilon:.4f}")
print(f"  First {window} avg:    {np.mean(rewards[:window]):.4f}")
print(f"  Last {window} avg:     {np.mean(rewards[-window:]):.4f}")
print(f"  Overall avg:      {np.mean(rewards):.4f}")
print(f"  Best episode:     {max(rewards):.4f}")

# ---- Evaluation run ----
print(f"\n{'='*60}")
print(f"{'EVALUATION AUCTION':^60}")
print(f"{'='*60}")
env.reset()
env.run_auction(save_json=True, json_path="auction_log.json")

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

print(f"\nAuction log saved to auction_log.json")
