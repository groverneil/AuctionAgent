"""
Train RL agent against heuristic bidders, then run an evaluation auction.

Usage:
    source venv/bin/activate
    python run_train.py
"""
import numpy as np
from collections import Counter
from env_reward import (
    Item,
    AuctionEnvironment,
    RLAgent,
    add_opponents_from_pool,
    train_rl_against_heuristics,
)
from bidders import build_opponent_pool

# ---- Config ----
N_ITEMS = 20
BUDGET = 10000
N_OPPONENTS = 8
BID_INCREMENT_RATIO = 0.1
BETA = 0.5
WEIGHT_SCHEME = "linear"
TRAIN_EPISODES = 1000
N_EVAL = 100  # Number of evaluation auctions to run
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

# ---- Create heuristic opponents ----
opponent_bidders = build_opponent_pool(
    n_opponents=N_OPPONENTS,
    budget=BUDGET,
    seed=SEED,
)

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
winners = history.get("winner_per_episode", [])
if winners:
    n_ep = len(winners)
    win_counts = Counter(winners)
    for name in sorted(win_counts.keys(), key=lambda n: (0 if n == "RL_Agent" else 1, n)):
        print(f"  {name}: {win_counts[name]} / {n_ep} wins")

# ---- Evaluation: N_EVAL auctions ----
print(f"\n{'='*60}")
print(f"{'EVALUATION':^60}")
print(f"{'='*60}")
print(f"  Running {N_EVAL} eval auctions...")

# Collect wins per agent per run: {agent_name: [wins_run1, wins_run2, ...]}
wins_per_agent = {}

for i in range(N_EVAL):
    env.reset()
    env.run_auction(save_json=(i == 0), json_path="auction_log.json")  # Save log only for first run
    for agent in env.agents:
        name = agent.name
        if name not in wins_per_agent:
            wins_per_agent[name] = []
        wins_per_agent[name].append(len(agent.items_won))

n_rounds = len(env.items)
agent_heuristic = {a.name: a.bidder.__class__.__name__ for a in env.agents if hasattr(a, "bidder") and a.bidder is not None}

print(f"\n  Eval summary ({N_EVAL} auctions, {n_rounds} rounds each):")
rankings = sorted(
    wins_per_agent.keys(),
    key=lambda n: (-np.mean(wins_per_agent[n]), n),  # sort by mean wins desc, then name
)
for rank, name in enumerate(rankings, 1):
    wins = wins_per_agent[name]
    mean_w = np.mean(wins)
    std_w = np.std(wins)
    pct = 100 * mean_w / n_rounds
    suffix = f" ({agent_heuristic[name]})" if name in agent_heuristic else ""
    print(f"    {rank}. {name}{suffix}: {mean_w:.1f} ± {std_w:.1f} rounds/auction ({pct:.1f}%)")

# Detailed output from last run (rl_agent state is from final iteration)
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

print(f"\nAuction log saved to auction_log.json")
