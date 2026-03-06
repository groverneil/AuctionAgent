"""
Train RL agent against heuristic bidders, then run an evaluation auction.

Usage:
    source venv/bin/activate
    python run_train.py
"""
import numpy as np
from typing import Dict, List
from tqdm import tqdm
from env_reward import (
    Item,
    AuctionEnvironment,
    RLAgent,
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
TRAIN_EPISODES = 2500
N_EVAL = 100  # Number of evaluation auctions to run
CHECKPOINT_EVERY = 100  # Evaluate and save best model every N episodes (0 = off)
CHECKPOINT_EVAL_N = 50  # Auctions per checkpoint eval (higher = more stable best-model selection)
SEEDS = [42, 123, 456]  # Multi-seed for variance reduction; first used for training
SEED = SEEDS[0]

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
    lr=3e-4,  # Lower LR for stability (was 1e-3)
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
    checkpoint_every=CHECKPOINT_EVERY,
    checkpoint_eval_n=CHECKPOINT_EVAL_N,
)

# ---- Print training summary ----
rewards = history["episode_reward"]
window = 50
print(f"\n{'='*60}")
print(f"{'TRAINING RESULTS':^60}")
print(f"{'='*60}")
if CHECKPOINT_EVERY > 0:
    print(f"  (Stats below are from full training run; final model = best checkpoint by eval)")
print(f"  Total episodes:   {len(rewards)}")
print(f"  Final epsilon:    {rl_agent.epsilon:.4f}")
print(f"  First {window} avg:    {np.mean(rewards[:window]):.4f}")
print(f"  Last {window} avg:     {np.mean(rewards[-window:]):.4f}")
print(f"  Overall avg:      {np.mean(rewards):.4f}")
print(f"  Best episode:     {max(rewards):.4f}")
losses = history.get("episode_loss", [])
if losses:
    loss_last50 = np.mean(losses[-window:]) if len(losses) >= window else np.mean(losses)
    print(f"  Loss (last {window}):  {loss_last50:.4f}")

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
            env.run_auction(save_json=(eval_seed == SEEDS[0] and i == 0), json_path="auction_log.json")
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
