"""
Evaluate a saved RL model against heuristic bidders (plus an LLM opponent).

Usage:
    source venv/bin/activate
    python eval_LLM.py
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from bidders import LLMBidder
from env_reward import (
    AuctionEnvironment,
    Item,
    OpponentAgent,
    RLAgent,
    add_opponents_from_pool,
)

load_dotenv()

# ---- Config (must match run_train.py) ----
N_ITEMS = 20
BUDGET = 10000
N_OPPONENTS = 8
BID_INCREMENT_RATIO = 0.1
BETA = 0.5
WEIGHT_SCHEME = "linear"
N_LLM_OPPONENTS = 1
N_HEURISTIC_OPPONENTS = N_OPPONENTS - N_LLM_OPPONENTS
N_EVAL = 10
SEEDS = [42]
SEED = SEEDS[0]
SAVE_PATH = "auction_model.pt"
LOG_PATH = "auction_log_LLM.json"


@dataclass
class AgentSnapshot:
    name: str
    heuristic: Optional[str]
    items_won: List[str]
    ranks_won: List[int]
    prices_paid: List[float]
    budget_left: float
    total_score: float


@dataclass
class AuctionResult:
    eval_seed: int
    eval_idx: int
    wins: Dict[str, int]
    item_order: List[str]
    agent_snapshots: Dict[str, AgentSnapshot]
    llm_call_count: int
    log_payload: Optional[Dict[str, Any]]


def build_item_specs(seed: int) -> List[Tuple[str, int, int]]:
    rng = np.random.default_rng(seed)
    return [
        (f"item_{i+1}", int(rng.integers(10, 100)), i + 1)
        for i in range(N_ITEMS)
    ]


ITEM_SPECS = build_item_specs(SEED)
MODEL_STATE = torch.load(SAVE_PATH, map_location="cpu", weights_only=True)


def clone_items() -> List[Item]:
    return [
        Item(name=name, value=value, rank=rank)
        for name, value, rank in ITEM_SPECS
    ]


def derive_seed(*parts: int) -> int:
    seed_seq = np.random.SeedSequence(parts)
    return int(seed_seq.generate_state(1, dtype=np.uint32)[0])


def build_eval_environment(auction_seed: int) -> AuctionEnvironment:
    items = clone_items()
    env = AuctionEnvironment(
        num_agents=1 + N_OPPONENTS,
        bid_increment_ratio=BID_INCREMENT_RATIO,
        items=items,
        rng=np.random.default_rng(auction_seed),
    )

    rl_agent = RLAgent(
        name="RL_Agent",
        priority=items,
        budget=BUDGET,
        beta=BETA,
        weight_scheme=WEIGHT_SCHEME,
    )
    rl_agent.model.load_state_dict(MODEL_STATE)
    rl_agent.model.eval()
    env.add_agent(rl_agent)

    for i in range(N_LLM_OPPONENTS):
        bidder_seed = derive_seed(auction_seed, N_HEURISTIC_OPPONENTS + i + 1)
        bidder = LLMBidder(
            bidder_id=N_HEURISTIC_OPPONENTS + i + 1,
            budget=BUDGET,
            debug=False,
        )
        bidder.set_seed(bidder_seed)
        env.add_agent(
            OpponentAgent(
                name=f"LLM_{i+1}",
                priority=items,
                bidder=bidder,
            )
        )

    add_opponents_from_pool(
        env,
        items,
        n_opponents=N_HEURISTIC_OPPONENTS,
        budget=BUDGET,
        seed=SEED,
    )
    return env


def snapshot_agent(agent) -> AgentSnapshot:
    heuristic = None
    if hasattr(agent, "bidder") and agent.bidder is not None:
        heuristic = agent.bidder.__class__.__name__
    return AgentSnapshot(
        name=agent.name,
        heuristic=heuristic,
        items_won=[item.name for item in agent.items_won],
        ranks_won=[item.rank for item in agent.items_won],
        prices_paid=[round(price, 2) for price in agent.prices_paid],
        budget_left=float(agent.remaining_budget),
        total_score=float(agent.accumulated_reward),
    )


async def evaluate_auction(
    eval_seed: int,
    eval_idx: int,
    capture_payload: bool,
) -> AuctionResult:
    auction_seed = derive_seed(eval_seed, eval_idx)
    env = build_eval_environment(auction_seed)
    env.reset()
    log_payload = await env.run_auction_async(
        return_payload=capture_payload,
        json_path=LOG_PATH,
    )

    wins: Dict[str, int] = {}
    agent_snapshots: Dict[str, AgentSnapshot] = {}
    llm_call_count = 0

    for agent in env.agents:
        wins[agent.name] = len(agent.items_won)
        agent_snapshots[agent.name] = snapshot_agent(agent)
        bidder = getattr(agent, "bidder", None)
        if isinstance(bidder, LLMBidder):
            llm_call_count += bidder.call_count

    return AuctionResult(
        eval_seed=eval_seed,
        eval_idx=eval_idx,
        wins=wins,
        item_order=[item.name for item in env.item_order],
        agent_snapshots=agent_snapshots,
        llm_call_count=llm_call_count,
        log_payload=log_payload,
    )


def print_agent_snapshot(snapshot: AgentSnapshot) -> None:
    print(f"    Items won:   {len(snapshot.items_won)}")
    if snapshot.heuristic is None:
        print(f"    Items:       {snapshot.items_won}")
    print(f"    Ranks won:   {snapshot.ranks_won}")
    print(f"    Prices paid: {snapshot.prices_paid}")
    print(f"    Budget left: {snapshot.budget_left:.2f}")
    print(f"    Total score: {snapshot.total_score:.4f}")


async def main() -> None:
    print(f"Loaded model weights from {SAVE_PATH}")
    print(f"\n{'='*60}")
    print(f"{'EVALUATION':^60}")
    print(f"{'='*60}")
    print(f"  Running {N_EVAL} eval auctions × {len(SEEDS)} seeds...")

    sample_identity = (SEEDS[0], 0)
    all_wins: Dict[str, List[float]] = {}
    all_results: List[AuctionResult] = []
    total_llm_calls = 0

    total_eval = N_EVAL * len(SEEDS)
    with tqdm(total=total_eval, desc="Evaluation", unit="auction") as pbar:
        for eval_seed in SEEDS:
            tasks = [
                asyncio.create_task(
                    evaluate_auction(
                        eval_seed=eval_seed,
                        eval_idx=eval_idx,
                        capture_payload=(eval_seed, eval_idx) == sample_identity,
                    )
                )
                for eval_idx in range(N_EVAL)
            ]
            for task in asyncio.as_completed(tasks):
                result = await task
                all_results.append(result)
                total_llm_calls += result.llm_call_count
                for name, wins in result.wins.items():
                    all_wins.setdefault(name, []).append(wins)
                pbar.update(1)

    sample_result = next(
        (
            result
            for result in all_results
            if (result.eval_seed, result.eval_idx) == sample_identity
        ),
        None,
    )
    if sample_result is None:
        raise RuntimeError("Sample auction result was not collected.")

    if sample_result.log_payload is not None:
        with open(LOG_PATH, "w", encoding="utf-8") as fp:
            json.dump(sample_result.log_payload, fp, indent=2)

    n_rounds = len(ITEM_SPECS)
    agent_heuristic = {
        name: snapshot.heuristic
        for name, snapshot in sample_result.agent_snapshots.items()
        if snapshot.heuristic is not None
    }

    print(f"\n  Eval summary ({N_EVAL * len(SEEDS)} auctions, {n_rounds} rounds each):")
    rankings = sorted(
        all_wins.keys(),
        key=lambda name: (-np.mean(all_wins[name]), name),
    )
    for rank, name in enumerate(rankings, 1):
        wins = all_wins[name]
        mean_w = np.mean(wins)
        std_w = np.std(wins)
        pct = 100 * mean_w / n_rounds
        suffix = f" ({agent_heuristic[name]})" if name in agent_heuristic else ""
        print(f"    {rank}. {name}{suffix}: {mean_w:.1f} ± {std_w:.1f} rounds/auction ({pct:.1f}%)")

    print(f"\n  Sample run (seed {sample_result.eval_seed}, auction {sample_result.eval_idx + 1}):")
    print(f"    Items in auction order: {sample_result.item_order}")
    print(f"\n  RL Agent:")
    print_agent_snapshot(sample_result.agent_snapshots["RL_Agent"])

    for name, snapshot in sample_result.agent_snapshots.items():
        if name == "RL_Agent":
            continue
        heuristic = snapshot.heuristic or "UnknownBidder"
        print(f"\n  {name} ({heuristic}):")
        print_agent_snapshot(snapshot)

    print(f"\nAuction log saved to {LOG_PATH}")
    print(f"Total LLM calls (LLMBidder._call_llm / _call_llm_async): {total_llm_calls}")


if __name__ == "__main__":
    asyncio.run(main())
