"""
Evaluate RL model vs LLM — LM Studio or MLX (local mlx_lm).

    python eval_mlx_llm.py --backend lmstudio   # LM Studio (localhost:1234), default
    python eval_mlx_llm.py --backend mlx        # MLX bidder (mlx_lm, no server)
    python eval_mlx_llm.py --fast               # 5 items, 3 opponents
    python eval_mlx_llm.py --quiet
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from bidders import LMStudioBidder, MLXBidder
from env_reward import (
    AuctionEnvironment,
    Item,
    OpponentAgent,
    RLAgent,
    add_opponents_from_pool,
)

load_dotenv()

# ---- Config ----
N_ITEMS = 20
BUDGET = 10000
N_OPPONENTS = 8
BID_INCREMENT_RATIO = 0.1
BETA = 0.5
WEIGHT_SCHEME = "linear"
N_LLM_OPPONENTS = 1
N_HEURISTIC_OPPONENTS = N_OPPONENTS - N_LLM_OPPONENTS
N_EVAL = 100
SEEDS = [42]
SEED = SEEDS[0]
SAVE_PATH = "auction_model.pt"
LOG_PATH = "auction_log_LLM.json"

FAST_ITEMS = 5
FAST_OPPONENTS = 3


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
    llm_call_durations: List[float]
    log_payload: Optional[Dict[str, Any]]


def build_item_specs(seed: int, n_items: int = N_ITEMS) -> List[Tuple[str, int, int]]:
    rng = np.random.default_rng(seed)
    return [
        (f"item_{i+1}", int(rng.integers(10, 100)), i + 1)
        for i in range(n_items)
    ]


def _get_config(fast: bool):
    n_items = FAST_ITEMS if fast else N_ITEMS
    n_opponents = FAST_OPPONENTS if fast else N_OPPONENTS
    n_heuristic = n_opponents - N_LLM_OPPONENTS
    return n_items, n_opponents, n_heuristic


ITEM_SPECS = build_item_specs(SEED)
MODEL_STATE = torch.load(SAVE_PATH, map_location="cpu", weights_only=True)


def clone_items(item_specs: Optional[List[Tuple[str, int, int]]] = None) -> List[Item]:
    specs = item_specs or ITEM_SPECS
    return [
        Item(name=name, value=value, rank=rank)
        for name, value, rank in specs
    ]


def derive_seed(*parts: int) -> int:
    seed_seq = np.random.SeedSequence(parts)
    return int(seed_seq.generate_state(1, dtype=np.uint32)[0])


def build_eval_environment(
    auction_seed: int,
    n_items: int = N_ITEMS,
    n_opponents: int = N_OPPONENTS,
    n_heuristic: int = N_HEURISTIC_OPPONENTS,
    debug: bool = False,
    backend: str = "lmstudio",
) -> AuctionEnvironment:
    item_specs = build_item_specs(auction_seed, n_items)
    items = clone_items(item_specs)
    env = AuctionEnvironment(
        num_agents=1 + n_opponents,
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
        bidder_seed = derive_seed(auction_seed, n_heuristic + i + 1)
        if backend == "mlx":
            bidder = MLXBidder(
                bidder_id=n_heuristic + i + 1,
                budget=BUDGET,
                debug=debug,
            )
        else:
            bidder = LMStudioBidder(
                bidder_id=n_heuristic + i + 1,
                budget=BUDGET,
                debug=debug,
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
        n_opponents=n_heuristic,
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
    verbose: bool = True,
    n_items: int = N_ITEMS,
    n_opponents: int = N_OPPONENTS,
    n_heuristic: int = N_HEURISTIC_OPPONENTS,
    debug: bool = False,
    backend: str = "lmstudio",
) -> AuctionResult:
    auction_seed = derive_seed(eval_seed, eval_idx)
    env = build_eval_environment(auction_seed, n_items, n_opponents, n_heuristic, debug=debug, backend=backend)
    env.reset()
    if verbose:
        tqdm.write(f"  [seed {eval_seed}, auction {eval_idx + 1}] Starting ({len(env.items)} items)...")
    log_payload = await env.run_auction_async(
        return_payload=capture_payload,
        json_path=LOG_PATH,
    )

    wins: Dict[str, int] = {}
    agent_snapshots: Dict[str, AgentSnapshot] = {}
    llm_call_count = 0
    llm_call_durations: List[float] = []

    for agent in env.agents:
        wins[agent.name] = len(agent.items_won)
        agent_snapshots[agent.name] = snapshot_agent(agent)
        bidder = getattr(agent, "bidder", None)
        if isinstance(bidder, (LMStudioBidder, MLXBidder)):
            llm_call_count += bidder.call_count
            llm_call_durations.extend(getattr(bidder, "call_durations", []))

    return AuctionResult(
        eval_seed=eval_seed,
        eval_idx=eval_idx,
        wins=wins,
        item_order=[item.name for item in env.item_order],
        agent_snapshots=agent_snapshots,
        llm_call_count=llm_call_count,
        llm_call_durations=llm_call_durations,
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


async def main(verbose: bool = True, fast: bool = False, debug: bool = False, backend: str = "lmstudio") -> None:
    n_items, n_opponents, _ = _get_config(fast)
    print(f"Loaded model weights from {SAVE_PATH}")
    print(f"LLM: {'MLX (mlx_lm local)' if backend == 'mlx' else 'LM Studio (localhost:1234)'}")
    if fast:
        print(f"  [FAST] {n_items} items, {n_opponents} opponents")
    print(f"\n{'='*60}")
    print(f"{'EVALUATION':^60}")
    print(f"{'='*60}")
    print(f"  Running {N_EVAL} eval auctions × {len(SEEDS)} seeds...")

    sample_identity = (SEEDS[0], 0)
    all_wins: Dict[str, List[float]] = {}
    all_results: List[AuctionResult] = []
    total_llm_calls = 0
    all_llm_durations: List[float] = []

    total_eval = N_EVAL * len(SEEDS)
    t_start = time.perf_counter()
    # MLX shares one GPU model; Metal cannot handle concurrent encode. Run auctions sequentially.
    sequential = backend == "mlx"
    with tqdm(total=total_eval, desc="Evaluation", unit="auction") as pbar:
        for eval_seed in SEEDS:
            if sequential:
                for eval_idx in range(N_EVAL):
                    result = await evaluate_auction(
                        eval_seed=eval_seed,
                        eval_idx=eval_idx,
                        capture_payload=(eval_seed, eval_idx) == sample_identity,
                        verbose=verbose,
                        n_items=n_items,
                        n_opponents=n_opponents,
                        n_heuristic=_get_config(fast)[2],
                        debug=debug,
                        backend=backend,
                    )
                    all_results.append(result)
                    total_llm_calls += result.llm_call_count
                    all_llm_durations.extend(result.llm_call_durations)
                    for name, wins in result.wins.items():
                        all_wins.setdefault(name, []).append(wins)
                    if verbose:
                        wins_str = " | ".join(f"{n}:{w}" for n, w in sorted(result.wins.items(), key=lambda x: -x[1]))
                        llm_info = f", {result.llm_call_count} LLM calls" if result.llm_call_count else ""
                        dur = sum(result.llm_call_durations)
                        dur_str = f", {dur:.1f}s" if dur > 0 else ""
                        pbar.write(f"  [seed {result.eval_seed}, auction {result.eval_idx + 1}] {wins_str}{llm_info}{dur_str}")
                    pbar.update(1)
            else:
                tasks = [
                    asyncio.create_task(
                        evaluate_auction(
                            eval_seed=eval_seed,
                            eval_idx=eval_idx,
                            capture_payload=(eval_seed, eval_idx) == sample_identity,
                            verbose=verbose,
                            n_items=n_items,
                            n_opponents=n_opponents,
                            n_heuristic=_get_config(fast)[2],
                            debug=debug,
                            backend=backend,
                        )
                    )
                    for eval_idx in range(N_EVAL)
                ]
                for task in asyncio.as_completed(tasks):
                    result = await task
                    all_results.append(result)
                    total_llm_calls += result.llm_call_count
                    all_llm_durations.extend(result.llm_call_durations)
                    for name, wins in result.wins.items():
                        all_wins.setdefault(name, []).append(wins)
                    if verbose:
                        wins_str = " | ".join(f"{n}:{w}" for n, w in sorted(result.wins.items(), key=lambda x: -x[1]))
                        llm_info = f", {result.llm_call_count} LLM calls" if result.llm_call_count else ""
                        dur = sum(result.llm_call_durations)
                        dur_str = f", {dur:.1f}s" if dur > 0 else ""
                        pbar.write(f"  [seed {result.eval_seed}, auction {result.eval_idx + 1}] {wins_str}{llm_info}{dur_str}")
                    pbar.update(1)
    t_elapsed = time.perf_counter() - t_start

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

    n_rounds = n_items
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
    print(f"Total LLM calls: {total_llm_calls}")
    if all_llm_durations:
        d = np.array(all_llm_durations)
        d_ms = d * 1000
        print(f"  LLM inference: {d_ms.mean():.0f} ± {d_ms.std():.0f} ms/call (min={d_ms.min():.0f}, max={d_ms.max():.0f})")
        print(f"  Total LLM time: {d.sum():.1f}s  |  Eval wall time: {t_elapsed:.1f}s")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RL model vs LLM (LM Studio or MLX)")
    p.add_argument("--backend", choices=["lmstudio", "mlx"], default="lmstudio", help="lmstudio=LM Studio, mlx=MLXBidder (mlx_lm local)")
    p.add_argument("--fast", action="store_true", help="5 items, 3 opponents")
    p.add_argument("--quiet", action="store_true", help="Disable per-auction logs")
    p.add_argument("--debug", action="store_true", help="Print LLM parse/API errors")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(verbose=not args.quiet, fast=args.fast, debug=args.debug, backend=args.backend))
