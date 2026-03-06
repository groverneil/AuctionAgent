from __future__ import annotations
from typing import List


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def rank_to_weight(
    rank: int,
    n_items: int,
    scheme: str = "linear",
) -> float:
    """
    Convert a 1-based priority rank into a weight in [0, 1].

    Parameters
    ----------
    rank    : item's priority rank (1 = most wanted)
    n_items : total number of items in the auction
    scheme  : "linear"      — w = (N - rank + 1) / N
              "exponential"  — w = 2^(N - rank) / 2^(N - 1), decays sharply
    """
    if n_items <= 0:
        raise ValueError("n_items must be positive.")
    if rank <= 0 or rank > n_items:
        return 0.0
    if scheme == "linear":
        return (n_items - rank + 1) / n_items
    elif scheme == "exponential":
        return 2 ** (n_items - rank) / 2 ** (n_items - 1)
    raise ValueError(f"Unknown weight scheme: {scheme!r}")


def ranks_to_weights(
    n_items: int,
    scheme: str = "linear",
) -> List[float]:
    """Return a list of weights for ranks 1 .. n_items."""
    return [rank_to_weight(r, n_items, scheme) for r in range(1, n_items + 1)]


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_basic(value: float, price: float, beta: float = 1.0) -> float:
    """
    score = V_i - β * p_i

    Parameters
    ----------
    value : item value V_i
    price : price paid p_i
    beta  : cost sensitivity weight
    """
    return value - beta * price


def score_normalized(
    value: float,
    price: float,
    v_max: float,
    budget: float,
    beta: float = 1.0,
) -> float:
    """
    score = (V_i / V_max) - β * (p_i / B)

    Parameters
    ----------
    value  : item value V_i
    price  : price paid p_i
    v_max  : maximum value in the episode
    budget : total starting budget B
    beta   : tradeoff weight
    """
    if v_max <= 0 or budget <= 0:
        raise ValueError("v_max and budget must be positive.")
    return (value / v_max) - beta * (price / budget)


def score_priority_weighted(
    rank: int,
    price: float,
    n_items: int,
    budget: float,
    beta: float = 1.0,
    weight_scheme: str = "linear",
    market_value: float = 0.0,
    gamma: float = 0.1,
) -> float:
    """
    score = w_i - β * (p_i / B) - γ * min(max(0, p_i/V_i - 1), 10)

    Cost: budget term + overpay penalty (capped at 10x to avoid explosion).
    """
    if budget <= 0:
        raise ValueError("budget must be positive.")
    w = rank_to_weight(rank, n_items, weight_scheme)
    cost_budget = beta * (price / budget)
    cost_overpay = 0.0
    if market_value > 0:
        overpay_excess = max(0.0, price / market_value - 1.0)
        overpay_excess = min(overpay_excess, 10.0)
        cost_overpay = gamma * overpay_excess
    return w - cost_budget - cost_overpay


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def compute_score(
    value: float,
    price: float,
    mode: str = "basic",
    *,
    beta: float = 1.0,
    v_max: float = 1.0,
    budget: float = 1.0,
    rank: int = 0,
    n_items: int = 1,
    weight_scheme: str = "linear",
    gamma: float = 0.25,
) -> float:
    """
    Unified entry point for scoring a single win.

    Parameters
    ----------
    value          : item market value V_i  (used by basic / normalized)
    price          : price paid p_i
    mode           : "basic", "normalized", or "priority_weighted"
    beta           : cost sensitivity weight
    v_max          : max value in episode          (normalized only)
    budget         : starting budget                (normalized / priority_weighted)
    rank           : item priority rank, 1 = best   (priority_weighted only)
    n_items        : total items in auction          (priority_weighted only)
    weight_scheme  : "linear" or "exponential"       (priority_weighted only)
    gamma         : overpay penalty (priority_weighted only; matches env_reward.OVERPAY_GAMMA)
    """
    if mode == "basic":
        return score_basic(value, price, beta)
    elif mode == "normalized":
        return score_normalized(value, price, v_max, budget, beta)
    elif mode == "priority_weighted":
        return score_priority_weighted(
            rank, price, n_items, budget, beta, weight_scheme,
            market_value=value,
            gamma=gamma,
        )
    raise ValueError(f"Unknown scoring mode: {mode!r}")
