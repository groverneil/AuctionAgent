from __future__ import annotations


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


def compute_score(
    value: float,
    price: float,
    mode: str = "basic",
    *,
    beta: float = 1.0,
    v_max: float = 1.0,
    budget: float = 1.0,
) -> float:
    """
    Unified entry point for scoring a single win.

    Parameters
    ----------
    value  : item value V_i
    price  : price paid p_i
    mode   : "basic" or "normalized"
    beta   : cost sensitivity weight
    v_max  : max value in episode (required for normalized)
    budget : starting budget (required for normalized)
    """
    if mode == "basic":
        return score_basic(value, price, beta)
    elif mode == "normalized":
        return score_normalized(value, price, v_max, budget, beta)
    raise ValueError(f"Unknown scoring mode: {mode!r}")
