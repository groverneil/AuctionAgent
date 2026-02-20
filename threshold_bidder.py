import random


def threshold_bidder(
    item_index,
    total_items,
    item_value,
    item_start_price,
    current_price,
    player_budget,
    agent_budget,
    stochastic=False,
    threshold_ratio=1.0,
    min_bid=10,
    shrink_rate=1.0,
    stochastic_scale=1.0,
):
    if agent_budget <= current_price:
        return None

    items_left = max(1, total_items - item_index)

    if min_bid < 1:
        min_bid = 1

    if threshold_ratio <= 0:
        threshold_ratio = 1.0
    threshold_price = (agent_budget / items_left) * threshold_ratio

    if threshold_price < 0:
        threshold_price = 0

    hard_cap = agent_budget

    if current_price < threshold_price:
        gap = threshold_price - current_price
        ratio = current_price / max(threshold_price, 1.0)
        if ratio < 0:
            ratio = 0
        if ratio > 1:
            ratio = 1
        if shrink_rate <= 0:
            shrink_rate = 1.0
        scaled = int(gap * ((1.0 - ratio) ** shrink_rate))
        step = max(min_bid, scaled)
        bid = current_price + step
        if bid > threshold_price:
            bid = threshold_price
        if bid > hard_cap:
            return None
        if bid <= current_price:
            return None
        return int(bid)

    distance = current_price - threshold_price
    if stochastic_scale <= 0:
        stochastic_scale = 1.0
    denom = max(float(min_bid), float(stochastic_scale), 1.0)
    quit_chance = 0.1 + (distance / denom) * 0.8
    if quit_chance < 0.1:
        quit_chance = 0.1
    if quit_chance > 0.95:
        quit_chance = 0.95

    if random.random() < quit_chance:
        return None

    bid = current_price + min_bid
    if bid > hard_cap:
        bid = hard_cap
    if bid <= current_price:
        return None

    return int(bid)

