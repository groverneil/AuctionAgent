def incremental_bidder(
    item_index,
    total_items,
    item_value,
    item_start_price,
    current_price,
    player_budget,
    agent_budget,
    stochastic=False,
    increment=10,
):
    _ = (item_index, total_items, item_value, item_start_price, player_budget, stochastic)
    if increment < 1:
        increment = 1
    next_bid = current_price + increment
    if next_bid <= agent_budget:
        return next_bid
    return None

