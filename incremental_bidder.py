class IncrementalBidder:
    def __init__(self, increment=10, stochastic=False):
        self.increment = increment
        self.stochastic = stochastic

    def bid(
        self,
        item_index,
        total_items,
        item_value,
        item_start_price,
        current_price,
        player_budget,
        agent_budget,
        current_winner,
    ):
        _ = (
            item_index,
            total_items,
            item_value,
            item_start_price,
            player_budget,
            self.stochastic,
            current_winner,
        )
        increment = self.increment
        if increment < 1:
            increment = 1
        next_bid = current_price + increment
        if next_bid <= agent_budget:
            return next_bid
        return None

