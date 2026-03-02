import math
from typing import Callable, Optional


class DPBidder:
    def __init__(
        self,
        item_values: list[int],
        starting_budget: int,
        win_prob_model: Optional[Callable[[int, int], float]] = None,
        enforce_start_price: bool = True,
        center_frac: float = 0.45,
        scale_frac: float = 0.15,
        payment_factor: float = 0.75,
        bid_step: int = 1,
    ):
        self.item_values = item_values
        self.N = len(item_values)
        self.B0 = max(0, int(starting_budget))
        self.enforce_start_price = enforce_start_price
        self.center_frac = max(0.0, float(center_frac))
        self.scale_frac = max(0.01, float(scale_frac))
        self.payment_factor = min(max(float(payment_factor), 0.05), 1.0)
        self.bid_step = max(1, int(bid_step))
        self.items_won_mask = 0

        if win_prob_model is None:
            self.win_prob_model = self._default_win_prob
        else:
            self.win_prob_model = win_prob_model

        self.best_bid = self._compute_policy()

    def _default_win_prob(self, item_index: int, ceiling: int) -> float:
        if ceiling <= 0:
            return 0.0
        value = float(self.item_values[item_index])
        center = max(1.0, value * self.center_frac)
        scale = max(1.0, value * self.scale_frac)
        x = (float(ceiling) - center) / scale
        p = 1.0 / (1.0 + math.exp(-x))
        if p < 0.0:
            return 0.0
        if p > 1.0:
            return 1.0
        return p

    def _effective_payment(self, ceiling: int) -> int:
        if ceiling <= 0:
            return 0
        paid = int(round(ceiling * self.payment_factor))
        if paid < 1:
            return 1
        return paid

    def _compute_policy(self) -> list[list[list[int]]]:
        num_masks = 1 << self.N
        mask_values = [0] * num_masks
        for mask in range(num_masks):
            total = 0
            for k in range(self.N):
                if (mask >> k) & 1:
                    total += self.item_values[k]
            mask_values[mask] = total

        dp = [
            [[0.0 for _ in range(self.B0 + 1)] for _ in range(num_masks)]
            for _ in range(self.N + 1)
        ]
        best_bid = [
            [[0 for _ in range(self.B0 + 1)] for _ in range(num_masks)]
            for _ in range(self.N + 1)
        ]

        for mask in range(num_masks):
            for budget in range(self.B0 + 1):
                dp[self.N][mask][budget] = float(mask_values[mask] + budget)
                best_bid[self.N][mask][budget] = 0

        for item_index in range(self.N - 1, -1, -1):
            start_price = int(math.floor(0.2 * self.item_values[item_index]))
            for mask in range(num_masks):
                for budget in range(self.B0 + 1):
                    best_value = dp[item_index + 1][mask][budget]
                    best_z = 0

                    for z in range(0, budget + 1):
                        if self.enforce_start_price and z != 0 and z < start_price:
                            continue

                        p = self.win_prob_model(item_index, z)
                        if p <= 0.0:
                            expected = dp[item_index + 1][mask][budget]
                        else:
                            p = min(max(p, 0.0), 1.0)
                            mask_win = mask | (1 << item_index)
                            expected_payment = self._effective_payment(z)
                            budget_after_win = budget - expected_payment
                            if budget_after_win < 0:
                                budget_after_win = 0
                            expected = (
                                p * dp[item_index + 1][mask_win][budget_after_win]
                                + (1.0 - p) * dp[item_index + 1][mask][budget]
                            )

                        if expected > best_value:
                            best_value = expected
                            best_z = z

                    dp[item_index][mask][budget] = best_value
                    best_bid[item_index][mask][budget] = best_z

        return best_bid

    def bid(
        self,
        item_index: int,
        total_items: int,
        item_value: int,
        item_start_price: int,
        current_price: int,
        player_budget: int,
        agent_budget: int,
        current_winner: str,
    ) -> Optional[int]:
        _ = (total_items, item_value, item_start_price, player_budget)
        if item_index < 0 or item_index >= self.N:
            return None
        if agent_budget <= 0:
            return None

        bounded_budget = max(0, min(int(agent_budget), self.B0))
        ceiling = self.best_bid[item_index][self.items_won_mask][bounded_budget]

        if current_winner == "none":
            min_required = current_price
        else:
            min_required = current_price + 1

        if ceiling < min_required:
            return None
        if min_required > agent_budget:
            return None
        target_bid = min_required + self.bid_step - 1
        bid = min(int(ceiling), int(target_bid))
        if bid < min_required:
            return None
        if bid > agent_budget:
            return None
        return int(bid)

    def notify_win(self, item_index: int) -> None:
        if 0 <= item_index < self.N:
            self.items_won_mask |= 1 << item_index
