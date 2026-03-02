from typing import Optional, Protocol
import random

from dp_bidder import DPBidder
from incremental_bidder import IncrementalBidder
from threshold_bidder import ThresholdBidder

STARTING_BUDGET = 25
PLAYER_INVOLVED = True
RANDOMIZE_ITEMS = True
RANDOMIZE_AGENT_ORDER = True
NUM_ITEMS = 5
ITEM_VALUES = [50, 80, 20, 30, 10]
DP_BIDDER1_CONFIG = {
    "center_frac": 0.60,
    "scale_frac": 0.18,
    "payment_factor": 0.60,
    "bid_step": 1,
    "enforce_start_price": True,
}
DP_BIDDER2_CONFIG = {
    "center_frac": 0.50,
    "scale_frac": 0.25,
    "payment_factor": 0.65,
    "bid_step": 1,
    "enforce_start_price": True,
}


def build_agent_setups(
    item_values: list[int],
    starting_budget: int,
) -> list[tuple[str, "Bidder"]]:
    return [
        (
            "aggressive_bidder",
            ThresholdBidder(
                threshold_ratio=1.2,
                min_bid=6,
                shrink_rate=0.7,
                stochastic_scale=10.0,
                stochastic=True,
            ),
        ),
        (
            "passive_bidder",
            ThresholdBidder(
                threshold_ratio=0.9,
                min_bid=1,
                shrink_rate=1.0,
                stochastic_scale=6.0,
                stochastic=True,
            ),
        ),
        #("incremental_bidder", IncrementalBidder(increment=5, stochastic=True)),
        (
            "dp_bidder1",
            DPBidder(item_values=item_values, starting_budget=starting_budget, **DP_BIDDER1_CONFIG),
        ),
        (
            "dp_bidder2",
            DPBidder(item_values=item_values, starting_budget=starting_budget, **DP_BIDDER2_CONFIG),
        ),
    ]


class Bidder(Protocol):
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
        ...


def display_name(name: str) -> str:
    if name == "player":
        return "you"
    if name == "none":
        return "none"
    cleaned = name.replace("_threshold_bidder", "_threshold").replace("_bidder", "")
    return cleaned.replace("_", " ")


def format_agent_budgets(agent_budgets: dict[str, int]) -> str:
    return ", ".join(f"{display_name(name)}:{budget}" for name, budget in agent_budgets.items())


def print_item_state(
    item_number: int,
    total_items: int,
    item_value: int,
    item_start_price: int,
    current_price: int,
    current_winner: str,
    player_budget: int,
    agent_budgets: dict[str, int],
    active_bidders: list[str],
) -> None:
    leader = display_name(current_winner)
    print(
        f"\nItem {item_number}/{total_items} | "
        f"value={item_value} | start={item_start_price} | current={current_price} | leader={leader}"
    )
    print(f"Budgets | you:{player_budget} | agents:{format_agent_budgets(agent_budgets)}")


def main() -> None:
    global ITEM_VALUES
    player_budget = STARTING_BUDGET
    if RANDOMIZE_ITEMS:
        ITEM_VALUES = [random.randint(5, 25) for _ in range(NUM_ITEMS)]
    total_items = len(ITEM_VALUES)
    setups_for_run = build_agent_setups(
        item_values=ITEM_VALUES,
        starting_budget=STARTING_BUDGET,
    )
    if RANDOMIZE_AGENT_ORDER:
        random.shuffle(setups_for_run)

    agent_bidders: dict[str, Bidder] = {}
    for name, bidder in setups_for_run:
        agent_bidders[name] = bidder
    agent_budgets: dict[str, int] = {name: STARTING_BUDGET for name in agent_bidders}
    player_won_items: list[int] = []
    agent_won_items: dict[str, list[int]] = {name: [] for name in agent_bidders}

    print("Auction Begins")
    if PLAYER_INVOLVED:
        print("On your turn, enter an integer bid or 'quit' to stop bidding on that item.")
    else:
        print("Running agents-only auction.")

    for idx, item_value in enumerate(ITEM_VALUES, start=1):
        start_price = int(round(item_value * 0.2))
        print(f"\nStarting item {idx} with starting price {start_price}.")
        current_price = start_price
        current_winner = "none"
        item_active = True
        if PLAYER_INVOLVED:
            active_bidders: list[str] = ["player", *agent_budgets.keys()]
        else:
            active_bidders: list[str] = [*agent_budgets.keys()]
        turn_index = 0

        while item_active and active_bidders:
            if turn_index >= len(active_bidders):
                turn_index = 0

            if len(active_bidders) == 1:
                if current_winner == "none":
                    lone_bidder = active_bidders[0]
                    if lone_bidder == "player" and player_budget >= current_price:
                        current_winner = "player"
                    elif lone_bidder in agent_budgets and agent_budgets[lone_bidder] >= current_price:
                        current_winner = lone_bidder
                item_active = False
                continue

            print_item_state(
                item_number=idx,
                total_items=total_items,
                item_value=item_value,
                item_start_price=start_price,
                current_price=current_price,
                current_winner=current_winner,
                player_budget=player_budget,
                agent_budgets=agent_budgets,
                active_bidders=active_bidders,
            )

            active_bidder = active_bidders[turn_index]
            if active_bidder == current_winner and current_winner != "none":
                turn_index += 1
                continue

            if active_bidder == "player":
                min_player_bid = current_price if current_winner == "none" else current_price + 1
                if player_budget < min_player_bid:
                    print(
                        f"you are forced to quit (need >= {min_player_bid}, have {player_budget})"
                    )
                    active_bidders.pop(turn_index)
                    continue

                raw = input(
                    f"Your turn [current bid {current_price}, your budget {player_budget}] "
                    f"(enter >= {min_player_bid} or 'quit'): "
                ).strip().lower()
                if raw == "quit":
                    print("you quit this item")
                    active_bidders.pop(turn_index)
                    continue

                try:
                    bid = int(raw)
                except ValueError:
                    print("invalid input (enter integer bid or 'quit')")
                    continue

                if current_winner == "none":
                    if bid < current_price:
                        print(f"bid too low (must be >= {current_price})")
                        continue
                elif bid <= current_price:
                    print(f"bid too low (must be > {current_price})")
                    continue
                if bid > player_budget:
                    print(f"bid too high (budget is {player_budget})")
                    continue

                current_price = bid
                current_winner = "player"
                print(f"you bid {bid}")
                turn_index += 1
                continue

            # Agent turn
            agent_name = active_bidder
            agent_budget = agent_budgets[agent_name]
            min_agent_bid = current_price if current_winner == "none" else current_price + 1
            if agent_budget < min_agent_bid:
                print(
                    f"{display_name(agent_name)} forced quit (need >= {min_agent_bid}, have {agent_budget})"
                )
                active_bidders.pop(turn_index)
                continue

            bidder = agent_bidders[agent_name]
            try:
                agent_bid = bidder.bid(
                    item_index=idx - 1,
                    total_items=total_items,
                    item_value=item_value,
                    item_start_price=start_price,
                    current_price=current_price,
                    player_budget=player_budget,
                    agent_budget=agent_budget,
                    current_winner=current_winner,
                )
            except TypeError:
                # Backward compatibility for bidder instances created before
                # adding the current_winner argument.
                agent_bid = bidder.bid(
                    item_index=idx - 1,
                    total_items=total_items,
                    item_value=item_value,
                    item_start_price=start_price,
                    current_price=current_price,
                    player_budget=player_budget,
                    agent_budget=agent_budget,
                )
            if agent_bid is None:
                print(f"{display_name(agent_name)} quits")
                active_bidders.pop(turn_index)
                continue
            if current_winner == "none" and agent_bid < current_price:
                print(f"{display_name(agent_name)} quits")
                active_bidders.pop(turn_index)
                continue
            if current_winner != "none" and agent_bid <= current_price:
                print(f"{display_name(agent_name)} quits")
                active_bidders.pop(turn_index)
                continue
            if agent_bid > agent_budget:
                print(f"{display_name(agent_name)} quits")
                active_bidders.pop(turn_index)
                continue

            current_price = agent_bid
            current_winner = agent_name
            print(f"{display_name(agent_name)} bids {agent_bid}")
            turn_index += 1

        # Resolve winner for this item and deduct winner's budget.
        if current_winner == "player":
            player_budget -= current_price
            player_won_items.append(idx)
            print(f"Result: you win item {idx} for {current_price}")
        elif current_winner in agent_budgets:
            agent_budgets[current_winner] -= current_price
            agent_won_items[current_winner].append(idx)
            winner_bidder = agent_bidders[current_winner]
            if hasattr(winner_bidder, "notify_win"):
                winner_bidder.notify_win(idx - 1)
            print(f"Result: {display_name(current_winner)} wins item {idx} for {current_price}")
        else:
            print(f"Result: item {idx} unsold")

        print(
            f"Budgets after item {idx}: "
            f"You={player_budget}, "
            f"Agents={format_agent_budgets(agent_budgets)}"
        )

    print("\nAuction complete.")
    print(f"Final budgets -> You: {player_budget}, Agents: {format_agent_budgets(agent_budgets)}")
    print(f"Items you won: {player_won_items if player_won_items else 'none'}")
    for agent_name in agent_budgets:
        won_items = agent_won_items[agent_name]
        print(f"Items {agent_name} won: {won_items if won_items else 'none'}")


if __name__ == "__main__":
    main()