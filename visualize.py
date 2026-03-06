import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Usage examples:
# - Step mode (default): python visualize.py auction_log.json
# - Auto-play:          python visualize.py auction_log.json --auto --delay 0.8
# - Plain text mode:    python visualize.py auction_log.json --no-color --auto --delay 0

try:
    from rich.columns import Columns
    from rich.console import Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    Columns = Group = Layout = Live = Panel = Table = Text = None


PALETTE = [
    "cyan",
    "magenta",
    "green",
    "yellow",
    "blue",
    "red",
    "bright_cyan",
    "bright_magenta",
    "bright_green",
    "bright_yellow",
    "bright_blue",
    "bright_red",
]


@dataclass
class ReplayState:
    agents: List[str]
    item_order: List[str]
    initial_budgets: Dict[str, Optional[float]]
    round_num: int = 0
    current_item: str = "-"
    highest_bid: float = 0.0
    highest_bidder: Optional[str] = None
    current_bids: Dict[str, Optional[float]] = field(default_factory=dict)
    spent: Dict[str, float] = field(default_factory=dict)
    dropped: set = field(default_factory=set)
    last_action: str = "Waiting for first event..."
    last_actor: Optional[str] = None
    last_winner: Optional[str] = None

    def __post_init__(self) -> None:
        self.current_bids = {agent: None for agent in self.agents}
        self.spent = {agent: 0.0 for agent in self.agents}

    def reset_round(self, round_num: int, item_name: str) -> None:
        self.round_num = round_num
        self.current_item = item_name
        self.highest_bid = 0.0
        self.highest_bidder = None
        self.current_bids = {agent: None for agent in self.agents}
        self.dropped = set()
        self.last_winner = None

    def remaining_budget(self, agent: str) -> Optional[float]:
        init = self.initial_budgets.get(agent)
        if init is None:
            return None
        return init - self.spent.get(agent, 0.0)


def _format_event_line(event: Dict[str, Any]) -> str:
    event_type = event.get("type")
    if event_type == "bid":
        return f"{event.get('agent', 'unknown')} bids {float(event.get('amount', 0.0)):.2f}"
    if event_type == "dropout":
        return f"{event.get('agent', 'unknown')} drops out"
    if event_type == "win":
        return (
            f"{event.get('agent', 'unknown')} wins "
            f"{event.get('item', 'unknown_item')} for {float(event.get('price', 0.0)):.2f}"
        )
    return f"Unknown event: {event}"


def _build_colors(agents: List[str]) -> Dict[str, str]:
    return {agent: PALETTE[i % len(PALETTE)] for i, agent in enumerate(agents)}


def _item_ribbon(state: ReplayState) -> Panel:
    parts = []
    for idx, item in enumerate(state.item_order, start=1):
        if idx < state.round_num:
            style = "dim"
        elif idx == state.round_num:
            style = "bold white on dark_green"
        else:
            style = "grey50"
        parts.append(f"[{style}] {item} [/{style}]")
    ribbon = " ".join(parts) if parts else "[grey50]No items[/grey50]"
    return Panel(ribbon, title="Items", border_style="white")


def _bidder_cards(state: ReplayState, colors: Dict[str, str]) -> Panel:
    cards = []
    for agent in state.agents:
        is_dropped = agent in state.dropped
        base_color = colors.get(agent, "white")
        card_color = "grey50" if is_dropped else base_color
        bid_val = state.current_bids.get(agent)
        bid_text = "--" if bid_val is None else f"{bid_val:.2f}"
        spent = state.spent.get(agent, 0.0)
        remaining = state.remaining_budget(agent)
        remaining_text = "?" if remaining is None else f"{remaining:.2f}"

        if is_dropped:
            status = "DROPPED"
        elif state.last_winner == agent:
            status = "WINNER"
        else:
            status = "ACTIVE"

        body = Text()
        body.append(f"{agent}\n", style=f"bold {card_color}")
        body.append(f"Bid: {bid_text}\n", style=card_color)
        body.append(f"Spent: {spent:.2f}\n", style=card_color)
        body.append(f"Budget: {remaining_text}\n", style=card_color)
        body.append(f"Status: {status}", style=("dim" if is_dropped else card_color))
        cards.append(Panel(body, border_style=card_color, expand=True))

    return Panel(Columns(cards, expand=True), title="Bidders", border_style="white")


def _stats_panel(state: ReplayState) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_row("Round", str(state.round_num))
    table.add_row("Current item", state.current_item)
    table.add_row("Highest bid", f"{state.highest_bid:.2f}")
    table.add_row("Leader", state.highest_bidder or "-")
    active_count = len(state.agents) - len(state.dropped)
    table.add_row("Active bidders", str(active_count))
    return Panel(table, title="Turn Stats", border_style="white")


def _action_panel(state: ReplayState, colors: Dict[str, str]) -> Panel:
    actor = state.last_actor
    if actor is None:
        return Panel(state.last_action, title="Action", border_style="white")
    color = "grey50" if actor in state.dropped else colors.get(actor, "white")
    action = Text(state.last_action, style=f"bold {color}")
    return Panel(action, title="Action", border_style=color)


def _render_layout(state: ReplayState, colors: Dict[str, str]) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="items", size=3),
        Layout(name="body", ratio=1),
        Layout(name="action", size=5),
    )
    layout["body"].split_row(
        Layout(name="bidders", ratio=3),
        Layout(name="stats", ratio=1),
    )

    layout["items"].update(_item_ribbon(state))
    layout["bidders"].update(_bidder_cards(state, colors))
    layout["stats"].update(_stats_panel(state))
    layout["action"].update(_action_panel(state, colors))
    return layout


def replay_auction(
    json_path: str,
    delay: float = 0.8,
    no_color: bool = False,
    step_mode: bool = True,
) -> None:
    with open(json_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    events = payload.get("events", [])
    if not isinstance(events, list):
        raise ValueError("Invalid log format: expected top-level 'events' list.")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    item_order = metadata.get("item_order", [])
    if not isinstance(item_order, list):
        item_order = []
    agents = metadata.get("agents", [])
    if not isinstance(agents, list):
        agents = []
    if not agents:
        inferred = []
        for event in events:
            if isinstance(event, dict):
                agent = event.get("agent")
                if isinstance(agent, str) and agent not in inferred:
                    inferred.append(agent)
        agents = inferred

    initial_budgets: Dict[str, Optional[float]] = {}
    budget_map = metadata.get("budgets")
    if isinstance(budget_map, dict):
        for agent in agents:
            value = budget_map.get(agent)
            if isinstance(value, (int, float)):
                initial_budgets[agent] = float(value)
            else:
                initial_budgets[agent] = None
    else:
        scalar_budget: Optional[float] = None
        meta_budget = metadata.get("starting_budget")
        if isinstance(meta_budget, (int, float)):
            scalar_budget = float(meta_budget)
        else:
            raise ValueError(
                "Invalid log format: expected metadata.budgets "
                "(per-agent) or metadata.starting_budget."
            )
        for agent in agents:
            initial_budgets[agent] = scalar_budget

    if not no_color and Live is None:
        raise ImportError(
            "Rich is required for color dashboard mode. Install with: pip install rich "
            "or run with --no-color."
        )

    if no_color:
        for event in events:
            if isinstance(event, dict):
                print(_format_event_line(event))
                if step_mode:
                    input("Press Enter for next event...")
                elif delay > 0:
                    time.sleep(delay)
        return

    state = ReplayState(
        agents=agents,
        item_order=item_order,
        initial_budgets=initial_budgets,
    )
    colors = _build_colors(agents)

    with Live(_render_layout(state, colors), refresh_per_second=30, screen=True) as live:
        for event in events:
            if not isinstance(event, dict):
                continue

            event_round = int(event.get("round", 0) or 0)
            event_item = str(event.get("item", state.current_item))
            if event_round != state.round_num:
                state.reset_round(event_round, event_item)
            else:
                state.current_item = event_item

            event_type = event.get("type")
            agent = str(event.get("agent", "unknown"))
            state.last_actor = agent

            if event_type == "bid":
                amount = float(event.get("amount", 0.0) or 0.0)
                state.current_bids[agent] = amount
                if amount >= state.highest_bid:
                    state.highest_bid = amount
                    state.highest_bidder = agent
                state.last_action = _format_event_line(event)
            elif event_type == "dropout":
                state.dropped.add(agent)
                state.last_action = _format_event_line(event)
            elif event_type == "win":
                price = float(event.get("price", 0.0) or 0.0)
                state.highest_bid = max(state.highest_bid, price)
                state.highest_bidder = agent
                state.last_winner = agent
                state.spent[agent] = state.spent.get(agent, 0.0) + price
                state.last_action = _format_event_line(event)
            else:
                state.last_action = _format_event_line(event)

            live.update(_render_layout(state, colors))
            if step_mode:
                input("Press Enter for next event...")
            elif delay > 0:
                time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay auction events from JSON log.")
    parser.add_argument("json_path", nargs="?", default="auction_log.json")
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--auto", action="store_true", help="Auto-play events (disable Enter-to-step).")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    replay_auction(
        args.json_path,
        delay=max(0.0, args.delay),
        no_color=args.no_color,
        step_mode=not args.auto,
    )


if __name__ == "__main__":
    main()
