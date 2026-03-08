# AuctionAgent

RL agent for multi-agent sequential auctions. The agent learns to bid against heuristic opponents in a turn-based auction where items are presented in random order and agents bid or drop out each round.

---

## How the Auction Works

### Structure

- **Items**: N items (default 20), each with a market value and a priority rank (1 = most wanted). All agents share the same preference order.
- **Agents**: 1 RL agent + 8 heuristic opponents. Each has a budget (default 10,000) and competes for items.
- **Order**: Items are shuffled at the start of each auction; the sequence is random and varies per episode.

### Per-Item Bidding

For each item, agents take turns in round-robin order:

1. **Bid** — Place a bid ≥ minimum valid (one increment above current high bid). Bid must be a multiple of the increment (10% of item value by default).
2. **Drop** — Exit the auction for this item. Dropping is permanent for that item only; the agent can bid on later items.

When only one agent remains, they win and pay the highest bid. If everyone drops, no one wins. The auction moves to the next item.

### Bidding Rules

- **Increment**: Bids must be in steps of `bid_increment_ratio × item.value` (e.g. 0.1 → 10% of value).
- **Min bid**: To beat a bid of X, you must bid at least `X + increment`.
- **Budget**: Cannot bid more than remaining budget. Invalid bids are treated as a drop.

---

## Scoring and Rewards

### Per-Win Score

When an agent wins an item, their score is:

```
score = w_i − β × (p_i / B) − γ × overpay_penalty
```

- **w_i**: Weight from priority rank (linear: `(N − rank + 1) / N`, or exponential).
- **p_i**: Price paid.
- **B**: Starting budget.
- **β**: Cost sensitivity (default 0.5).
- **γ**: Overpay penalty (default 0.25). Penalty applies when paying above market value, capped at 10× overpay.

Higher-priority items have higher w_i. The budget term discourages overspending; the overpay term discourages paying far above market value.

### RL Reward

The RL agent gets this score (normalized) as its step reward when it wins. When it wins via opponent dropout, the reward is still attributed correctly for policy updates.

---

## RL Agent

### State (input to the policy)

Vector of size `2 × n_items + 6`:

| Component | Size | Description |
|-----------|------|-------------|
| `items_done` | n_items | 1 if item already auctioned, 0 otherwise |
| `current_item_onehot` | n_items | One-hot of which item is being auctioned |
| `market_value_norm` | 1 | Current item value / max value |
| `current_bid_norm` | 1 | Highest bid so far / max value |
| `my_val_current_norm` | 1 | Agent’s weight for current item |
| `current_item_rank_norm` | 1 | rank / n_items |
| `items_remaining_norm` | 1 | Items left / n_items |
| `budget_fraction` | 1 | remaining_budget / budget |

### Actions

- **0**: Drop out.
- **1–5**: Bid at 0%, 25%, 50%, 75%, or 100% of the allowed range between min valid bid and a value-aware ceiling.

The ceiling is `min(margin_limit, market_cap, remaining_budget)` where:
- `margin_limit = (w_i × B) / β` (break-even style cap),
- `market_cap = market_value × 50` (avoids extreme overbids).

### Policy

- 3-layer MLP (128 hidden units), REINFORCE with baseline.
- Action mask: invalid bids (e.g. above budget or ceiling) are masked out.
- Bias against dropping to encourage participation.
- ε-greedy exploration during training.

---

## Heuristic Opponents

Eight opponent types from `bidders.py`, each with randomized parameters:

| Bidder | Strategy |
|--------|----------|
| PositiveMarginBidder | Bids uniformly up to break-even |
| MarginPlusSafetyBidder | Bids only when profit exceeds a margin |
| BudgetPacedMarginBidder | Caps by margin and pace (remaining / items left) |
| TopKSpecialistBidder | Bids only on top-K items |
| FlatFractionBidder | Bids fixed fraction of value |
| DescendingAggressionBidder | More aggressive early, less over time |
| SnipeBidder | Skips early items, bids aggressively later |
| RandomBidder | Random bids within budget fraction |

---

## Training and Evaluation

### Training

- **Episodes**: 2500 auctions per run.
- **Checkpointing**: Every 100 episodes, run 50 eval auctions and keep the best model by rounds won.
- **Final model**: Best checkpoint by eval, not the last training step.

### Evaluation

- **Metric**: Rounds won per auction (items won).
- **Eval**: 100 auctions × 3 seeds (300 total).
- **Bulk runs**: `bulk_log.py` runs 10 training runs with different seeds in parallel and writes summaries to `bulk_logs/`.

---

## Project Structure

```
├── env_reward.py    # Auction environment, RL agent, training loop
├── model.py         # AuctionModel (policy network)
├── bidders.py       # Heuristic opponents
├── scoring.py       # Reward/scoring functions
├── run_train.py     # Single train + eval run
├── eval_LLM.py      # Evaluate saved model vs LLM + heuristic opponents
├── bulk_log.py      # Parallel bulk runs
├── visualize.py     # Replay auction from JSON log
├── .env.example     # Environment variable template (copy to .env)
└── env.py           # Legacy/simpler env (unused)
```

---

## Usage

```bash
# Activate environment
source venv/bin/activate

# Set up your API key (required for LLMBidder / eval_LLM.py):
cp .env.example .env
# Then edit .env and replace the placeholder with your Triton AI API key

# Single run (train + eval); saves model weights to auction_model.pt by default
# Set SAVE_MODEL = False in run_train.py to disable
python run_train.py

# Evaluate the saved model against an LLM opponent + heuristic bidders
# Requires a valid API_KEY in .env. Runs 3 auctions and saves auction_log_LLM.json
python eval_LLM.py

# Bulk runs (10 seeds, parallel)
python bulk_log.py

# Replay auction (after run_train saves auction_log.json)
python visualize.py auction_log.json
# Or: python visualize.py auction_log.json --auto --delay 0.8
```

---

## Typical Results

From bulk runs (10 seeds, 20 items, 9 agents):

- RL agent ranks 1st in all runs.
- ~4 rounds/auction (~20% share).
- Loss (last 50) typically negative (policy gradient).
