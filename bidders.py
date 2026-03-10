from __future__ import annotations
import asyncio
import importlib
import os
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, TYPE_CHECKING

from scoring import rank_to_weight

if TYPE_CHECKING:
    from env_reward import Item


@dataclass
class BidderState:
    """Mutable per-episode state tracked for each bidder."""

    budget: float
    remaining_budget: float
    n_items: int = 0
    spent: float = 0.0
    items_won: List[Item] = field(default_factory=list)
    prices_paid: List[float] = field(default_factory=list)
    total_score: float = 0.0

    @property
    def n_won(self) -> int:
        return len(self.items_won)


class BaseBidder(ABC):
    """Abstract base for every bidder (heuristic, RL, LLM, etc.)."""

    def __init__(self, bidder_id: int, budget: float):
        self.bidder_id = bidder_id
        self.budget = budget
        self._rng = random.Random()

    def _priority_value(
        self,
        item: Item,
        state: BidderState,
        weight_scheme: str = "linear",
    ) -> float:
        w = rank_to_weight(item.rank, state.n_items, weight_scheme)
        return w * state.budget

    @abstractmethod
    def place_bid(
        self,
        item: Item,
        state: BidderState,
        items_remaining: int,
        weight_scheme: str = "linear",
    ) -> float:
        ...

    async def place_bid_async(
        self,
        item: Item,
        state: BidderState,
        items_remaining: int,
        weight_scheme: str = "linear",
    ) -> float:
        return self.place_bid(
            item,
            state,
            items_remaining=items_remaining,
            weight_scheme=weight_scheme,
        )

    def new_state(self, n_items: int = 0) -> BidderState:
        return BidderState(
            budget=self.budget, remaining_budget=self.budget, n_items=n_items,
        )

    def set_seed(self, seed: Optional[int]) -> None:
        self._rng.seed(seed)

    def _uniform(self, low: float, high: float) -> float:
        return self._rng.uniform(low, high)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.bidder_id}, budget={self.budget:.1f})"


class PositiveMarginBidder(BaseBidder):
    def __init__(self, bidder_id, budget, beta=1.0, min_bid=0.01):
        super().__init__(bidder_id, budget)
        self.beta = beta
        self.min_bid = min_bid

    def place_bid(self, item, state, items_remaining, weight_scheme="linear"):
        if state.remaining_budget <= 0:
            return 0.0
        value = self._priority_value(item, state, weight_scheme)
        max_bid = min(value / self.beta, state.remaining_budget)
        if max_bid < self.min_bid:
            return 0.0
        return self._uniform(self.min_bid, max_bid)


class MarginPlusSafetyBidder(BaseBidder):
    def __init__(self, bidder_id, budget, beta=1.0, margin=1.0):
        super().__init__(bidder_id, budget)
        self.beta = beta
        self.margin = margin

    def place_bid(self, item, state, items_remaining, weight_scheme="linear"):
        if state.remaining_budget <= 0:
            return 0.0
        value = self._priority_value(item, state, weight_scheme)
        max_bid = min((value - self.margin) / self.beta, state.remaining_budget)
        if max_bid <= 0:
            return 0.0
        return self._uniform(max_bid * 0.5, max_bid)


class BudgetPacedMarginBidder(BaseBidder):
    def __init__(self, bidder_id, budget, beta=1.0, c=1.2, top_k=3):
        super().__init__(bidder_id, budget)
        self.beta = beta
        self.c = c
        self.top_k = top_k

    def place_bid(self, item, state, items_remaining, weight_scheme="linear"):
        if state.remaining_budget <= 0 or items_remaining <= 0:
            return 0.0
        value = self._priority_value(item, state, weight_scheme)
        margin_limit = value / self.beta
        pace_limit = self.c * (state.remaining_budget / items_remaining)
        is_top_k = item.rank <= self.top_k
        if is_top_k:
            max_bid = min(margin_limit, state.remaining_budget)
        else:
            max_bid = min(margin_limit, pace_limit, state.remaining_budget)
        if max_bid <= 0:
            return 0.0
        low = max_bid * (0.7 if is_top_k else 0.4)
        return self._uniform(low, max_bid)


class TopKSpecialistBidder(BaseBidder):
    def __init__(self, bidder_id, budget, beta=1.0, top_k=3, margin=0.0):
        super().__init__(bidder_id, budget)
        self.beta = beta
        self.top_k = top_k
        self.margin = margin

    def place_bid(self, item, state, items_remaining, weight_scheme="linear"):
        if state.remaining_budget <= 0:
            return 0.0
        if item.rank > self.top_k:
            return 0.0
        value = self._priority_value(item, state, weight_scheme)
        max_bid = min((value - self.margin) / self.beta, state.remaining_budget)
        if max_bid <= 0:
            return 0.0
        return self._uniform(max_bid * 0.75, max_bid)


class FlatFractionBidder(BaseBidder):
    def __init__(self, bidder_id, budget, f=0.8):
        super().__init__(bidder_id, budget)
        self.f = f

    def place_bid(self, item, state, items_remaining, weight_scheme="linear"):
        if state.remaining_budget <= 0:
            return 0.0
        value = self._priority_value(item, state, weight_scheme)
        return min(self.f * value, state.remaining_budget)


class DescendingAggressionBidder(BaseBidder):
    def __init__(self, bidder_id, budget, beta=1.0, f_start=0.95, f_end=0.2):
        super().__init__(bidder_id, budget)
        self.beta = beta
        self.f_start = f_start
        self.f_end = f_end

    def place_bid(self, item, state, items_remaining, weight_scheme="linear"):
        if state.remaining_budget <= 0:
            return 0.0
        spend_ratio = state.spent / state.budget if state.budget > 0 else 0.0
        aggression = self.f_start - (self.f_start - self.f_end) * spend_ratio
        value = self._priority_value(item, state, weight_scheme)
        max_bid = min(aggression * value / self.beta, state.remaining_budget)
        if max_bid <= 0:
            return 0.0
        return max_bid * self._uniform(0.9, 1.0)


class SnipeBidder(BaseBidder):
    def __init__(self, bidder_id, budget, beta=1.0, snipe_from_rank=6, aggression=1.5):
        super().__init__(bidder_id, budget)
        self.beta = beta
        self.snipe_from_rank = snipe_from_rank
        self.aggression = aggression

    def place_bid(self, item, state, items_remaining, weight_scheme="linear"):
        if state.remaining_budget <= 0 or items_remaining <= 0:
            return 0.0
        if item.rank < self.snipe_from_rank:
            return 0.0
        value = self._priority_value(item, state, weight_scheme)
        margin_limit = value / self.beta
        snipe_limit = self.aggression * (state.remaining_budget / items_remaining)
        max_bid = min(margin_limit, snipe_limit, state.remaining_budget)
        if max_bid <= 0:
            return 0.0
        return self._uniform(max_bid * 0.8, max_bid)


class RandomBidder(BaseBidder):
    def __init__(self, bidder_id, budget, max_fraction=0.5):
        super().__init__(bidder_id, budget)
        self.max_fraction = max_fraction

    def place_bid(self, item, state, items_remaining, weight_scheme="linear"):
        if state.remaining_budget <= 0:
            return 0.0
        return self._uniform(0.0, self.max_fraction * state.remaining_budget)


class LLMBidder(BaseBidder):
    """LLM-driven bidder using Triton AI API. Binary 0/1 decision; if 1,
    bids min_required (same increment logic as RL agent)."""

    BASE_URL = "https://tritonai-api.ucsd.edu"
    MODEL = "api-gpt-oss-120b"
    BID_INCREMENT_RATIO = 0.1

    def __init__(self, bidder_id, budget, beta=1.0, temperature=0.1, debug=False):
        super().__init__(bidder_id, budget)
        self.beta = beta
        self.temperature = temperature
        self.debug = debug
        self._client = None
        self._async_client = None
        self.call_count = 0

    @staticmethod
    def _load_api_key() -> str:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        api_key = os.getenv("API_KEY", "")
        if not api_key:
            raise ValueError("Missing API_KEY environment variable.")
        return api_key

    def _get_client(self):
        if self._client is None:
            openai_mod = importlib.import_module("openai")
            self._client = getattr(openai_mod, "OpenAI")(
                api_key=self._load_api_key(), base_url=self.BASE_URL
            )
        return self._client

    def _get_async_client(self):
        if self._async_client is None:
            openai_mod = importlib.import_module("openai")
            AsyncOpenAI = getattr(openai_mod, "AsyncOpenAI", None)
            if AsyncOpenAI is None:
                raise AttributeError("AsyncOpenAI not available.")
            self._async_client = AsyncOpenAI(
                api_key=self._load_api_key(), base_url=self.BASE_URL
            )
        return self._async_client

    @staticmethod
    def _parse_binary(text: str) -> Optional[int]:
        """Parse model response as 0 or 1."""
        text = text.strip()
        if text in ("0", "1"):
            return int(text)
        match = re.search(r"[01]", text)
        return int(match.group(0)) if match else None

    def _build_prompt(self, item, state, items_remaining, high_bid, min_required):
        n = max(state.n_items, 1)
        budget_frac = state.remaining_budget / state.budget if state.budget else 0.0
        lines = [
            f"You are bidding in a sequential auction of {n} items against {n - 1} other bidders.",
            "Your goal is to win as many items as possible while managing your budget.",
            "",
            f"Current item: highest bid so far is {high_bid:.2f}.",
            f"If you bid, your bid will be {min_required:.2f} (the minimum increment).",
            f"Your remaining budget: {state.remaining_budget:.2f} (fraction: {budget_frac:.3f})",
            f"Items remaining: {items_remaining} of {n}",
            "",
            "Output 1 to bid (your bid = the minimum increment).",
            "Output 0 to drop out of this item.",
            "Output only 0 or 1:",
        ]
        return "\n".join(lines)

    def _request_messages(self, prompt):
        return [
            {"role": "system", "content": "Output only 0 or 1."},
            {"role": "user", "content": prompt},
        ]

    def _call_llm_once(self, prompt):
        resp = self._get_client().chat.completions.create(
            model=self.MODEL,
            temperature=self.temperature,
            messages=self._request_messages(prompt),
        )
        raw = resp.choices[0].message.content if resp.choices else ""
        return self._parse_binary(raw or "")

    def _call_llm(self, prompt):
        self.call_count += 1
        try:
            return self._call_llm_once(prompt)
        except Exception as exc:
            if self.debug:
                print(f"[LLMBidder {self.bidder_id}] API error: {exc!r}")
            return None

    async def _call_llm_async(self, prompt):
        self.call_count += 1
        try:
            resp = await self._get_async_client().chat.completions.create(
                model=self.MODEL,
                temperature=self.temperature,
                messages=self._request_messages(prompt),
            )
            raw = resp.choices[0].message.content if resp.choices else ""
            return self._parse_binary(raw or "")
        except (ImportError, AttributeError, ModuleNotFoundError):
            try:
                return await asyncio.to_thread(self._call_llm_once, prompt)
            except Exception as exc:
                if self.debug:
                    print(f"[LLMBidder {self.bidder_id}] API error: {exc!r}")
                return None
        except Exception as exc:
            if self.debug:
                print(f"[LLMBidder {self.bidder_id}] API error: {exc!r}")
            return None

    def _build_bid_request(self, item, state, items_remaining):
        bid_history = [float(x) for x in item.bids]
        high_bid = max(bid_history) if bid_history else 0.0
        min_required = high_bid + self.BID_INCREMENT_RATIO * float(item.value)
        return self._build_prompt(item, state, items_remaining, high_bid, min_required), min_required

    def _decision_to_bid(self, decision, state, min_required):
        if decision is None or decision == 0:
            return 0.0
        bid = min_required
        if bid > state.remaining_budget:
            bid = state.remaining_budget
        return float(bid)

    def place_bid(self, item, state, items_remaining, weight_scheme="linear"):
        if state.remaining_budget <= 0:
            return 0.0
        prompt, min_required = self._build_bid_request(item, state, items_remaining)
        decision = self._call_llm(prompt)
        if self.debug:
            print(f"[LLMBidder {self.bidder_id}] item={item.name} decision={decision} min_req={min_required:.2f} budget={state.remaining_budget:.2f}")
        return self._decision_to_bid(decision, state, min_required)

    async def place_bid_async(self, item, state, items_remaining, weight_scheme="linear"):
        del weight_scheme
        if state.remaining_budget <= 0:
            return 0.0
        prompt, min_required = self._build_bid_request(item, state, items_remaining)
        decision = await self._call_llm_async(prompt)
        if self.debug:
            print(f"[LLMBidder {self.bidder_id}] item={item.name} decision={decision} min_req={min_required:.2f} budget={state.remaining_budget:.2f}")
        return self._decision_to_bid(decision, state, min_required)


class LMStudioBidder(BaseBidder):
    """LLM-driven bidder for LM Studio (localhost:1234). Binary 0/1 decision;
    if 1, bids min_required (same increment logic as RL agent)."""

    BASE_URL = "http://localhost:1234/v1"
    MODEL = "local"
    API_KEY = "lm-studio"
    BID_INCREMENT_RATIO = 0.1

    def __init__(self, bidder_id, budget, beta=1.0, temperature=0.1, debug=False):
        super().__init__(bidder_id, budget)
        self.beta = beta
        self.temperature = temperature
        self.debug = debug
        self._client = None
        self._async_client = None
        self.call_count = 0
        self.call_durations: List[float] = []

    def _get_client(self):
        if self._client is None:
            openai_mod = importlib.import_module("openai")
            self._client = getattr(openai_mod, "OpenAI")(
                api_key=self.API_KEY, base_url=self.BASE_URL
            )
        return self._client

    def _get_async_client(self):
        if self._async_client is None:
            openai_mod = importlib.import_module("openai")
            AsyncOpenAI = getattr(openai_mod, "AsyncOpenAI", None)
            if AsyncOpenAI is None:
                raise AttributeError("AsyncOpenAI not available.")
            self._async_client = AsyncOpenAI(api_key=self.API_KEY, base_url=self.BASE_URL)
        return self._async_client

    @staticmethod
    def _parse_binary(text: str) -> Optional[int]:
        text = text.strip()
        if text in ("0", "1"):
            return int(text)
        match = re.search(r"[01]", text)
        return int(match.group(0)) if match else None

    def _build_prompt(self, item, state, items_remaining, high_bid, min_required):
        n = max(state.n_items, 1)
        budget_frac = state.remaining_budget / state.budget if state.budget else 0.0
        lines = [
            f"You are bidding in a sequential auction of {n} items against {n - 1} other bidders.",
            "Your goal is to win as many items as possible while managing your budget.",
            "",
            f"Current item: highest bid so far is {high_bid:.2f}.",
            f"If you bid, your bid will be {min_required:.2f} (the minimum increment).",
            f"Your remaining budget: {state.remaining_budget:.2f} (fraction: {budget_frac:.3f})",
            f"Items remaining: {items_remaining} of {n}",
            "",
            "Output 1 to bid (your bid = the minimum increment).",
            "Output 0 to drop out of this item.",
            "Output only 0 or 1:",
        ]
        return "\n".join(lines)

    def _request_messages(self, prompt):
        return [
            {"role": "system", "content": "Output only 0 or 1."},
            {"role": "user", "content": prompt},
        ]

    def _call_llm_once(self, prompt):
        t0 = time.perf_counter()
        try:
            resp = self._get_client().chat.completions.create(
                model=self.MODEL,
                temperature=self.temperature,
                messages=self._request_messages(prompt),
            )
            raw = resp.choices[0].message.content if resp.choices else ""
            out = self._parse_binary(raw or "")
            self.call_durations.append(time.perf_counter() - t0)
            return out
        except Exception:
            self.call_durations.append(time.perf_counter() - t0)
            raise

    def _call_llm(self, prompt):
        self.call_count += 1
        try:
            return self._call_llm_once(prompt)
        except Exception as exc:
            if self.debug:
                print(f"[LMStudioBidder {self.bidder_id}] API error: {exc!r}")
            return None

    async def _call_llm_async(self, prompt):
        self.call_count += 1
        t0 = time.perf_counter()
        try:
            resp = await self._get_async_client().chat.completions.create(
                model=self.MODEL,
                temperature=self.temperature,
                messages=self._request_messages(prompt),
            )
            raw = resp.choices[0].message.content if resp.choices else ""
            out = self._parse_binary(raw or "")
            self.call_durations.append(time.perf_counter() - t0)
            return out
        except (ImportError, AttributeError, ModuleNotFoundError):
            try:
                return await asyncio.to_thread(self._call_llm_once, prompt)
            except Exception as exc:
                if self.debug:
                    print(f"[LMStudioBidder {self.bidder_id}] API error: {exc!r}")
                return None
        except Exception as exc:
            if self.debug:
                print(f"[LMStudioBidder {self.bidder_id}] API error: {exc!r}")
            return None

    def _build_bid_request(self, item, state, items_remaining):
        bid_history = [float(x) for x in item.bids]
        high_bid = max(bid_history) if bid_history else 0.0
        min_required = high_bid + self.BID_INCREMENT_RATIO * float(item.value)
        return self._build_prompt(item, state, items_remaining, high_bid, min_required), min_required

    def _decision_to_bid(self, decision, state, min_required):
        if decision is None or decision == 0:
            return 0.0
        bid = min_required
        if bid > state.remaining_budget:
            bid = state.remaining_budget
        return float(bid)

    def place_bid(self, item, state, items_remaining, weight_scheme="linear"):
        if state.remaining_budget <= 0:
            return 0.0
        prompt, min_required = self._build_bid_request(item, state, items_remaining)
        decision = self._call_llm(prompt)
        if self.debug:
            print(f"[LMStudioBidder {self.bidder_id}] item={item.name} decision={decision} min_req={min_required:.2f} budget={state.remaining_budget:.2f}")
        return self._decision_to_bid(decision, state, min_required)

    async def place_bid_async(self, item, state, items_remaining, weight_scheme="linear"):
        del weight_scheme
        if state.remaining_budget <= 0:
            return 0.0
        prompt, min_required = self._build_bid_request(item, state, items_remaining)
        decision = await self._call_llm_async(prompt)
        if self.debug:
            print(f"[LMStudioBidder {self.bidder_id}] item={item.name} decision={decision} min_req={min_required:.2f} budget={state.remaining_budget:.2f}")
        return self._decision_to_bid(decision, state, min_required)


class MLXBidder(BaseBidder):
    """MLX-native bidder for gpt-oss-20b on Apple Silicon — no HTTP round-trip.
    Binary 0/1 decision; if 1, bids min_required (same as RL agent).
    Model loaded once at class level, shared across instances.

    Raw model output: <|channel|>analysis<|message|>...<|channel|>final<|message|>0 or 1
    """

    MODEL_REPO = "mlx-community/gpt-oss-20b-MXFP4-Q4"
    BID_INCREMENT_RATIO = 0.1

    _model = None
    _tokenizer = None

    def __init__(self, bidder_id, budget, beta=1.0, reasoning_effort="low", debug=False):
        super().__init__(bidder_id, budget)
        self.beta = beta
        self.reasoning_effort = reasoning_effort
        self.debug = debug
        self.call_count = 0
        self.call_durations: List[float] = []
        MLXBidder._ensure_model_loaded()

    @classmethod
    def _ensure_model_loaded(cls) -> None:
        if cls._model is None:
            from mlx_lm import load
            print(f"[MLXBidder] Loading {cls.MODEL_REPO} ...")
            cls._model, cls._tokenizer = load(cls.MODEL_REPO)
            print("[MLXBidder] Model loaded.")

    @staticmethod
    def _parse_binary(text: str) -> Optional[int]:
        text = text.strip()
        if text in ("0", "1"):
            return int(text)
        match = re.search(r"[01]", text)
        return int(match.group(0)) if match else None

    @staticmethod
    def _extract_final_channel(text: str) -> Optional[str]:
        """Extract content from <|channel|>final<|message|>..."""
        marker = "<|channel|>final<|message|>"
        idx = text.rfind(marker)
        if idx == -1:
            return None
        rest = text[idx + len(marker):].strip()
        end = rest.find("<")
        return rest[:end].strip() if end != -1 else rest

    def _build_prompt(self, item, state, items_remaining, high_bid, min_required):
        n = max(state.n_items, 1)
        budget_frac = state.remaining_budget / state.budget if state.budget else 0.0
        lines = [
            f"You are bidding in a sequential auction of {n} items against {n - 1} other bidders.",
            "Your goal is to win as many items as possible while managing your budget.",
            "",
            f"Current item: highest bid so far is {high_bid:.2f}.",
            f"If you bid, your bid will be {min_required:.2f} (the minimum increment).",
            f"Your remaining budget: {state.remaining_budget:.2f} (fraction: {budget_frac:.3f})",
            f"Items remaining: {items_remaining} of {n}",
            "",
            "Output 1 to bid (your bid = the minimum increment).",
            "Output 0 to drop out of this item.",
            "Output only 0 or 1:",
        ]
        return "\n".join(lines)

    def _format_prompt(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": f"Reasoning: {self.reasoning_effort}\nOutput only 0 or 1.",
            },
            {"role": "user", "content": prompt},
        ]
        return MLXBidder._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def _call_mlx(self, prompt: str) -> Optional[int]:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        self.call_count += 1
        t0 = time.perf_counter()
        try:
            formatted = self._format_prompt(prompt)
            sampler = make_sampler(temp=0.0)
            raw = generate(
                MLXBidder._model, MLXBidder._tokenizer,
                prompt=formatted,
                sampler=sampler, verbose=False,
            )
            self.call_durations.append(time.perf_counter() - t0)

            raw_text = raw.strip() if isinstance(raw, str) else raw.text.strip()

            # Try final channel first
            content = self._extract_final_channel(raw_text)
            if content is not None:
                decision = self._parse_binary(content)
                if self.debug:
                    print(f"[MLXBidder {self.bidder_id}] final_channel={repr(content)} -> decision={decision} ({self.call_durations[-1]:.2f}s)")
                return decision

            # No final channel — try parsing whole response
            decision = self._parse_binary(raw_text)
            if self.debug:
                preview = repr(raw_text)[:120]
                print(f"[MLXBidder {self.bidder_id}] no final channel, raw={preview} -> decision={decision} ({self.call_durations[-1]:.2f}s)")
            return decision

        except Exception as exc:
            self.call_durations.append(time.perf_counter() - t0)
            if self.debug:
                print(f"[MLXBidder {self.bidder_id}] Error: {exc!r}")
            return None

    def _build_bid_request(self, item, state, items_remaining):
        bid_history = [float(x) for x in item.bids]
        high_bid = max(bid_history) if bid_history else 0.0
        min_required = high_bid + self.BID_INCREMENT_RATIO * float(item.value)
        return self._build_prompt(item, state, items_remaining, high_bid, min_required), min_required

    def _decision_to_bid(self, decision, state, min_required):
        if decision is None or decision == 0:
            return 0.0
        bid = min_required
        if bid > state.remaining_budget:
            bid = state.remaining_budget
        return float(bid)

    def place_bid(self, item, state, items_remaining, weight_scheme="linear"):
        if state.remaining_budget <= 0:
            return 0.0
        prompt, min_required = self._build_bid_request(item, state, items_remaining)
        decision = self._call_mlx(prompt)
        if self.debug:
            print(f"[MLXBidder {self.bidder_id}] item={item.name} decision={decision} min_req={min_required:.2f} budget={state.remaining_budget:.2f}")
        return self._decision_to_bid(decision, state, min_required)

    async def place_bid_async(self, item, state, items_remaining, weight_scheme="linear"):
        """MLX generate is synchronous — offload to thread."""
        del weight_scheme
        if state.remaining_budget <= 0:
            return 0.0
        prompt, min_required = self._build_bid_request(item, state, items_remaining)
        decision = await asyncio.to_thread(self._call_mlx, prompt)
        if self.debug:
            print(f"[MLXBidder {self.bidder_id}] item={item.name} decision={decision} min_req={min_required:.2f} budget={state.remaining_budget:.2f}")
        return self._decision_to_bid(decision, state, min_required)


_PARAM_RANGES = {
    "PositiveMarginBidder": {"beta": (0.5, 2.5), "min_bid": (0.01, 1.0)},
    "MarginPlusSafetyBidder": {"beta": (0.5, 2.5), "margin": (0.0, 4.0)},
    "BudgetPacedMarginBidder": {"beta": (0.5, 2.5), "c": (0.8, 2.0), "top_k": (1, 5)},
    "TopKSpecialistBidder": {"beta": (0.5, 2.0), "top_k": (1, 5), "margin": (0.0, 2.0)},
    "FlatFractionBidder": {"f": (0.3, 1.5)},
    "DescendingAggressionBidder": {"beta": (0.5, 2.0), "f_start": (0.7, 1.0), "f_end": (0.1, 0.4)},
    "SnipeBidder": {"beta": (0.5, 2.0), "snipe_from_rank": (4, 8), "aggression": (1.0, 2.5)},
    "RandomBidder": {"max_fraction": (0.2, 0.8)},
}


def build_opponent_pool(
    n_opponents: int,
    budget: float,
    seed: Optional[int] = None,
    budget_noise: float = 0.2,
) -> List[BaseBidder]:
    """
    Builds a diverse pool of randomized heuristic opponents.
    Cycles through all 8 bidder types in round-robin order.
    """
    rng = random.Random(seed)
    bidder_classes = [
        PositiveMarginBidder, MarginPlusSafetyBidder, BudgetPacedMarginBidder,
        TopKSpecialistBidder, FlatFractionBidder, DescendingAggressionBidder,
        SnipeBidder, RandomBidder,
    ]
    bidders: List[BaseBidder] = []
    for i in range(n_opponents):
        cls = bidder_classes[i % len(bidder_classes)]
        ranges = _PARAM_RANGES[cls.__name__]
        bgt = budget * rng.uniform(1.0 - budget_noise, 1.0 + budget_noise)
        params: Dict = {}
        for param, (lo, hi) in ranges.items():
            params[param] = rng.randint(lo, hi) if isinstance(lo, int) and isinstance(hi, int) else rng.uniform(lo, hi)
        bidder = cls(bidder_id=i + 1, budget=bgt, **params)
        bidder.set_seed(rng.randrange(2**63))
        bidders.append(bidder)
    return bidders