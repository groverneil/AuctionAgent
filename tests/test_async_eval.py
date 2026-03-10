import asyncio
import time
import unittest
from unittest.mock import AsyncMock

import numpy as np

from bidders import BaseBidder, BidderState, LLMBidder
from env_reward import AuctionEnvironment, Item, OpponentAgent


class SlowAsyncBidder(BaseBidder):
    def __init__(self, bidder_id: int, budget: float, delay: float) -> None:
        super().__init__(bidder_id, budget)
        self.delay = delay

    def place_bid(
        self,
        item: Item,
        state: BidderState,
        items_remaining: int,
        weight_scheme: str = "linear",
    ) -> float:
        del item, state, items_remaining, weight_scheme
        return 0.0

    async def place_bid_async(
        self,
        item: Item,
        state: BidderState,
        items_remaining: int,
        weight_scheme: str = "linear",
    ) -> float:
        del item, state, items_remaining, weight_scheme
        await asyncio.sleep(self.delay)
        return 0.0


class LLMBidderAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bidder = LLMBidder(bidder_id=1, budget=100.0)
        self.item = Item(name="item_1", value=100, rank=1)
        self.item.bids.extend([20.0])
        self.state = BidderState(
            budget=100.0,
            remaining_budget=50.0,
            n_items=1,
        )

    async def test_place_bid_async_uses_valid_numeric_bid(self) -> None:
        self.bidder._call_llm_async = AsyncMock(return_value=35.0)

        bid = await self.bidder.place_bid_async(self.item, self.state, items_remaining=1)

        self.assertEqual(bid, 35.0)

    async def test_place_bid_async_clamps_to_min_required(self) -> None:
        self.bidder._call_llm_async = AsyncMock(return_value=25.0)

        bid = await self.bidder.place_bid_async(self.item, self.state, items_remaining=1)

        self.assertEqual(bid, 30.0)

    async def test_place_bid_async_clamps_to_remaining_budget(self) -> None:
        self.bidder._call_llm_async = AsyncMock(return_value=75.0)

        bid = await self.bidder.place_bid_async(self.item, self.state, items_remaining=1)

        self.assertEqual(bid, 50.0)

    async def test_place_bid_async_returns_zero_on_parse_failure(self) -> None:
        self.bidder._call_llm_async = AsyncMock(return_value=None)

        bid = await self.bidder.place_bid_async(self.item, self.state, items_remaining=1)

        self.assertEqual(bid, 0.0)

    async def test_call_llm_async_returns_none_on_api_exception(self) -> None:
        class FailingCompletions:
            async def create(self, **kwargs):
                del kwargs
                raise RuntimeError("boom")

        class FailingChat:
            completions = FailingCompletions()

        class FailingClient:
            chat = FailingChat()

        self.bidder._get_async_client = lambda: FailingClient()

        bid = await self.bidder._call_llm_async("prompt")

        self.assertIsNone(bid)
        self.assertEqual(self.bidder.call_count, 1)


class AsyncAuctionOverlapTests(unittest.IsolatedAsyncioTestCase):
    def build_env(self, delay: float) -> AuctionEnvironment:
        items = [Item(name="item_1", value=100, rank=1)]
        env = AuctionEnvironment(
            num_agents=2,
            bid_increment_ratio=0.1,
            items=items,
            rng=np.random.default_rng(0),
        )
        env.add_agent(
            OpponentAgent(
                name="A",
                priority=items,
                bidder=SlowAsyncBidder(bidder_id=1, budget=100.0, delay=delay),
            )
        )
        env.add_agent(
            OpponentAgent(
                name="B",
                priority=items,
                bidder=SlowAsyncBidder(bidder_id=2, budget=100.0, delay=delay),
            )
        )
        env.reset()
        return env

    async def run_one(self, delay: float) -> None:
        env = self.build_env(delay)
        await env.run_auction_async()

    async def test_run_auction_async_overlaps_waiting_auctions(self) -> None:
        delay = 0.05
        n_auctions = 4

        start = time.perf_counter()
        for _ in range(n_auctions):
            await self.run_one(delay)
        sequential = time.perf_counter() - start

        start = time.perf_counter()
        await asyncio.gather(*(self.run_one(delay) for _ in range(n_auctions)))
        parallel = time.perf_counter() - start

        self.assertLess(parallel, sequential * 0.7)


if __name__ == "__main__":
    unittest.main()
