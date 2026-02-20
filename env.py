from typing import List, Dict, Any

class Item:
    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value
        self.bids = []

class Agent:
    def __init__(self, name: str, priority: List[Item]):
        self.name = name
        self.priority = priority
        self.type = None

    def bind_model(self, model: Any, params: Dict[str, Any]):
        # define the type of agent model here
        if model["type"] == "rl":
            # self.model = RLModel(params)
            self.type = "rl"
        else:
            self.type = "llm"
            # self.model = LLMModel(params)

class AuctionEnvironment:
    def __init__(self, num_agents: int, items: List[Item]):
        self.num_agents = num_agents
        self.items = items
        self.agents = []

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

