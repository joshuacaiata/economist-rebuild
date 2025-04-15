# Base Agent file

class BaseAgent:
    def __init__(
            self, 
            agent_class: str, 
            agent_id: int, 
            config: dict, 
            env
        ):
        self.agent_class = agent_class
        self.agent_id = agent_id
        self.config = config
        self.env = env

    def observe(self, env):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass
