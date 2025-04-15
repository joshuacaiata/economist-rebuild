import torch
import numpy as np
from agents.base_agent import BaseAgent

class BankAgent(BaseAgent):
    def __init__(self, agent_class, agent_id, config, env):
        super().__init__(agent_class, agent_id, config, env)
        self.agent_id = agent_id
        self.config = config
        self.env = env
        self.agent_class = agent_class

        self.target_inflation = config["target_inflation"]
        self.starting_interest_rate = config["starting_interest_rate"]

        self.min_interest_rate_limit = self.starting_interest_rate - 0.01 
        self.max_interest_rate_limit = self.starting_interest_rate + 0.01
        self.final_min_interest_rate = config.get("final_min_interest_rate", -2.5)  # -250%
        self.final_max_interest_rate = config.get("final_max_interest_rate", 2.5)   # 250%

        self.min_monetary_injection_limit = -1
        self.max_monetary_injection_limit = 1
        self.final_min_monetary_injection = config.get("final_min_monetary_injection", -100)
        self.final_max_monetary_injection = config.get("final_max_monetary_injection", 100)

        self.annealing_duration = config.get("annealing_duration", 50) 
        self.current_episode = 0

        # track all of the trades per resource
        # have the format (time, price)
        self.trades = {
            "wood": [],
            "stone": [],
        }

        self.money_supply = 0

        self.inflation_rates = []

        self.inflation_rate = 0
        self.interest_rate = self.starting_interest_rate

        self.action_range = 7

        self.utility_alpha = self.config.get("utility_alpha", 100)

        self.utility_0 = self.get_utility()
        self.utility = self.get_utility()

        self.monetary_injections = 0

    def get_observation(self, env):
        # target inflation, interest rate, inflation rate
        inflation_rate = self.get_inflation_rate()
        money_supply = self.get_money_supply()
        observation = {
            "inflation_rate": inflation_rate,
            "interest_rate": self.interest_rate,
            "target_inflation": self.target_inflation,
            "money_supply": money_supply
        }
        return observation
    
    def get_money_supply(self):
        money_supply = 0
        for agent in self.env.mobile_agents:
            money_supply += agent.inventory["coins"]
        return money_supply

    def get_inflation_rate(self):
        inflation_rates = []
        for resource in self.trades:
            current_prices = [price for time, price in self.trades[resource] if time == self.env.time]
            if len(current_prices) == 0:
                inflation_rates.append(0)
            else:
                current_price = sum(current_prices) / len(current_prices)
                last_trade_times = [time for time, _ in self.trades[resource] if time < self.env.time]
                if len(last_trade_times) == 0:
                    inflation_rates.append(0)
                else:
                    last_trade_time = max(last_trade_times)
                    last_trade_prices = [price for time, price in self.trades[resource] if time == last_trade_time]
                    last_trade_price = sum(last_trade_prices) / len(last_trade_prices)
                    last_trade_price = max(last_trade_price, 1)
                    inflation_rate = (current_price - last_trade_price) / (last_trade_price)
                    inflation_rates.append(inflation_rate)

        inflation_rate = sum(inflation_rates) / len(inflation_rates)
        self.inflation_rates.append(inflation_rate)

        if len(self.inflation_rates) < self.env.config["tax_period_length"]:
            return sum(self.inflation_rates) / len(self.inflation_rates)
        else:
            return sum(self.inflation_rates[-self.env.config["tax_period_length"]:]) / self.env.config["tax_period_length"]
    
    def flatten_observation(self, observation):
        features = [observation["inflation_rate"], observation["interest_rate"], observation["target_inflation"], observation["money_supply"]]
        features = np.array(features, dtype=np.float32)
        return torch.FloatTensor(features)
    
    def get_action_mask(self):
        if self.env.time % self.env.config["tax_period_length"] == 0:
            mask = np.ones(self.action_range)
            mask[-2:] = 0
        elif self.env.time % self.env.config["tax_period_length"] == self.env.config["tax_period_length"] - 1:
            mask = np.ones(self.action_range)
            mask[:-2] = 0
        else:
            mask = np.zeros(self.action_range)
            mask[0] = 1
            return mask

        
        if self.monetary_injections >= self.max_monetary_injection_limit:
            mask[5] = 0
        if self.monetary_injections <= self.min_monetary_injection_limit:
            mask[6] = 0

        if self.interest_rate + 0.025 > self.max_interest_rate_limit:
            mask[1] = 0
            mask[3] = 0
        
        if self.interest_rate - 0.025 < self.min_interest_rate_limit:
            mask[2] = 0
            mask[4] = 0
            
        if self.max_interest_rate_limit - self.interest_rate < 0.01 and self.max_interest_rate_limit - self.interest_rate >= 0.0025:
            mask[3] = 0 
            
        if self.interest_rate - self.min_interest_rate_limit < 0.01 and self.interest_rate - self.min_interest_rate_limit >= 0.0025:
            mask[4] = 0 
        
        return mask
    
    def get_action(self, flattened_obs, random_sampling=False):
        action_mask = self.get_action_mask()
        
        if random_sampling:
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                return int(np.random.choice(valid_actions))
            return 0
        
        device = next(self.policy_net.parameters()).device
        flattened_obs = flattened_obs.to(device)
        
        self.policy_net.eval()
        with torch.no_grad():
            logits, _, _ = self.policy_net(flattened_obs.unsqueeze(0))
            
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=device)
            action_mask_tensor = action_mask_tensor.unsqueeze(0)
            masked_logits = logits.clone()
            masked_logits[~action_mask_tensor] = -1e10 
            
            probs = torch.nn.functional.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action_tensor = dist.sample()
            
        return int(action_tensor.item())

    def update_annealing_limits(self, episode):
        """Update interest rate and monetary injection limits based on current episode"""
        self.current_episode = episode
        progress = min(1.0, episode / self.annealing_duration)
        
        # Anneal interest rate limits
        self.min_interest_rate_limit = self.starting_interest_rate - 0.01 + progress * (self.final_min_interest_rate - (self.starting_interest_rate - 0.01))
        self.max_interest_rate_limit = self.starting_interest_rate + 0.01 + progress * (self.final_max_interest_rate - (self.starting_interest_rate + 0.01))
        
        # Anneal monetary injection limits
        self.min_monetary_injection_limit = -1 + progress * (self.final_min_monetary_injection + 1)
        self.max_monetary_injection_limit = 1 + progress * (self.final_max_monetary_injection - 1)
        
    def step(self, action):
        for agent in self.env.mobile_agents:
            n = self.env.config["tax_period_length"]
            step_interest = (1 + self.interest_rate) ** (1/n) - 1
            agent.inventory["coins"] += step_interest * agent.inventory["coins"]
            agent.build_payout += step_interest * agent.build_payout

        # Apply interest rate changes with limits
        if action == 1:
            new_interest_rate = self.interest_rate + 0.0025
            self.interest_rate = min(self.max_interest_rate_limit, max(self.min_interest_rate_limit, new_interest_rate))
        
        elif action == 2:
            new_interest_rate = self.interest_rate - 0.0025
            self.interest_rate = min(self.max_interest_rate_limit, max(self.min_interest_rate_limit, new_interest_rate))

        elif action == 3:
            new_interest_rate = self.interest_rate + 0.01
            self.interest_rate = min(self.max_interest_rate_limit, max(self.min_interest_rate_limit, new_interest_rate))

        elif action == 4:
            new_interest_rate = self.interest_rate - 0.01
            self.interest_rate = min(self.max_interest_rate_limit, max(self.min_interest_rate_limit, new_interest_rate))

        elif action == 5:
            self.increase_money_supply()
            new_injection_count = self.monetary_injections + 1
            if new_injection_count <= self.max_monetary_injection_limit:
                self.monetary_injections = new_injection_count
        
        elif action == 6:
            self.decrease_money_supply()
            new_injection_count = self.monetary_injections - 1
            if new_injection_count >= self.min_monetary_injection_limit:
                self.monetary_injections = new_injection_count

        self.inflation_rate = self.get_inflation_rate()
        self.money_supply = self.get_money_supply()
        self.utility = self.get_utility()

    def increase_money_supply(self):
        for agent in self.env.mobile_agents:
            agent.inventory["coins"] = agent.inventory["coins"] * 1.1
        self.money_supply = self.get_money_supply()
        
    def decrease_money_supply(self):
        for agent in self.env.mobile_agents:
            agent.inventory["coins"] = agent.inventory["coins"] * 0.9
        self.money_supply = self.get_money_supply()

    def get_utility(self):
        utility = (self.utility_alpha * (self.target_inflation - self.inflation_rate)) ** 2
        return 1 - utility
    
    def reset_year(self):
        return
    
    def reset_episode(self):
        self.interest_rate = self.starting_interest_rate
        self.money_supply = 0
        self.inflation_rate = 0
        self.monetary_injections = 0
        self.trades = {
            "wood": [],
            "stone": [],
        }

    
    

    
    
