import numpy as np
import torch
from agents.base_agent import BaseAgent

class PlannerAgent(BaseAgent):
    def __init__(self, agent_class, agent_id, config, env):
        super().__init__(agent_class, agent_id, config, env)
        self.agent_id = agent_id
        self.config = config
        self.env = env
        self.agent_class = agent_class

        self.n_tax_brackets = config["n_tax_brackets"]
        self.tax_brackets = config["tax_brackets"] # list of thresholds: [10, 20, 30] would be 4 tax brackets
        assert len(self.tax_brackets) == self.n_tax_brackets - 1
        
        # Initialize tax rates with zeros
        self.tax_rates = np.zeros(self.n_tax_brackets)

        self.tax_collected = 0
        self.total_taxes_collected = 0
        self.taxes_this_episode = []

        self.last_tax_collection_step = 0
        self.last_tax_rate_setting_step = 0

        self.action_space_size = self.n_tax_brackets

        self.previous_year_incomes = {agent.agent_id: 0 for agent in self.env.mobile_agents}
        self.policy_net = None
        self.rollout_buffer = None
        self.obs_stats = None
        
    def get_observation(self, env):
        """
        1. Bids, asks, traded prices, number of trades per resource (wood and stone)
        2. Agent inventories (coins, wood, stone)
        3. Current tax rates, previous year's income, tax brackets
        4. Current time step and time to next tax year
        """

        market_data = self.get_market_data(env)
        agent_inventories = self.get_agent_inventories(env)

        tax_info = {
            "tax_rates": self.tax_rates,
            "previous_year_incomes": self.previous_year_incomes,
            "time_to_tax_year": env.config["tax_period_length"] - (env.time % env.config["tax_period_length"])
        }

        observation = {
            "market_data": market_data,
            "agent_inventories": agent_inventories,
            "tax_info": tax_info,
            "current_step": env.time,
            "time_in_tax_year": env.time % env.config["tax_period_length"]
        }

        return observation
    
    def get_market_data(self, env):
        trading_system = env.trading_system
        # get number of bids, number of asks
        wood_bids = len(trading_system.bids["wood"])
        wood_asks = len(trading_system.asks["wood"])
        stone_bids = len(trading_system.bids["stone"])
        stone_asks = len(trading_system.asks["stone"])

        wood_traded_price = trading_system.last_price["wood"]
        stone_traded_price = trading_system.last_price["stone"]

        wood_trades = trading_system.num_trades["wood"]
        stone_trades = trading_system.num_trades["stone"]

        return {
            "wood_bids": wood_bids,
            "wood_asks": wood_asks,
            "stone_bids": stone_bids,
            "stone_asks": stone_asks,
            "wood_traded_price": wood_traded_price,
            "stone_traded_price": stone_traded_price,
            "wood_trades": wood_trades,
            "stone_trades": stone_trades
        }

    def get_agent_inventories(self, env):
        inventories = []

        for agent in env.mobile_agents:
            inventories.append({
                "coins": agent.inventory["coins"],
                "wood": agent.inventory["wood"],
                "stone": agent.inventory["stone"]
            })
        return inventories
    
    def flatten_observation(self, observation):
        # flatten market data
        market_data = observation["market_data"]
        market_features = np.array([
            market_data["wood_bids"],
            market_data["wood_asks"],
            market_data["stone_bids"],
            market_data["stone_asks"],
            market_data["wood_traded_price"],
            market_data["stone_traded_price"],
            market_data["wood_trades"],
            market_data["stone_trades"]
        ], dtype=np.float32)

        # flatten agent inventories
        agent_inventories = observation["agent_inventories"]
        try:
            inventory_features = np.concatenate([
                [agent["coins"] for agent in agent_inventories],
                [np.mean([agent["coins"] for agent in agent_inventories])],
                [np.std([agent["coins"] for agent in agent_inventories])],
                [agent["wood"] for agent in agent_inventories],
                [agent["stone"] for agent in agent_inventories]
            ], dtype=np.float32)
        except Exception as e:
            print(e)
            # print the types of each element in agent_inventories
            # print the values of each element in agent_inventories
            for i_f in inventory_features:
                print(type(i_f))
                print(i_f)
            raise e

        # flatten tax info
        tax_info = observation["tax_info"]
        tax_rates = tax_info["tax_rates"] # array
        previous_year_incomes = tax_info["previous_year_incomes"] # dict
        previous_year_incomes_array = np.array([previous_year_incomes[agent.agent_id] for agent in self.env.mobile_agents], dtype=np.float32)
        time_to_tax_year = tax_info["time_to_tax_year"] # int
        # take 2 arrays and an int and turn it into a single array
        tax_features = np.concatenate([tax_rates, previous_year_incomes_array, [time_to_tax_year]])

        # flatten time info
        current_step = observation["current_step"] # int
        time_in_tax_year = observation["time_in_tax_year"] # int

        time_features = np.array([current_step, time_in_tax_year], dtype=np.float32)

        # flatten all features
        flattened_observation = np.concatenate([market_features, inventory_features, tax_features, time_features])

        if self.obs_stats is not None:
            std = np.sqrt(self.obs_stats["var"] + 1e-8)
            flattened_observation = (flattened_observation - self.obs_stats["mean"]) / std

        return torch.FloatTensor(flattened_observation)

    def get_action(self, flattened_obs, random_sampling=False):
        """
        Returns a tax rate array with values between 0 and 1 for each tax bracket
        """
        if random_sampling or not self.policy_net:
            # Generate random tax rates between 0 and 1
            return np.random.rand(self.n_tax_brackets)
        
        device = next(self.policy_net.parameters()).device
        flattened_obs = flattened_obs.to(device)

        self.policy_net.eval()
        with torch.no_grad():
            mean, std, value, lstm_state = self.policy_net(flattened_obs.unsqueeze(0))
            tax_rates = mean.squeeze(0)

        return tax_rates.cpu().numpy()
    
    def step(self, action):
        current_step = self.env.time
        tax_period_length = self.env.config["tax_period_length"]
        time_in_tax_year = current_step % tax_period_length

        # Set tax rates
        if time_in_tax_year == 0 and current_step != self.last_tax_rate_setting_step:
            self.tax_rates = action
            self.last_tax_rate_setting_step = current_step
            return True
        
        # Collect and redistribute taxes
        if time_in_tax_year == tax_period_length - 1 and current_step != self.last_tax_collection_step:
            self._collect_and_redistribute_taxes()
            self.last_tax_collection_step = current_step
            return True
        
        return True
    
    def _collect_and_redistribute_taxes(self):
        # Store incomes as a dictionary mapping agent IDs to incomes
        self.previous_year_incomes = {
            agent.agent_id: agent.inventory["coins"] - agent.ending_coins_previous_year 
            for agent in self.env.mobile_agents
        }

        total_tax = 0

        for agent in self.env.mobile_agents:
            income = agent.inventory["coins"] - agent.ending_coins_previous_year
            tax = self._calculate_tax(income)
            if agent.inventory["coins"] < tax:
                raise ValueError("Agent has negative coins")
            agent.inventory["coins"] -= tax
            total_tax += tax

        self.total_taxes_collected += total_tax
        self.taxes_this_episode.append(total_tax)
        self.tax_collected = total_tax

        if total_tax > 0 and self.env.mobile_agents:
            redistribution_amount = total_tax / len(self.env.mobile_agents)
            for agent in self.env.mobile_agents:
                agent.inventory["coins"] += redistribution_amount
        
    def _calculate_tax(self, income):
        tax = 0
        
        for i in range(self.n_tax_brackets):
            if i == 0:
                bracket_min = 0
                bracket_max = self.tax_brackets[0] if self.n_tax_brackets > 1 else float('inf')
            elif i == self.n_tax_brackets - 1:
                bracket_min = self.tax_brackets[i-1]
                bracket_max = float('inf')
            else:
                bracket_min = self.tax_brackets[i-1]
                bracket_max = self.tax_brackets[i]
                
            income_in_bracket = max(0, min(income, bracket_max) - bracket_min)
            
            tax += income_in_bracket * self.tax_rates[i]
                
        return tax
            
    def get_utility(self):
        utility_type = self.config.get("utility_type", "utilitarian")

        assert utility_type in ["utilitarian", "nash_welfare"]

        if utility_type == "utilitarian":
            return self.get_utilitarian_utility()
        else:
            return self.get_nash_welfare_utility()
        
    def get_utilitarian_utility(self):
        agent_coins = {
            agent.agent_id: max(1e-8, agent.inventory["coins"])
            for agent in self.env.mobile_agents
        }
        agent_utilities = {
            agent.agent_id: agent.get_utility()
            for agent in self.env.mobile_agents
        }

        inverse_weights = {
            agent_id: 1.0 / coins
            for agent_id, coins in agent_coins.items()
        }
        
        total_inverse_weight = sum(inverse_weights.values())
        if total_inverse_weight > 0:
            normalized_weights = {
                agent_id: weight / total_inverse_weight
                for agent_id, weight in inverse_weights.items()
            }
        else:
            normalized_weights = {
                agent_id: 1.0 / len(self.env.mobile_agents)
                for agent_id in agent_coins.keys()
            }
        
        return sum(
            normalized_weights[agent_id] * agent_utilities[agent_id]
            for agent_id in agent_coins.keys()
        )
        
    def get_nash_welfare_utility(self):
        coins = [agent.inventory["coins"] for agent in self.env.mobile_agents]
        n_agents = len(self.env.mobile_agents)

        if n_agents <= 1:
            return sum(coins)
        
        gini = self.calculate_gini(coins)

        equality = 1 - (n_agents / (n_agents - 1)) * gini

        productivity = sum(coins)

        return equality * productivity
    
    def calculate_gini(self, values):
        numerator = 0
        for i, value in enumerate(values):
            for j, value2 in enumerate(values):
                numerator += abs(value - value2)
        
        denominator = 2 * len(values) * sum(values)

        return numerator / (denominator + 1e-8)

    def reset_year(self):
        self.tax_collected = 0
    
    def reset_episode(self):
        self.tax_rates = np.zeros(self.n_tax_brackets)
        self.tax_collected = 0
        self.total_taxes_collected = 0
        self.taxes_this_episode = []
        self.previous_year_incomes = {agent.agent_id: 0 for agent in self.env.mobile_agents}
        self.last_tax_collection_step = 0
        self.last_tax_rate_setting_step = 0

    def get_action_mask(self):
        current_step = self.env.time
        tax_period_length = self.env.config["tax_period_length"]
        time_in_tax_year = current_step % tax_period_length

        return time_in_tax_year == 0 and current_step > 0
    
    def get_tax_bracket(self, income):
        if not self.tax_brackets or income < self.tax_brackets[0]:
            return 0
        
        for i in range(len(self.tax_brackets) - 1):
            if self.tax_brackets[i] <= income < self.tax_brackets[i + 1]:
                return i + 1
        
        return self.n_tax_brackets - 1

        

