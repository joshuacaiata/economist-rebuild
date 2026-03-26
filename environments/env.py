import numpy as np
import random
from environments.trading_system import TradingSystem
from agents.mobile_agent import MobileAgent
from environments.logger import Logger
from agents.planner_agent import PlannerAgent
from agents.bank_agent import BankAgent
import torch
# Environment file


class EconomyEnv:
    def __init__(self, config: dict):
        self.config = config

        self.map_size = config["map_size"]
        self.map_path = config["map_path"]
        self.wood_regen_prob = config["wood_regen_prob"]
        self.stone_regen_prob = config["stone_regen_prob"]

        self.map = self.generate_map()
        self.trading_system = TradingSystem(config, self)

        self.agent_risk_aversion = config["agent_risk_aversion"]

        self.n_agents = config["n_agents"]
        self.mobile_agents = []
        for i in range(self.n_agents):
            self.mobile_agents.append(
                MobileAgent(
                    "MobileAgent",
                    i,
                    config, 
                    10 * random.randint(3, 8),
                    self.agent_risk_aversion,
                    self
                )
            )
        
        self.has_bank = config["bank"]
        self.bank = None
        self.bank = BankAgent("BankAgent", self.n_agents, config, self)

        self.has_planner = config["planner"]
        self.planner = None
        self.planner = PlannerAgent("PlannerAgent", self.n_agents, config, self)

        self.all_agents = self.mobile_agents.copy()
        
        self.all_agents.append(self.bank)
        self.all_agents.append(self.planner)
        
        self.agent_initial_positions = {}

        self.initialize_agents()
        self.time = 0
        self.tax_period_length = config["tax_period_length"]
        self.episode_length = config["episode_length"]

        self.eval_folder = config["eval_folder"]

        self.logger = Logger(self)

        random.seed(config["seed"])


    def generate_map(self):
        map_dict = {
            "Wood": np.zeros(self.map_size, dtype=int),
            "Stone": np.zeros(self.map_size, dtype=int),
            "Water": np.zeros(self.map_size, dtype=int),
            "Buildable": np.ones(self.map_size, dtype=int),
            "Houses": np.zeros(self.map_size, dtype=int),
        }

        # Read the map file
        with open(self.map_path, 'r') as f:
            content = f.read()
            map_lines = content.split(';')[:-1]

        self.wood_tiles = []
        self.stone_tiles = []
        self.water_tiles = []
        
        # Process each line of the map
        for row, line in enumerate(map_lines):
            for col, char in enumerate(line):
                if char == 'W':
                    # Wood with probability
                    if random.random() < self.wood_regen_prob:
                        map_dict["Wood"][row, col] = 1
                    
                    self.wood_tiles.append((row, col))
                    map_dict["Buildable"][row, col] = 0
                
                elif char == 'S':
                    # Stone with probability
                    if random.random() < self.stone_regen_prob:
                        map_dict["Stone"][row, col] = 1
                    
                    self.stone_tiles.append((row, col))
                    map_dict["Buildable"][row, col] = 0
                
                elif char == '@':
                    # Water is always blocking
                    map_dict["Water"][row, col] = 1
                    map_dict["Buildable"][row, col] = 0
                    self.water_tiles.append((row, col))
        
        return map_dict

    def initialize_agents(self):
        
        valid_positions = []
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.map["Water"][i, j] == 0:
                    valid_positions.append((i, j))
        
        if len(valid_positions) < self.n_agents:
            raise ValueError(f"Not enough valid positions for {self.n_agents} agents")
        
        selected_positions = random.sample(valid_positions, self.n_agents)
        
        for i, agent in enumerate(self.mobile_agents):
            agent.position = selected_positions[i]
            self.agent_initial_positions[agent.agent_id] = agent.position
            assert self.map["Water"][agent.position[0], agent.position[1]] == 0
    

    def get_observations(self):
        observations = {}
        
        shuffled_agents = random.sample(self.mobile_agents, len(self.mobile_agents))
        
        for agent in shuffled_agents:
            agent_obs = agent.get_observations(self)
            observations[agent.agent_id] = agent_obs
            
        return observations
    
    def regen_tiles(self):
        for i, (row, col) in enumerate(self.wood_tiles):
            if self.map["Wood"][row, col] == 0 and random.random() < self.wood_regen_prob:
                self.map["Wood"][row, col] = 1

        for i, (row, col) in enumerate(self.stone_tiles):
            if self.map["Stone"][row, col] == 0 and random.random() < self.stone_regen_prob:
                self.map["Stone"][row, col] = 1
    
    def reset_year(self):
        # For each agent, call reset year

        shuffled_agents = random.sample(self.mobile_agents, len(self.mobile_agents))

        for agent in shuffled_agents:
            agent.reset_year()

    def reset_env(self, randomize_agent_positions=False):
        # Undo all houses (set to 0)
        self.map["Houses"] = np.zeros(self.map_size, dtype=int)
        
        # Reset Buildable tiles to 1 (except for water, wood, and stone tiles)
        self.map["Buildable"] = np.ones(self.map_size, dtype=int)
        
        # Reset Wood and Stone resources
        self.map["Wood"] = np.zeros(self.map_size, dtype=int)
        self.map["Stone"] = np.zeros(self.map_size, dtype=int)
        
        # Regenerate resources based on probabilities
        for row, col in self.wood_tiles:
            if random.random() < self.wood_regen_prob:
                self.map["Wood"][row, col] = 1
            self.map["Buildable"][row, col] = 0
        
        for row, col in self.stone_tiles:
            if random.random() < self.stone_regen_prob:
                self.map["Stone"][row, col] = 1
            self.map["Buildable"][row, col] = 0
            
        # Set water tiles as non-buildable
        for row, col in self.water_tiles:
            self.map["Buildable"][row, col] = 0

        # Clear all agents inventory
        for agent in self.all_agents:
            agent.reset_episode()
    
        # if randomize_agent_positions, randomize the agent positions
        if randomize_agent_positions:
            self.initialize_agents()

        # otherwise, put them back to the initial positions
        else:
            for agent in self.mobile_agents:
                agent.position = self.agent_initial_positions[agent.agent_id]

        # clear all orders
        self.trading_system.reset_episode()


    def step(self, random_sampling=False):
        time_in_tax_year = self.time % self.tax_period_length
        
        # If it's the first step in the tax year (time_in_tax_year == 0) and we have a planner
        # have the planner go first, then the mobile agents
        if time_in_tax_year == 0 and self.has_planner and self.time > 0:
            planner_obs = self.planner.get_observation(self)
            flattened_obs = self.planner.flatten_observation(planner_obs)
            action = self.planner.get_action(flattened_obs, random_sampling)
            self.planner.step(action)
            self.logger.log_planner_data_timestep(self.planner, self.time)
            
            # Process mobile agents
            mobile_agents = random.sample(self.mobile_agents, len(self.mobile_agents))
            for agent in mobile_agents:
                observations = agent.get_observations(self)
                neighbourhood, numeric = agent.flatten_observation(observations)
                neighbourhood = neighbourhood.unsqueeze(0)
                numeric = numeric.unsqueeze(0)
                action = agent.get_action(neighbourhood, numeric, random_sampling)
                action_success = agent.step(action)
                self.logger.log_agent_data_timestep(agent, action, self.time, action_success)
        
        # Otherwise, for other steps, process all agents in random order
        else:
            mobile_agents = random.sample(self.mobile_agents, len(self.mobile_agents))
            
            # Process mobile agents first
            for agent in mobile_agents:
                observations = agent.get_observations(self)
                neighbourhood, numeric = agent.flatten_observation(observations)
                neighbourhood = neighbourhood.unsqueeze(0)
                numeric = numeric.unsqueeze(0)
                action = agent.get_action(neighbourhood, numeric, random_sampling)
                action_success = agent.step(action)
                self.logger.log_agent_data_timestep(agent, action, self.time, action_success)
            
            # If we have a planner, process it after the mobile agents
            if self.has_planner:
                planner_obs = self.planner.get_observation(self)
                flattened_obs = self.planner.flatten_observation(planner_obs)
                action = self.planner.get_action(flattened_obs, random_sampling)
                self.planner.step(action)
                self.logger.log_planner_data_timestep(self.planner, self.time)
        
        self.trading_system.step()

        if self.has_bank:
            bank_obs = self.bank.get_observation(self)
            flattened_obs = self.bank.flatten_observation(bank_obs)
            action = self.bank.get_action(flattened_obs, random_sampling)
            self.bank.step(action)
            self.logger.log_bank_data_timestep(self.bank, self.time)

        self.logger.log_env_data_timestep(self.time)
        
        if self.time == self.episode_length - 1: 
            self.logger.save_data(self.logger.per_timestep_agent_data, self.config["json_folder"], f"timestep_per_agent_data.json")
            self.logger.save_data(self.logger.per_timestep_env_data, self.config["json_folder"], f"timestep_per_env_data.json")

        self.time += 1
        self.regen_tiles()

        if self.time > 0 and self.time % self.tax_period_length == 0:
            self.reset_year()

    def run_evaluation(self, policy):
        for agent in self.mobile_agents:
            agent.policy_net = policy
            agent.policy_net.eval()
        
        if self.has_planner:
            self.planner.policy_net = policy
            self.planner.policy_net.eval()
        
        self.time = 0
        while self.time < self.episode_length:
            self.step()
        
        self.logger.plot_agent_data(self.logger.per_timestep_agent_data, self.eval_folder + "/agent_data", "evaluation_plots.png")
        self.logger.plot_env_data(self.logger.per_timestep_env_data, self.eval_folder + "/env_data", "evaluation_plots.png")
        if self.has_planner:
            self.logger.plot_planner_data(self.logger.per_timestep_planner_data, self.eval_folder + "/planner_data", "evaluation_plots.png")
        if self.has_bank:
            self.logger.plot_bank_data(self.logger.per_timestep_bank_data, self.eval_folder + "/bank_data", "evaluation_plots.png")
        
    