import torch
import torch.multiprocessing as mp
import numpy as np
from environments.env import EconomyEnv
import copy
import multiprocessing

class ZeroTaxPlannerWrapper:
    def __init__(self, planner):
        self.planner = planner
        
    def __getattr__(self, name):
        return getattr(self.planner, name)
    
    def get_action(self, flattened_obs, random_sampling=False):
        return np.zeros(self.planner.n_tax_brackets)

class ZeroBankPolicyWrapper:
    def __init__(self, bank):
        self.bank = bank
        self.bank.target_inflation = 0.0
        self.bank.interest_rate = 0.0
        self.bank.starting_interest_rate = 0.0
        
    def __getattr__(self, name):
        return getattr(self.bank, name)
    
    def get_action(self, flattened_obs, random_sampling=False):
        return 0

class VectorizedEnv:
    def __init__(self, config):
        self.config = config
        self.total_envs = config.get("n_envs", 4)
        
        self.available_cores = multiprocessing.cpu_count()
        print(f"Available CPU cores: {self.available_cores}")
        
        self.batch_size = max(1, int(self.available_cores * 0.9))
        print(f"Running environments in batches of {self.batch_size}")
        
        self.mp_ctx = mp.get_context('spawn')
        self.processes = []
        self.pipes = []
        
        self.env_ref = EconomyEnv(config.copy())
        
        for i in range(self.total_envs):
            parent_conn, child_conn = self.mp_ctx.Pipe()
            env_config = config.copy()
            env_config["seed"] = config.get("seed", 42) + i
            
            p = self.mp_ctx.Process(
                target=self._env_worker, 
                args=(child_conn, env_config, i)
            )
            p.daemon = True
            p.start()
            self.processes.append(p)
            self.pipes.append(parent_conn)
            
        self.pipes[0].send(("get_info", None))
        info = self.pipes[0].recv()
        self.n_agents = info["n_agents"]
        self.map_size = info["map_size"]
        self.tax_period_length = info["tax_period_length"]
        self.episode_length = info["episode_length"]
            
    @staticmethod
    def _env_worker(pipe, config, worker_id):
        env = EconomyEnv(config)
        
        while True:
            try:
                cmd, data = pipe.recv()
                
                if cmd == "step":
                    env.trading_system.step()
                    env.time += 1
                    env.regen_tiles()
                    if env.time > 0 and env.time % env.tax_period_length == 0:
                        env.reset_year()
                    pipe.send(("ok", None))
                    
                elif cmd == "update_environment":
                    env.trading_system.step()
                    env.time += 1
                    env.regen_tiles()
                    if env.time > 0 and env.time % env.tax_period_length == 0:
                        env.reset_year()
                    pipe.send(("ok", None))
                    
                elif cmd == "reset":
                    randomize = data if data is not None else True
                    env.reset_env(randomize)
                    pipe.send(("ok", None))
                    
                elif cmd == "update_config":
                    for key, value in data.items():
                        config[key] = value
                        if key.endswith("_labour"):
                            for agent in env.mobile_agents:
                                agent.config[key] = value
                    pipe.send(("ok", None))
                    
                elif cmd == "get_info":
                    info = {
                        "n_agents": env.n_agents,
                        "map_size": env.map_size, 
                        "tax_period_length": env.tax_period_length,
                        "episode_length": env.episode_length
                    }
                    pipe.send(info)
                    
                elif cmd == "get_agents_data":
                    agents_data = []
                    for agent in env.mobile_agents:
                        obs = agent.get_observations(env)
                        agent_data = {
                            "agent_id": agent.agent_id,
                            "utility": agent.get_utility(),
                            "obs": obs
                        }
                        agents_data.append(agent_data)
                    pipe.send(agents_data)
                    
                elif cmd == "agent_step":
                    agent_id, action = data
                    agent = next((a for a in env.mobile_agents if a.agent_id == agent_id), None)
                    if agent:
                        prev_util = agent.get_utility()
                        agent.step(action)
                        current_util = agent.get_utility()
                        reward = current_util - prev_util
                        pipe.send((reward, current_util))
                    else:
                        pipe.send((0.0, 0.0))
                    
                elif cmd == "close":
                    pipe.close()
                    break
                    
                elif cmd == "get_agent_utility":
                    agent_id = data
                    agent = next((a for a in env.mobile_agents if a.agent_id == agent_id), None)
                    if agent:
                        utility = agent.get_utility()
                        pipe.send(utility)
                    else:
                        pipe.send(0.0)
                    
                elif cmd == "get_agent_action_mask":
                    agent_id = data
                    agent = next((a for a in env.mobile_agents if a.agent_id == agent_id), None)
                    if agent:
                        action_mask = agent.get_action_mask()
                        pipe.send(action_mask)
                    else:
                        pipe.send(np.ones(env.mobile_agents[0].action_range + 1, dtype=bool))
                
                elif cmd == "get_planner_obs":
                    if env.has_planner and env.planner:
                        planner_obs = env.planner.get_observation(env)
                        flattened_obs = env.planner.flatten_observation(planner_obs)
                        pipe.send(flattened_obs.numpy())
                    else:
                        pipe.send(None)
                
                elif cmd == "get_planner_action_mask":
                    if env.has_planner and env.planner:
                        action_mask = env.planner.get_action_mask()
                        pipe.send(action_mask)
                    else:
                        pipe.send(False)
                
                elif cmd == "planner_step":
                    action = data
                    if env.has_planner and env.planner:
                        pre_utility = env.planner.get_utility()
                        env.planner.step(action)
                        post_utility = env.planner.get_utility()
                        pipe.send((pre_utility, post_utility))
                    else:
                        pipe.send((0.0, 0.0))
                
                elif cmd == "get_planner_utility":
                    if env.has_planner and env.planner:
                        utility = env.planner.get_utility()
                        pipe.send(utility)
                    else:
                        pipe.send(0.0)
                
                elif cmd == "wrap_planner_with_zero_tax":
                    if env.has_planner and env.planner:
                        env.planner = ZeroTaxPlannerWrapper(env.planner)
                    pipe.send(("ok", None))

                elif cmd == "get_bank_obs":
                    if env.has_bank and env.bank:
                        bank_obs = env.bank.get_observation(env)
                        flattened_obs = env.bank.flatten_observation(bank_obs)
                        pipe.send(flattened_obs)
                    else:
                        pipe.send(None)
                
                elif cmd == "get_bank_action_mask":
                    if env.has_bank and env.bank:
                        action_mask = env.bank.get_action_mask()
                        pipe.send(action_mask)
                    else:
                        pipe.send(False)
                
                elif cmd == "bank_step":
                    action = data
                    if env.has_bank and env.bank:
                        pre_utility = env.bank.get_utility()
                        env.bank.step(action)
                        post_utility = env.bank.get_utility()
                        pipe.send((pre_utility, post_utility))
                    else:
                        pipe.send((0.0, 0.0))
                
                elif cmd == "get_bank_utility":
                    if env.has_bank and env.bank:
                        utility = env.bank.get_utility()
                        pipe.send(utility)
                    else:
                        pipe.send(0.0)
                
                elif cmd == "wrap_bank_with_zero_policy":
                    if env.has_bank and env.bank:
                        env.bank = ZeroBankPolicyWrapper(env.bank)
                    pipe.send(("ok", None))
                    
                elif cmd == "update_bank_annealing":
                    episode = data
                    if env.has_bank and env.bank:
                        env.bank.update_annealing_limits(episode)
                    pipe.send(("ok", None))
                    
            except (EOFError, BrokenPipeError):
                break

    def _process_batch(self, start_idx, end_idx, cmd, data=None):
        results = []
        for i in range(start_idx, end_idx):
            try:
                self.pipes[i].send((cmd, data))
                result = self.pipes[i].recv()
                results.append((i, result))
            except (EOFError, BrokenPipeError):
                results.append((i, None))
        return results

    def step_envs(self):
        all_results = []
        for start_idx in range(0, self.total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.total_envs)
            batch_results = self._process_batch(start_idx, end_idx, "step")
            all_results.extend(batch_results)
        return all_results

    def reset_all(self, randomize_agent_positions=True):
        all_results = []
        for start_idx in range(0, self.total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.total_envs)
            batch_results = self._process_batch(start_idx, end_idx, "reset", randomize_agent_positions)
            all_results.extend(batch_results)
        return all_results

    def get_all_agent_observations(self):
        all_agents_data = []
        for start_idx in range(0, self.total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.total_envs)
            batch_results = self._process_batch(start_idx, end_idx, "get_agents_data")
            
            for env_idx, agents_data in batch_results:
                if agents_data is not None:
                    for agent_data in agents_data:
                        agent_data["env_idx"] = env_idx
                        all_agents_data.append(agent_data)

        return all_agents_data

    def agent_step(self, env_idx, agent_id, action):
        if env_idx >= self.total_envs:
            return 0.0, 0.0
            
        try:
            self.pipes[env_idx].send(("agent_step", (agent_id, action)))
            reward, current_util = self.pipes[env_idx].recv()
            return reward, current_util
        except (EOFError, BrokenPipeError):
            return 0.0, 0.0

    def get_agent_utility(self, env_idx, agent_id):
        if env_idx >= self.total_envs:
            return 0.0
        
        try:
            self.pipes[env_idx].send(("get_agent_utility", agent_id))
            utility = self.pipes[env_idx].recv()
            return utility
        except (EOFError, BrokenPipeError):
            return 0.0

    def get_agent_action_mask(self, env_idx, agent_id):
        if env_idx >= self.total_envs:
            return np.ones(self.env_ref.mobile_agents[0].action_range + 1, dtype=bool)
            
        try:
            self.pipes[env_idx].send(("get_agent_action_mask", agent_id))
            action_mask = self.pipes[env_idx].recv()
            return action_mask
        except (EOFError, BrokenPipeError):
            return np.ones(self.env_ref.mobile_agents[0].action_range + 1, dtype=bool)

    def get_planner_utility(self, env_idx):
        if env_idx >= self.total_envs:
            return 0.0
        
        try:
            self.pipes[env_idx].send(("get_planner_utility", None))
            utility = self.pipes[env_idx].recv()
            return utility
        except (EOFError, BrokenPipeError):
            return 0.0

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except (EOFError, BrokenPipeError):
                continue
        
        for p in self.processes:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
