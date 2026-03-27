import yaml
from environments.env import EconomyEnv
import os
import torch
import numpy as np
import argparse
from training.mobile_agent_policy import MobileAgentPolicy
from training.vectorized_env import ZeroTaxPlannerWrapper

def main(config_path: str, try_load: bool):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    env = EconomyEnv(config)
    
    if env.has_planner:
        env.planner = ZeroTaxPlannerWrapper(env.planner)
    
    os.makedirs("random_plots/agent_data", exist_ok=True)
    os.makedirs("random_plots/env_data", exist_ok=True)
    os.makedirs("random_plots/planner_data", exist_ok=True)

    n_agents = config.get("n_agents", 4)
    experiment_name = config.get("experiment_name", "default_experiment")
    network_name = f"mobile_agents-phase_1-n_agents={n_agents}-" \
            f"experiment_name={experiment_name}" \

    network = None

    if try_load:
        random_sampling = False
        try:
            network_path = os.path.join("networks", f"{network_name}_COMPLETE.pth")
            print(f"Loading network from {network_path}")
            state_dict = torch.load(network_path, map_location=torch.device('cpu'))
            
            basic_numeric_size = 7 + n_agents  # position(2) + inventory(3) + build_payout(1) + time_to_tax(1) + incomes(n_agents)
            planner_features_size = config.get("n_tax_brackets", 7) + 1  # tax_rates (n_brackets) + tax_bracket
            bank_features_size = 4  # inflation_rate, interest_rate, money_supply
            
            network = MobileAgentPolicy(
                config=config,
                num_numeric=basic_numeric_size + planner_features_size + bank_features_size,
                action_range=env.mobile_agents[0].action_range
            )
            network.load_state_dict(state_dict)
            print(f"Loaded complete network from {network_path}")
        except FileNotFoundError:
            try:
                print(f"Loading partial network from {network_path}")
                network_path = os.path.join("networks", f"{network_name}_PARTIAL.pth")
                state_dict = torch.load(network_path, map_location=torch.device('cpu'))
                
                basic_numeric_size = 7 + n_agents  # position(2) + inventory(3) + build_payout(1) + time_to_tax(1) + incomes(n_agents)
                planner_features_size = config.get("n_tax_brackets", 7) + 1  # tax_rates (n_brackets) + tax_bracket
                bank_features_size = 4 # inflation_rate, interest_rate, money_supply
                
                network = MobileAgentPolicy(
                    config=config,
                    num_numeric=basic_numeric_size + planner_features_size + bank_features_size,
                    action_range=env.mobile_agents[0].action_range
                )
                network.load_state_dict(state_dict)
                print(f"Loaded partial network from {network_path}")
            except FileNotFoundError:
                print(f"No network found. Running random actions.")
                random_sampling = True
    else:
        random_sampling = True

    if network is not None:
        stats_path = os.path.join("networks", f"{network_name}_COMPLETE_obs_stats.npz")
        if not os.path.exists(stats_path):
            stats_path = os.path.join("networks", f"{network_name}_PARTIAL_obs_stats.npz")
        obs_stats = None
        if os.path.exists(stats_path):
            data = np.load(stats_path)
            obs_stats = {"mean": data["mean"], "var": data["var"]}
            print(f"Loaded obs normalization stats from {stats_path}")
        for agent in env.mobile_agents:
            agent.policy_net = network
            agent.obs_stats = obs_stats

    
    if random_sampling:
        print(f"Running random actions for {env.episode_length} steps...")
    while env.time < env.episode_length:
        env.step(random_sampling=random_sampling)
        if env.time % 100 == 0:
            print(f"Step {env.time}/{env.episode_length}")
    
    env.logger.plot_agent_data(env.logger.per_timestep_agent_data, 
                             "random_plots/agent_data", 
                             "random_evaluation_plots.png")
    env.logger.plot_env_data(env.logger.per_timestep_env_data, 
                            "random_plots/env_data", 
                            "random_evaluation_plots.png")
    if env.has_planner:
        env.logger.plot_planner_data(env.logger.per_timestep_planner_data, 
                                    "random_plots/planner_data", 
                                    "random_evaluation_plots.png")
    if env.has_bank:
        env.logger.plot_bank_data(env.logger.per_timestep_bank_data, 
                                "random_plots/bank_data", 
                                "random_evaluation_plots.png")
    print("Done! Plots saved in random_plots directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--try-load", action="store_true", default=False, help="Try to load in the network")
    args = parser.parse_args()

    main(args.config, args.try_load)
