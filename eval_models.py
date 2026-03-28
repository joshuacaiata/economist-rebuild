import yaml
from environments.env import EconomyEnv
import os
import torch
import argparse
import numpy as np
import scipy.stats
from training.mobile_agent_policy import MobileAgentPolicy
from training.planner_policy import PlannerPolicy
from training.bank_policy import BankPolicy
from training.vectorized_env import ZeroTaxPlannerWrapper, ZeroBankPolicyWrapper

def load_obs_stats(network_path):
    stats_path = network_path.replace("_COMPLETE.pth", "_COMPLETE_obs_stats.npz")
    if not os.path.exists(stats_path):
        stats_path = network_path.replace("_COMPLETE.pth", "_PARTIAL_obs_stats.npz")
    if os.path.exists(stats_path):
        data = np.load(stats_path)
        print(f"Loaded obs normalization stats from {stats_path}")
        return {"mean": data["mean"], "var": data["var"]}
    print("No obs normalization stats found, using raw observations")
    return None

def apply_obs_normalization(obs_numeric, obs_stats):
    if obs_stats is None:
        return obs_numeric
    std = np.sqrt(obs_stats["var"] + 1e-8)
    return (obs_numeric - obs_stats["mean"]) / std

def load_network(network_path, network_class, config, **kwargs):
    try:
        state_dict = torch.load(network_path, map_location=torch.device('cpu'))
        network = network_class(config=config, **kwargs)
        network.load_state_dict(state_dict)
        print(f"Loaded network from {network_path}")
        return network
    except FileNotFoundError:
        partial_path = network_path.replace("_COMPLETE.pth", "_PARTIAL.pth")
        try:
            state_dict = torch.load(partial_path, map_location=torch.device('cpu'))
            network = network_class(config=config, **kwargs)
            network.load_state_dict(state_dict)
            print(f"Loaded partial network from {partial_path}")
            return network
        except FileNotFoundError:
            print(f"No network found at {network_path} or {partial_path}")
            return None

def plot_data(env, plot_folder, envs_list=None):
    if envs_list is None:
        env.logger.plot_agent_data(env.logger.per_timestep_agent_data, 
                                 f"{plot_folder}/agent_data", 
                                 "evaluation_plots.png")
        env.logger.plot_env_data(env.logger.per_timestep_env_data, 
                                f"{plot_folder}/env_data", 
                                "evaluation_plots.png")
        if env.has_planner:
            env.logger.plot_planner_data(env.logger.per_timestep_planner_data, 
                                        f"{plot_folder}/planner_data", 
                                        "evaluation_plots.png")
        if env.has_bank:
            env.logger.plot_bank_data(env.logger.per_timestep_bank_data,
                                     f"{plot_folder}/bank_data",
                                     "evaluation_plots.png")
    else:
        print(f"Generating plots with statistics across {len(envs_list)} runs...")
        
        env.logger.plot_agent_data(env.logger.per_timestep_agent_data, 
                                 f"{plot_folder}/agent_data", 
                                 "evaluation_plots.png")
        
        env_dicts = [e.logger.per_timestep_env_data for e in envs_list]
        planner_dicts = [e.logger.per_timestep_planner_data for e in envs_list]
        bank_dicts = [e.logger.per_timestep_bank_data for e in envs_list]

        env.logger.plot_env_data(env_dicts, f"{plot_folder}/env_data", "evaluation_plots.png")
        
        if env.has_planner:
            env.logger.plot_planner_data(planner_dicts, f"{plot_folder}/planner_data", "evaluation_plots.png")
        
        if env.has_bank:
            env.logger.plot_bank_data(bank_dicts, f"{plot_folder}/bank_data", "evaluation_plots.png")

def calculate_r_squared(x, y):
    if len(x) <= 1 or len(y) <= 1:
        return 0.0
    
    correlation_coefficient, _ = scipy.stats.pearsonr(x, y)
    
    r_squared = correlation_coefficient ** 2
    return r_squared

def main(config_path: str, try_load: bool, phase: int, episode_length: int, plot_folder: str, get_collection_houses: bool, num_collections: int = 0):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(plot_folder, exist_ok=True)
    os.makedirs(os.path.join(plot_folder, "agent_data"), exist_ok=True)
    os.makedirs(os.path.join(plot_folder, "env_data"), exist_ok=True)
    os.makedirs(os.path.join(plot_folder, "planner_data"), exist_ok=True)
    os.makedirs(os.path.join(plot_folder, "bank_data"), exist_ok=True)
    
    if get_collection_houses and num_collections > 0:
        print(f"Running experiment {num_collections} times to collect data...")
        
        all_build_payouts = []
        all_houses_built = []
        all_environments = []
        
        for run in range(num_collections):
            print(f"\nRun {run+1}/{num_collections}")
            
            env = EconomyEnv(config)
            
            n_agents = config.get("n_agents", 4)
            experiment_name = config.get("experiment_name", "default_experiment")
            networks_dir = config.get("network_folder", "networks")

            mobile_network = None
            planner_network = None
            bank_network = None
            random_sampling = True

            if try_load:
                mobile_network_name = f"mobile_agents-phase_{phase}-n_agents={n_agents}-experiment_name={experiment_name}"
                mobile_network_path = os.path.join(networks_dir, f"{mobile_network_name}_COMPLETE.pth")
                
                basic_numeric_size = 7 + n_agents  # position(2) + inventory(3) + build_payout(1) + time_to_tax(1) + incomes(n_agents)
                planner_features_size = config.get("n_tax_brackets", 7) + 1  # tax_rates (n_brackets) + tax_bracket
                bank_features_size = 4  # inflation_rate, interest_rate, money_supply

                mobile_network = load_network(
                    mobile_network_path,
                    MobileAgentPolicy,
                    config,
                    num_numeric=basic_numeric_size + planner_features_size + bank_features_size,
                    action_range=env.mobile_agents[0].action_range
                )
                
                if mobile_network is not None:
                    random_sampling = False

                if phase >= 2 and env.has_planner:
                    planner_network_name = f"planner_agent-phase_{phase}-n_agents={n_agents}-experiment_name={experiment_name}"
                    planner_network_path = os.path.join(networks_dir, f"{planner_network_name}_COMPLETE.pth")
                    
                    n_tax_brackets = config.get("n_tax_brackets", 7)
                    observation_size = 8 + (3 * n_agents + 2) + (n_tax_brackets + n_agents + 1) + 2
                    
                    planner_network = load_network(
                        planner_network_path,
                        PlannerPolicy,
                        config,
                        input_size=observation_size,
                        output_size=n_tax_brackets
                    )

                if phase == 3 and env.has_bank:
                    bank_network_name = f"bank_agent-phase_{phase}-n_agents={n_agents}-experiment_name={experiment_name}"
                    bank_network_path = os.path.join(networks_dir, f"{bank_network_name}_COMPLETE.pth")
                    
                    bank_network = load_network(
                        bank_network_path,
                        BankPolicy,
                        config,
                        input_size=4,
                        output_size=7 
                    )

            if mobile_network is not None:
                obs_stats = load_obs_stats(mobile_network_path)
                for agent in env.mobile_agents:
                    agent.policy_net = mobile_network
                    agent.obs_stats = obs_stats
                    random_sampling = False

            # Force enable planner and bank with zero policies to match training behavior
            if not env.has_planner:
                from training.vectorized_env import ZeroTaxPlannerWrapper
                env.planner = ZeroTaxPlannerWrapper(env.planner)
                env.has_planner = True
                print("Force-enabled planner with zero tax policy (matching training)")
            elif planner_network is not None and env.has_planner:
                env.planner.policy_net = planner_network
                planner_obs_stats = load_obs_stats(planner_network_path)
                env.planner.obs_stats = planner_obs_stats
            elif phase < 2 and env.has_planner:
                env.planner = ZeroTaxPlannerWrapper(env.planner)
                env.has_planner = True
                print("Using zero tax policy for planner")
            elif random_sampling and env.has_planner:
                env.planner.policy_net = None
                print("Using random actions for planner")
            
            if not env.has_bank:
                from training.vectorized_env import ZeroBankPolicyWrapper
                env.bank = ZeroBankPolicyWrapper(env.bank)
                env.has_bank = True
                print("Force-enabled bank with zero policy (matching training)")
            elif bank_network is not None and env.has_bank:
                env.bank.policy_net = bank_network
            elif phase < 3 and env.has_bank:
                env.bank = ZeroBankPolicyWrapper(env.bank)
                env.has_bank = True
                print("Using zero policy for bank")
            elif random_sampling and env.has_bank:
                env.bank.policy_net = None
                print("Using random actions for bank")

            env.bank.min_interest_rate_limit = env.bank.final_min_interest_rate
            env.bank.max_interest_rate_limit = env.bank.final_max_interest_rate
            env.bank.min_monetary_injection_limit = env.bank.final_min_monetary_injection
            env.bank.max_monetary_injection_limit = env.bank.final_max_monetary_injection

            print(f"Running experiment for {episode_length} steps...")
            while env.time < episode_length:
                if env.time % 200 == 0:
                    print(f"Step {env.time}/{episode_length}")
                env.step(random_sampling=random_sampling)
            
            all_environments.append(env)
            
            run_payouts = []
            run_houses = []
            
            print(f"\nResults for run {run+1}:")
            print(f"{'Agent ID':<10} {'Build Payout':<15} {'Houses Built':<15}")
            print("-" * 40)
            
            for agent in env.mobile_agents:
                build_payout = agent.original_build_payout
                houses_built = agent.houses_built
                
                run_payouts.append(build_payout)
                run_houses.append(houses_built)
                
                print(f"{agent.agent_id:<10} {build_payout:<15.2f} {houses_built:<15}")
            
            all_build_payouts.extend(run_payouts)
            all_houses_built.extend(run_houses)
            
            if len(run_payouts) > 1:
                run_r_squared = calculate_r_squared(run_payouts, run_houses)
                print(f"\nR² for run {run+1}: {run_r_squared:.4f}")

            if run % 5 == 0:
                plot_data(env, plot_folder, all_environments)
            
        plot_data(all_environments[-1], plot_folder, all_environments)
        
        overall_r_squared = calculate_r_squared(all_build_payouts, all_houses_built)
        print("\n" + "=" * 50)
        print(f"Overall R² across all {num_collections} runs: {overall_r_squared:.4f}")
        print("=" * 50)
        
        print("\nAggregate data by build payout:")
        
        payout_to_houses = {}
        for payout, houses in zip(all_build_payouts, all_houses_built):
            if payout not in payout_to_houses:
                payout_to_houses[payout] = []
            payout_to_houses[payout].append(houses)
        
        print(f"{'Build Payout':<15} {'Avg Houses Built':<20} {'Min':<10} {'Max':<10} {'Std Dev':<10} {'Count':<10}")
        print("-" * 75)
        
        for payout, houses_list in sorted(payout_to_houses.items()):
            avg_houses = np.mean(houses_list)
            min_houses = np.min(houses_list)
            max_houses = np.max(houses_list)
            std_houses = np.std(houses_list)
            count = len(houses_list)
            
            print(f"{payout:<15.2f} {avg_houses:<20.2f} {min_houses:<10.2f} {max_houses:<10.2f} {std_houses:<10.2f} {count:<10}")
        
    else:
        env = EconomyEnv(config)

        n_agents = config.get("n_agents", 4)
        experiment_name = config.get("experiment_name", "default_experiment")
        networks_dir = config.get("network_folder", "networks")

        mobile_network = None
        planner_network = None
        bank_network = None
        random_sampling = True

        if try_load:
            mobile_network_name = f"mobile_agents-phase_{phase}-n_agents={n_agents}-experiment_name={experiment_name}"
            mobile_network_path = os.path.join(networks_dir, f"{mobile_network_name}_COMPLETE.pth")
            
            basic_numeric_size = 7 + n_agents  # position(2) + inventory(3) + build_payout(1) + time_to_tax(1) + incomes(n_agents)
            planner_features_size = config.get("n_tax_brackets", 7) + 1  # tax_rates (n_brackets) + tax_bracket
            bank_features_size = 4  # inflation_rate, interest_rate, money_supply

            mobile_network = load_network(
                mobile_network_path,
                MobileAgentPolicy,
                config,
                num_numeric=basic_numeric_size + planner_features_size + bank_features_size,
                action_range=env.mobile_agents[0].action_range
            )
            
            if mobile_network is not None:
                random_sampling = False

            if phase >= 2 and env.has_planner:
                planner_network_name = f"planner_agent-phase_{phase}-n_agents={n_agents}-experiment_name={experiment_name}"
                planner_network_path = os.path.join(networks_dir, f"{planner_network_name}_COMPLETE.pth")
                
                n_tax_brackets = config.get("n_tax_brackets", 7)
                observation_size = 8 + (3 * n_agents + 2) + (n_tax_brackets + n_agents + 1) + 2
                
                planner_network = load_network(
                    planner_network_path,
                    PlannerPolicy,
                    config,
                    input_size=observation_size,
                    output_size=n_tax_brackets
                )

            if phase == 3 and env.has_bank:
                bank_network_name = f"bank_agent-phase_{phase}-n_agents={n_agents}-experiment_name={experiment_name}"
                bank_network_path = os.path.join(networks_dir, f"{bank_network_name}_COMPLETE.pth")
                
                bank_network = load_network(
                    bank_network_path,
                    BankPolicy,
                    config,
                    input_size=4, 
                    output_size=7
                )

        if mobile_network is not None:
            obs_stats = load_obs_stats(mobile_network_path)
            for agent in env.mobile_agents:
                agent.policy_net = mobile_network
                agent.obs_stats = obs_stats
                random_sampling = False

        # Force enable planner and bank with zero policies to match training behavior
        if not env.has_planner:
            from training.vectorized_env import ZeroTaxPlannerWrapper
            env.planner = ZeroTaxPlannerWrapper(env.planner)
            env.has_planner = True
            print("Force-enabled planner with zero tax policy (matching training)")
        elif planner_network is not None and env.has_planner:
            env.planner.policy_net = planner_network
            planner_obs_stats = load_obs_stats(planner_network_path)
            env.planner.obs_stats = planner_obs_stats
        elif phase < 2 and env.has_planner:
            env.planner = ZeroTaxPlannerWrapper(env.planner)
            env.has_planner = True
            print("Using zero tax policy for planner")
        elif random_sampling and env.has_planner:
            env.planner.policy_net = None
            print("Using random actions for planner")
        
        if not env.has_bank:
            from training.vectorized_env import ZeroBankPolicyWrapper
            env.bank = ZeroBankPolicyWrapper(env.bank)
            env.has_bank = True
            print("Force-enabled bank with zero policy (matching training)")
        elif bank_network is not None and env.has_bank:
            env.bank.policy_net = bank_network
        elif phase < 3 and env.has_bank:
            env.bank = ZeroBankPolicyWrapper(env.bank)
            env.has_bank = True
            print("Using zero policy for bank")
        elif random_sampling and env.has_bank:
            env.bank.policy_net = None
            print("Using random actions for bank")

        env.bank.min_interest_rate_limit = env.bank.final_min_interest_rate
        env.bank.max_interest_rate_limit = env.bank.final_max_interest_rate
        env.bank.min_monetary_injection_limit = env.bank.final_min_monetary_injection
        env.bank.max_monetary_injection_limit = env.bank.final_max_monetary_injection

        print(f"Running evaluation for {env.episode_length} steps...")

        
        while env.time < episode_length:
            if env.time % 100 == 0:
                print(f"Step {env.time}/{episode_length}")
            
            env.step(random_sampling=random_sampling)

        
        plot_data(env, plot_folder)
        print("Done! Plots saved in eval_plots directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--try-load", action="store_true", default=False, help="Try to load in the networks")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True, help="Which phase of training to evaluate")
    parser.add_argument("--ep-len", type=int, default=1000, help="Number of steps to run")
    parser.add_argument("--plot-folder", type=str, default="eval_plots", help="Folder to save plots")
    parser.add_argument("--get-collections", type=int, default=0, help="Number of collections to run (0 for standard single run)")
    args = parser.parse_args()

    get_collection_houses = args.get_collections > 0
    main(args.config, args.try_load, args.phase, args.ep_len, args.plot_folder, get_collection_houses, args.get_collections)