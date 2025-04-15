import json
import os
import matplotlib.pyplot as plt
import numpy as np

class Logger:
    def __init__(self, env):
        self.per_timestep_agent_data = {}
        self.per_timestep_env_data = {}
        self.per_timestep_planner_data = {}
        self.per_timestep_bank_data = {}
        self.env = env

    def log_agent_data_timestep(self, agent, action, timestep, action_success):
        agent_id = agent.agent_id

        if timestep not in self.per_timestep_agent_data:
            self.per_timestep_agent_data[timestep] = {}

        self.per_timestep_agent_data[timestep][agent_id] = {
            "coins": agent.inventory["coins"],
            "stone": agent.inventory["stone"],
            "wood": agent.inventory["wood"],
            "position": agent.position,
            "action": action,
            "labour": agent.labour,
            "escrow": agent.escrow,
            "action_success": action_success,
            "houses_built": agent.houses_built,
            "utility": agent.get_utility(),
            "escrow_coins": agent.escrow["coins"],
            "escrow_stone": agent.escrow["stone"],
            "escrow_wood": agent.escrow["wood"],
            "moves_made": agent.moves_made,
            "no_ops_made": agent.no_ops_made,
            "unsuccessful_moves": agent.unsuccessful_moves,
            "build_payout": agent.build_payout
        }
    
    def log_env_data_timestep(self, timestep):
        self.per_timestep_env_data[timestep] = {}

        """
        We want to track:
        1. Number of active orders (wood and stone)
        2. Number of wood bids, number of stone bids, same for asks
        3. Total number of trades over time (per resource)
        4. Average price of the resource over time (each time its traded, update the last price)
        """

        # Number of active orders
        num_wood_orders = len(self.env.trading_system.bids["wood"]) + len(self.env.trading_system.asks["wood"])
        num_stone_orders = len(self.env.trading_system.bids["stone"]) + len(self.env.trading_system.asks["stone"])

        # get average utility of all agents
        average_utility = sum([agent.get_utility() for agent in self.env.mobile_agents]) / len(self.env.mobile_agents)

        # get the number of house eligible tiles that dont have a house on them
        num_house_eligible_tiles = np.sum(self.env.map["Buildable"] == 1)

        self.per_timestep_env_data[timestep] = {
            "active_orders": num_wood_orders + num_stone_orders,
            "num_wood_bids": len(self.env.trading_system.bids["wood"]),
            "num_stone_bids": len(self.env.trading_system.bids["stone"]),
            "num_wood_asks": len(self.env.trading_system.asks["wood"]),
            "num_stone_asks": len(self.env.trading_system.asks["stone"]),
            "num_trades_wood": self.env.trading_system.num_trades["wood"],
            "num_trades_stone": self.env.trading_system.num_trades["stone"],
            "num_trades": self.env.trading_system.num_trades["wood"] + self.env.trading_system.num_trades["stone"],
            "last_price_wood": self.env.trading_system.last_price["wood"],
            "last_price_stone": self.env.trading_system.last_price["stone"],
            "average_utility": average_utility,
            "build_eligible_tiles": int(num_house_eligible_tiles),
            "num_houses_built": sum([agent.houses_built for agent in self.env.mobile_agents])
        }        

    def log_planner_data_timestep(self, planner, timestep):
        self.per_timestep_planner_data[timestep] = {}

        # get the tax rates
        tax_rates = planner.tax_rates
        previous_year_incomes = planner.previous_year_incomes
        tax_collected = planner.tax_collected
        utility = planner.get_utility()

        self.per_timestep_planner_data[timestep] = {
            "tax_rates": tax_rates, # array of length n_tax_brackets
            "previous_year_incomes": previous_year_incomes, # dict of agent_id to income
            "tax_collected": tax_collected, # int
            "utility": utility # float
        }

    def log_bank_data_timestep(self, bank, timestep):
        self.per_timestep_bank_data[timestep] = {}

        self.per_timestep_bank_data[timestep] = {
            "interest_rate": float(bank.interest_rate),
            "inflation_rate": float(bank.inflation_rate),
            "money_supply": float(bank.money_supply),
            "utility": float(bank.utility),
            "monetary_inections": float(bank.monetary_injections)
        }

        
    def save_data(self, dict, folder, filename):
        os.makedirs(folder, exist_ok=True)
        # save the data to a json file  
        with open(os.path.join(folder, filename), "w") as f:
            json.dump(dict, f, indent=4, sort_keys=True)

    def plot_agent_data(self, dict, folder, filename):
        os.makedirs(folder, exist_ok=True)
        agents = self.env.mobile_agents
        
        # Skip plotting if there's no data
        if not dict:
            print("No data to plot")
            return
        
        # Define which metrics should be line charts vs pie charts
        line_chart_metrics_agent = [
            "coins", "stone", 
            "wood", "labour", 
            "houses_built", "utility", 
            "escrow_coins", "escrow_stone", 
            "escrow_wood", "moves_made",
            "no_ops_made", "unsuccessful_moves",
            "build_payout"
        ]
        pie_chart_metrics = ["action", "action_success"]
        
        # Create line charts for specified metrics
        for metric in line_chart_metrics_agent:
            plt.figure(figsize=(10, 6))
            
            agent_ids = set()
            for timestep in dict:
                agent_ids.update(dict[timestep].keys())
            
            for agent_id in agent_ids:
                timesteps = []
                values = []
                
                # Collect data points across timesteps
                for timestep in sorted(dict.keys(), key=int):
                    if agent_id in dict[timestep] and metric in dict[timestep][agent_id]:
                        value = dict[timestep][agent_id][metric]
                        if isinstance(value, (int, float)): 
                            timesteps.append(int(timestep))
                            values.append(value)
                
                # Plot this agent's line
                if timesteps and values:
                    if agents is not None:
                        agent = next((a for a in agents if a.agent_id == agent_id), None)
                        if agent and hasattr(agent, 'build_payout'):
                            label = f"Agent {agent_id} (Skill: {agent.build_payout})"
                        else:
                            label = f"Agent {agent_id}"
                    else:
                        label = f"Agent {agent_id}"
                        
                    plt.plot(timesteps, values, linestyle='-', label=label)
            
            # Add plot details
            plt.title(f"{metric.capitalize()} over Time")
            plt.xlabel("Timestep")
            plt.ylabel(metric.capitalize())
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Save the plot
            plot_filename = f"{filename.split('.')[0]}_{metric}.png"
            plt.savefig(os.path.join(folder, plot_filename))
            plt.close()
        
        # Create pie charts for action and action_success
        for metric in pie_chart_metrics:
            value_counts = {}
            
            # Count occurrences across all timesteps and agents
            for timestep in dict:
                for agent_id in dict[timestep]:
                    if metric in dict[timestep][agent_id]:
                        value = str(dict[timestep][agent_id][metric])
                        if value not in value_counts:
                            value_counts[value] = 0
                        value_counts[value] += 1
            
            if value_counts:
                plt.figure(figsize=(10, 8))
                
                # Create pie chart
                labels = list(value_counts.keys())
                sizes = list(value_counts.values())
                
                max_index = sizes.index(max(sizes))
                explode = [0.1 if i == max_index else 0 for i in range(len(sizes))]
                
                plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                       shadow=True, startangle=90)
                plt.axis('equal')  
                plt.title(f"Distribution of {metric.replace('_', ' ').capitalize()}")
                
                plot_filename = f"{filename.split('.')[0]}_{metric}_pie.png"
                plt.savefig(os.path.join(folder, plot_filename))
                plt.close()

    def plot_env_data(self, dict_or_list, folder, filename):
        os.makedirs(folder, exist_ok=True)
        
        if isinstance(dict_or_list, list):
            dict_agg = {}
            for timestep in dict_or_list[0].keys():
                dict_agg[timestep] = {}
                for metric in dict_or_list[0][timestep].keys():
                    values = [d[timestep][metric] for d in dict_or_list if timestep in d and metric in d[timestep]]
                    if values:
                        dict_agg[timestep][metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values)
                        }
            dict = dict_agg
            show_std = True
        else:
            dict = dict_or_list
            show_std = False
        
        # Skip plotting if there's no data
        if not dict:
            print("No data to plot")
            return
        
        plot_configs = [
            {
                "title": "Active Orders Over Time",
                "metrics": ["active_orders"],
                "ylabel": "Number of Orders",
                "allow_negative": False
            },
            {
                "title": "Number of Bids Over Time",
                "metrics": ["num_wood_bids", "num_stone_bids"],
                "ylabel": "Number of Bids",
                "allow_negative": False
            },
            {
                "title": "Number of Asks Over Time",
                "metrics": ["num_wood_asks", "num_stone_asks"],
                "ylabel": "Number of Asks",
                "allow_negative": False
            },
            {
                "title": "Number of Trades Over Time",
                "metrics": ["num_trades_wood", "num_trades_stone", "num_trades"],
                "ylabel": "Number of Trades",
                "allow_negative": False
            },
            {
                "title": "Wood Price Over Time",
                "metrics": ["last_price_wood"],
                "ylabel": "Price",
                "use_rolling_avg": True,
                "allow_negative": False
            },
            {
                "title": "Stone Price Over Time",
                "metrics": ["last_price_stone"],
                "ylabel": "Price",
                "use_rolling_avg": True,
                "allow_negative": False
            },
            {
                "title": "Average Utility Over Time",
                "metrics": ["average_utility"],
                "ylabel": "Utility",
                "allow_negative": True
            },
            {
                "title": "Number of Buildable Tiles Over Time",
                "metrics": ["build_eligible_tiles"],
                "ylabel": "Number of Buildable Tiles",
                "allow_negative": False
            },
            {
                "title": "Number of Houses Built Over Time",
                "metrics": ["num_houses_built"],
                "ylabel": "Number of Houses Built",
                "allow_negative": False
            }
        ]
        
        for config in plot_configs:
            plt.figure(figsize=(10, 6))
            allow_negative = config.get("allow_negative", True)
            
            for metric in config["metrics"]:
                timesteps = []
                values = []
                stds = []
                
                for timestep in sorted(dict.keys(), key=int):
                    if metric in dict[timestep] or (show_std and metric in dict[timestep]):
                        timesteps.append(int(timestep))
                        if show_std:
                            values.append(dict[timestep][metric]['mean'])
                            stds.append(dict[timestep][metric]['std'])
                        else:
                            values.append(dict[timestep][metric])
                
                if config.get("use_rolling_avg", False) and timesteps and values:
                    label = metric.replace("_", " ").replace("num ", "").title()
                    
                    window_size = self.env.config["tax_period_length"]
                    
                    if len(values) >= window_size:
                        rolling_avg = []
                        rolling_std = []
                        
                        for i in range(len(values)):
                            window_start = max(0, i - window_size + 1)
                            window = values[window_start:i+1]
                            rolling_avg.append(np.mean(window))
                            if show_std:
                                window_std = stds[window_start:i+1]
                                rolling_std.append(np.mean(window_std))
                            else:
                                rolling_std.append(np.std(window))
                        
                        plt.plot(timesteps, rolling_avg, linestyle='-', color='blue', 
                                label=f'{label} (Rolling Avg)')
                        
                        lower_bound = [avg - std for avg, std in zip(rolling_avg, rolling_std)]
                        upper_bound = [avg + std for avg, std in zip(rolling_avg, rolling_std)]
                        
                        if not allow_negative:
                            lower_bound = [max(0, lb) for lb in lower_bound]
                        
                        plt.fill_between(timesteps, lower_bound, upper_bound,
                                        color='blue', alpha=0.2, label='±1 std dev')
                    else:
                        plt.plot(timesteps, values, linestyle='-', label=label)
                        if show_std and len(stds) > 0:
                            lower_bound = [v - s for v, s in zip(values, stds)]
                            upper_bound = [v + s for v, s in zip(values, stds)]
                            
                            if not allow_negative:
                                lower_bound = [max(0, lb) for lb in lower_bound]
                            
                            plt.fill_between(timesteps, lower_bound, upper_bound, alpha=0.2)
                
                elif not config.get("use_rolling_avg", False) and timesteps and values:
                    label = metric.replace("_", " ").replace("num ", "").title()
                    plt.plot(timesteps, values, linestyle='-', label=label)
                    
                    if show_std and len(stds) > 0:
                        lower_bound = [v - s for v, s in zip(values, stds)]
                        upper_bound = [v + s for v, s in zip(values, stds)]
                        
                        if not allow_negative:
                            lower_bound = [max(0, lb) for lb in lower_bound]
                        
                        plt.fill_between(timesteps, lower_bound, upper_bound, alpha=0.2)
            
            plt.title(config["title"])
            plt.xlabel("Timestep")
            plt.ylabel(config["ylabel"])
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            metrics_str = "_".join([m.split("_")[-1] for m in config["metrics"]])
            plot_filename = f"{filename.split('.')[0]}_{metrics_str}.png"
            plt.savefig(os.path.join(folder, plot_filename))
            plt.close()

    def plot_planner_data(self, dict_or_list, folder, filename):
        os.makedirs(folder, exist_ok=True)

        if isinstance(dict_or_list, list):
            dict_agg = {}
            for timestep in set().union(*[d.keys() for d in dict_or_list if d]):
                dict_agg[timestep] = {}
                
                tax_rates_list = [d[timestep].get("tax_rates", []) for d in dict_or_list if timestep in d and "tax_rates" in d[timestep]]
                if tax_rates_list:
                    max_len = max(len(rates) for rates in tax_rates_list)
                    
                    padded_rates = []
                    for rates in tax_rates_list:
                        if isinstance(rates, np.ndarray):
                            rates_list = rates.tolist()
                        else:
                            rates_list = list(rates)
                        padded_rates.append(rates_list + [0] * (max_len - len(rates_list)))
                    
                    tax_rates_mean = []
                    tax_rates_std = []
                    for i in range(max_len):
                        bracket_rates = [rates[i] for rates in padded_rates if i < len(rates)]
                        if bracket_rates:
                            tax_rates_mean.append(np.mean(bracket_rates))
                            tax_rates_std.append(np.std(bracket_rates))
                    
                    dict_agg[timestep]["tax_rates"] = {
                        'mean': tax_rates_mean,
                        'std': tax_rates_std
                    }
                
                for metric in ["tax_collected", "utility"]:
                    values = [d[timestep].get(metric, 0) for d in dict_or_list if timestep in d and metric in d[timestep]]
                    if values:
                        dict_agg[timestep][metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values)
                        }
                
                income_dicts = [d[timestep].get("previous_year_incomes", {}) for d in dict_or_list 
                              if timestep in d and "previous_year_incomes" in d[timestep]]
                if income_dicts:
                    all_agent_ids = set()
                    for income_dict in income_dicts:
                        if isinstance(income_dict, dict):
                            all_agent_ids.update(income_dict.keys())
                    
                    if all_agent_ids:
                        agent_income_means = {}
                        agent_income_stds = {}
                        
                        for agent_id in all_agent_ids:
                            agent_incomes = [income_dict.get(agent_id, 0) for income_dict in income_dicts 
                                          if isinstance(income_dict, dict) and agent_id in income_dict]
                            if agent_incomes:
                                agent_income_means[agent_id] = np.mean(agent_incomes)
                                agent_income_stds[agent_id] = np.std(agent_incomes)
                        
                        dict_agg[timestep]["previous_year_incomes"] = {
                            'means': agent_income_means,
                            'stds': agent_income_stds
                        }
            
            dictionary = dict_agg
            show_std = True
        else:
            dictionary = dict_or_list
            show_std = False

        if not dictionary:
            print("No data to plot for planner")
            return
        
        thresholds = ["0"] + [str(t) for t in self.env.planner.tax_brackets]
        n_brackets = self.env.planner.n_tax_brackets
        
        n_rows = (n_brackets + 2) // 3  
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        bracket_data = []
        for i in range(n_brackets):
            timesteps = []
            rates = []
            stds = []
            
            for timestep in sorted(dictionary.keys(), key=int):
                if "tax_rates" in dictionary[timestep]:
                    if show_std:
                        tax_rates = dictionary[timestep]["tax_rates"]["mean"]
                        tax_rates_std = dictionary[timestep]["tax_rates"]["std"]
                        if i < len(tax_rates):
                            timesteps.append(int(timestep))
                            rates.append(tax_rates[i])
                            stds.append(tax_rates_std[i])
                    else:
                        tax_rates = dictionary[timestep]["tax_rates"]
                        if i < len(tax_rates):
                            timesteps.append(int(timestep))
                            rates.append(tax_rates[i])
            
            if i == 0:
                bracket_label = f"0-{thresholds[1]}"
            elif i == len(thresholds) - 1:
                bracket_label = f"{thresholds[i]}+"
            else:
                bracket_label = f"{thresholds[i]}-{thresholds[i+1]}"
            
            bracket_data.append({
                "timesteps": timesteps,
                "rates": rates,
                "stds": stds if show_std else None,
                "label": bracket_label
            })
        
        for i, data in enumerate(bracket_data):
            ax = plt.subplot(n_rows, 3, i + 1)
            
            if data["timesteps"] and data["rates"]:
                ax.plot(data["timesteps"], data["rates"], 'b-')
                
                if show_std and data["stds"]:
                    ax.fill_between(
                        data["timesteps"], 
                        [max(0, r - s) for r, s in zip(data["rates"], data["stds"])],
                        [min(1, r + s) for r, s in zip(data["rates"], data["stds"])],
                        color='blue', alpha=0.2
                    )
                
                ax.set_title(f"Bracket {data['label']}")
                ax.set_xlabel("Timestep")
                ax.set_ylabel("Tax Rate")
                ax.set_ylim(0, 1) 
                ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plot_filename = f"{filename.split('.')[0]}_tax_rates_individual.png"
        plt.savefig(os.path.join(folder, plot_filename))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        
        agent_ids = set()
        for timestep in dictionary:
            if "previous_year_incomes" in dictionary[timestep]:
                if show_std:
                    if isinstance(dictionary[timestep]["previous_year_incomes"]["means"], dict):
                        agent_ids.update(dictionary[timestep]["previous_year_incomes"]["means"].keys())
                elif isinstance(dictionary[timestep]["previous_year_incomes"], dict):
                    agent_ids.update(dictionary[timestep]["previous_year_incomes"].keys())
        
        for agent_id in agent_ids:
            timesteps = []
            incomes = []
            stds = []
            
            for timestep in sorted(dictionary.keys(), key=int):
                if "previous_year_incomes" in dictionary[timestep]:
                    if show_std:
                        previous_year_incomes = dictionary[timestep]["previous_year_incomes"]["means"]
                        previous_year_stds = dictionary[timestep]["previous_year_incomes"]["stds"]
                        if isinstance(previous_year_incomes, dict) and agent_id in previous_year_incomes:
                            timesteps.append(int(timestep))
                            incomes.append(previous_year_incomes[agent_id])
                            stds.append(previous_year_stds[agent_id])
                    else:
                        previous_year_incomes = dictionary[timestep]["previous_year_incomes"]
                        if isinstance(previous_year_incomes, dict) and agent_id in previous_year_incomes:
                            timesteps.append(int(timestep))
                            incomes.append(previous_year_incomes[agent_id])
            
            if timesteps and incomes:
                plt.plot(timesteps, incomes, linestyle='-', label=f"Agent {agent_id}")
                
                if show_std and stds:
                    plt.fill_between(
                        timesteps, 
                        [i - s for i, s in zip(incomes, stds)],
                        [i + s for i, s in zip(incomes, stds)],
                        alpha=0.2
                    )
        
        if not agent_ids:
            timesteps = []
            all_incomes = []
            all_stds = []
            
            for timestep in sorted(dictionary.keys(), key=int):
                if "previous_year_incomes" in dictionary[timestep]:
                    if show_std:
                        # This case should be handled differently for aggregated data
                        pass
                    else:
                        incomes = dictionary[timestep]["previous_year_incomes"]
                        if isinstance(incomes, list):
                            timesteps.append(int(timestep))
                            all_incomes.append(sum(incomes) / len(incomes) if incomes else 0)
            
            if timesteps and all_incomes:
                plt.plot(timesteps, all_incomes, linestyle='-', label="Average Income")
        
        plt.title("Previous Year Incomes Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Income")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plot_filename = f"{filename.split('.')[0]}_previous_year_incomes.png"
        plt.savefig(os.path.join(folder, plot_filename))
        plt.close()
        
        # Plot tax collected over time
        plt.figure(figsize=(10, 6))
        timesteps = []
        tax_collected = []
        tax_collected_stds = []
        
        for timestep in sorted(dictionary.keys(), key=int):
            if "tax_collected" in dictionary[timestep]:
                timesteps.append(int(timestep))
                if show_std:
                    tax_collected.append(dictionary[timestep]["tax_collected"]["mean"])
                    tax_collected_stds.append(dictionary[timestep]["tax_collected"]["std"])
                else:
                    tax_collected.append(dictionary[timestep]["tax_collected"])
        
        if timesteps and tax_collected:
            plt.plot(timesteps, tax_collected, linestyle='-', label="Tax Collected")
            
            if show_std and tax_collected_stds:
                plt.fill_between(
                    timesteps, 
                    [max(0, t - s) for t, s in zip(tax_collected, tax_collected_stds)],
                    [t + s for t, s in zip(tax_collected, tax_collected_stds)],
                    alpha=0.2
                )
        
        plt.title("Tax Collected Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Amount")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plot_filename = f"{filename.split('.')[0]}_tax_collected.png"
        plt.savefig(os.path.join(folder, plot_filename))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        timesteps = []
        utilities = []
        utility_stds = []
        
        for timestep in sorted(dictionary.keys(), key=int):
            if "utility" in dictionary[timestep]:
                timesteps.append(int(timestep))
                if show_std:
                    utilities.append(dictionary[timestep]["utility"]["mean"])
                    utility_stds.append(dictionary[timestep]["utility"]["std"])
                else:
                    utilities.append(dictionary[timestep]["utility"])
        
        if timesteps and utilities:
            plt.plot(timesteps, utilities, linestyle='-', label="Planner Utility")
            
            if show_std and utility_stds:
                plt.fill_between(
                    timesteps, 
                    [u - s for u, s in zip(utilities, utility_stds)],
                    [u + s for u, s in zip(utilities, utility_stds)],
                    alpha=0.2
                )
        
        plt.title("Planner Utility Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Utility")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plot_filename = f"{filename.split('.')[0]}_utility.png"
        plt.savefig(os.path.join(folder, plot_filename))
        plt.close()

    def plot_bank_data(self, dict_or_list, folder, filename):
        os.makedirs(folder, exist_ok=True)
        
        if isinstance(dict_or_list, list):
            dict_agg = {}
            for timestep in set().union(*[d.keys() for d in dict_or_list if d]):
                dict_agg[timestep] = {}
                for metric in ["interest_rate", "inflation_rate", "money_supply", "utility", "monetary_inections"]:
                    values = [d[timestep].get(metric, 0) for d in dict_or_list 
                            if timestep in d and metric in d[timestep]]
                    if values:
                        dict_agg[timestep][metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values)
                        }
            
            dict = dict_agg
            show_std = True
        else:
            dict = dict_or_list
            show_std = False


        for metric in next(iter(dict.values())).keys() if dict else []:
            plt.figure(figsize=(10, 6))
            timesteps = []
            values = []
            stds = []
            allow_negative = True

            for timestep in sorted(dict.keys(), key=int):
                if metric in dict[timestep]:
                    timesteps.append(int(timestep))
                    if show_std:
                        values.append(dict[timestep][metric]['mean'])
                        stds.append(dict[timestep][metric]['std'])
                    else:
                        values.append(dict[timestep][metric])

            if metric == "inflation_rate":
                print(f"Average inflation rate: {np.mean(values)}")
                print(f"Max inflation rate: {np.max(values)}")
                if hasattr(self.env, 'config') and 'target_inflation' in self.env.config:
                    print(f"MSE from {self.env.config['target_inflation'] * 100}%: {(np.mean(values) - self.env.config['target_inflation']) ** 2}")
                if len(timesteps) >= 100:
                    last_timesteps = sorted(dict.keys(), key=int)[-1000:]
                    if show_std:
                        last_utilities = [dict[timestep]['utility']['mean'] for timestep in last_timesteps 
                                       if 'utility' in dict[timestep]]
                    else:
                        last_utilities = [dict[timestep]['utility'] for timestep in last_timesteps 
                                       if 'utility' in dict[timestep]]
                    if last_utilities:
                        print(f"Average utility of last 100 timesteps: {np.mean(last_utilities)}\n")
                
                if hasattr(self.env, 'config') and 'tax_period_length' in self.env.config:
                    tax_year_length = 10*self.env.config["tax_period_length"]
                    target_inflation = self.env.config.get("target_inflation", 0)
                    if len(values) >= tax_year_length:
                        rolling_avg = []
                        rolling_std = []
                        
                        for i in range(len(values)):
                            window_start = max(0, i - tax_year_length + 1)
                            window = values[window_start:i+1]
                            rolling_avg.append(np.mean(window))
                            if show_std:
                                window_std = stds[window_start:i+1]
                                rolling_std.append(np.mean(window_std))
                            else:
                                rolling_std.append(np.std(window))
                        
                        plt.plot(timesteps, rolling_avg, linestyle='-', color='blue', 
                                label=f'Rolling avg')
                        
                        lower_bound = [avg - std for avg, std in zip(rolling_avg, rolling_std)]
                        upper_bound = [avg + std for avg, std in zip(rolling_avg, rolling_std)]
                        
                        plt.fill_between(timesteps, lower_bound, upper_bound,
                                        color='blue', alpha=0.2, label='±1 std dev')
                        
                        plt.axhline(y=target_inflation, color='black', linestyle='--', label='Target Inflation')
                    else:
                        plt.plot(timesteps, values, linestyle='-', color='blue', label=metric)
                        if show_std:
                            lower_bound = [v - s for v, s in zip(values, stds)]
                            upper_bound = [v + s for v, s in zip(values, stds)]
                            
                            plt.fill_between(timesteps, lower_bound, upper_bound, alpha=0.2)
                else:
                    plt.plot(timesteps, values, linestyle='-', color='blue', label=metric)
                    if show_std:
                        lower_bound = [v - s for v, s in zip(values, stds)]
                        upper_bound = [v + s for v, s in zip(values, stds)]
                        
                        plt.fill_between(timesteps, lower_bound, upper_bound, alpha=0.2)
            else:
                plt.plot(timesteps, values, linestyle='-', color='blue', label=metric)
                if show_std:
                    lower_bound = [v - s for v, s in zip(values, stds)]
                    upper_bound = [v + s for v, s in zip(values, stds)]
                    
                    if not allow_negative:
                        lower_bound = [max(0, lb) for lb in lower_bound]
                    
                    plt.fill_between(timesteps, lower_bound, upper_bound, alpha=0.2)
            
            plt.title(f"{metric.replace('_', ' ').capitalize()} Over Time")
            plt.xlabel("Timestep")
            plt.ylabel(metric.replace('_', ' ').capitalize())
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            plot_filename = f"{filename.split('.')[0]}_{metric}.png"
            plt.savefig(os.path.join(folder, plot_filename))
            plt.close()
            
            
        
