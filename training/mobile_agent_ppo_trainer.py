import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from training.metrics_logger import MetricsLogger
import os
from tqdm import tqdm
import time
from datetime import timedelta
from training.mobile_agent_policy import MobileAgentPolicy
import random
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from training.gpu_worker import RemoteUpdater

class MultiAgentPPOTrainer:
    def __init__(self, vec_env, config, use_gpu=True):
        self.vec_env = vec_env
        self.config = config
        self.num_workers = self.config.get("n_envs", 4)

        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        self.remote_updates = config.get("remote_updates", False)

        self.mat_config = config["mobile_agent_training"]

        self.horizon = config.get("episode_length", 1000)
        self.num_updates = self.mat_config.get("num_updates", 1000)
        self.exploration_steps = min(self.mat_config.get("exploration_steps", 10), self.horizon)
        self.gamma = self.mat_config.get("gamma", 0.99)
        self.lam = self.mat_config.get("lambda", 0.95)
        self.clip_epsilon = self.mat_config.get("clip_epsilon", 0.2)
        self.ppo_epochs = self.mat_config.get("ppo_epochs", 4)
        self.mini_batch_size = self.mat_config.get("mini_batch_size", 64)
        self.learning_rate = self.mat_config.get("learning_rate", 3e-4)
        self.value_loss_weight = self.mat_config.get("value_loss_weight", 0.5)
        self.entropy_weight = self.mat_config.get("entropy_weight", 0.01)
        self.target_kl = self.mat_config.get("target_kl", 0.01)
        self.epsilon_explore = self.mat_config.get("epsilon_explore", 0.1)
        
        self.lstm_states = {}
        
        self.original_labour_values = {
            "move_labour": self.config.get("move_labour", 0.0),
            "gather_labour": self.config.get("gather_labour", 0.0),
            "trade_labour": self.config.get("trade_labour", 0.0),
            "build_labour": self.config.get("build_labour", 0.0),
            "no_op_labour": self.config.get("no_op_labour", 0.0),
        }
        
        basic_numeric_size = 7 + self.vec_env.n_agents  # position(2) + inventory(3) + build_payout(1) + time_to_tax(1) + incomes(n_agents)
        
        planner_features_size = self.config.get("n_tax_brackets", 7) + 1  # tax_rates (n_brackets) + tax_bracket
        bank_features_size = 4  # inflation_rate, interest_rate, money_supply
        
        if not self.config.get("planner", False):
            self.config["planner"] = True
            for env_idx in range(self.vec_env.total_envs):
                self.vec_env.pipes[env_idx].send(("wrap_planner_with_zero_tax", None))
                try:
                    self.vec_env.pipes[env_idx].recv()
                except (EOFError, BrokenPipeError):
                    continue
                    
        if not self.config.get("bank", False):
            self.config["bank"] = True
            for env_idx in range(self.vec_env.total_envs):
                self.vec_env.pipes[env_idx].send(("wrap_bank_with_zero_policy", None))
                try:
                    self.vec_env.pipes[env_idx].recv()
                except (EOFError, BrokenPipeError):
                    continue

        self.num_numeric = basic_numeric_size + planner_features_size + bank_features_size
        
        self.shared_policy = MobileAgentPolicy(
            config=self.config,
            num_numeric=self.num_numeric,
            action_range=self.vec_env.env_ref.mobile_agents[0].action_range
        )
        self.shared_policy.to(self.device)
        self.shared_optimizer = optim.Adam(self.shared_policy.parameters(), lr=self.learning_rate)

        self.rollout_buffers = {}

        self.network_folder = self.config.get("network_folder", "networks")
        self.network_name = f"mobile_agents-phase_1-n_agents={self.vec_env.n_agents}-" \
            f"experiment_name={self.config.get('experiment_name', 'default_experiment')}"
        
        os.makedirs(self.network_folder, exist_ok=True)
        
        self.reward_normalizer = {
            "running_mean": 0,
            "running_var": 1,
            "count": 0,
            "gamma": 0.99
        }
    
    def collect_rollouts(self, random_sampling=False):
        if random_sampling:
            print(f"Collecting rollouts with random sampling.")
        else:
            print(f"Collecting rollouts with network-based sampling.")  
        
        if not self.rollout_buffers:
            all_agents_data = self.vec_env.get_all_agent_observations()
            for data in all_agents_data:
                env_idx = data["env_idx"]
                agent_id = data["agent_id"]
                buffer_key = f"env{env_idx}_agent{agent_id}"
                self.rollout_buffers[buffer_key] = []
        
        if not hasattr(self, 'lstm_states'):
            self.lstm_states = {}
        
        for t in range(self.horizon):
            all_agents_data = self.vec_env.get_all_obs_and_masks()

            neigh_list, numeric_list = [], []
            buffer_keys, agent_ids, env_idxs = [], [], []
            lstm_states_batch = [[], []]
            action_masks = []

            for agent_data in all_agents_data:
                env_idx = agent_data["env_idx"]
                agent_id = agent_data["agent_id"]
                buffer_key = f"env{env_idx}_agent{agent_id}"
                obs = agent_data["obs"]

                neighbourhood, numeric = self.flatten_observation(obs)
                neigh_list.append(neighbourhood)
                numeric_list.append(numeric)
                buffer_keys.append(buffer_key)
                agent_ids.append(agent_id)
                env_idxs.append(env_idx)
                action_masks.append(agent_data["action_mask"])

                agent_lstm_state = self.lstm_states.get(buffer_key, None)

                if agent_lstm_state is not None:
                    h, c = agent_lstm_state
                    lstm_states_batch[0].append(h)
                    lstm_states_batch[1].append(c)

            neighbourhood_batch = torch.stack(neigh_list).to(self.device)
            numeric_batch = torch.stack(numeric_list).to(self.device)

            if lstm_states_batch[0]:
                h_batch = torch.cat(lstm_states_batch[0], dim=1)
                c_batch = torch.cat(lstm_states_batch[1], dim=1)
                lstm_state_batch = (h_batch, c_batch)
            else:
                lstm_state_batch = None

            action_masks_tensor = torch.tensor(np.array(action_masks), dtype=torch.bool).to(self.device)

            if random_sampling or random.random() < self.epsilon_explore:
                actions_batch = []
                log_probs_batch = []
                values_batch = torch.zeros(len(buffer_keys), 1).to(self.device)
                lstm_state_out = (torch.zeros_like(h_batch), torch.zeros_like(c_batch)) if lstm_state_batch else None

                for mask in action_masks:
                    valid_actions = np.where(mask)[0]
                    if len(valid_actions) > 0:
                        action = np.random.choice(valid_actions)
                        actions_batch.append(action)
                        log_prob = -np.log(len(valid_actions))
                        log_probs_batch.append(log_prob)
                    else:
                        actions_batch.append(0)
                        log_probs_batch.append(0.0)
                
                actions_batch = torch.tensor(actions_batch, device=self.device)
                log_probs_batch = torch.tensor(log_probs_batch, device=self.device)
            else:
                self.shared_policy.eval()
                with torch.no_grad():
                    logits_batch, values_batch, lstm_state_out = self.shared_policy(
                        neighbourhood_batch, numeric_batch, lstm_state_batch
                    )
                    
                    masked_logits_batch = logits_batch.clone()
                    masked_logits_batch[~action_masks_tensor] = -1e10
                    
                    probs_batch = F.softmax(masked_logits_batch, dim=-1)
                    dist_batch = torch.distributions.Categorical(probs_batch)
                    actions_batch = dist_batch.sample()
                    log_probs_batch = dist_batch.log_prob(actions_batch)

            if lstm_state_out is not None:
                h_out, c_out = lstm_state_out
                for i, buffer_key in enumerate(buffer_keys):
                    h_agent = h_out[:, i:i+1, :].detach()
                    c_agent = c_out[:, i:i+1, :].detach()
                    self.lstm_states[buffer_key] = (h_agent, c_agent)

            # Build actions grouped by env for batch stepping
            actions_by_env = {}
            for i in range(len(buffer_keys)):
                env_idx = env_idxs[i]
                agent_id = agent_ids[i]
                action = int(actions_batch[i].item())
                if env_idx not in actions_by_env:
                    actions_by_env[env_idx] = {}
                actions_by_env[env_idx][agent_id] = action

            # Step all agents and env in all envs in parallel (single round-trip)
            batch_results = self.vec_env.step_agents_and_env(actions_by_env)

            for i in range(len(buffer_keys)):
                env_idx = env_idxs[i]
                agent_id = agent_ids[i]
                buffer_key = buffer_keys[i]
                result = batch_results.get(env_idx, {}).get(agent_id, {})
                reward = result.get("reward", 0.0)

                self.rollout_buffers[buffer_key].append({
                    "neighbourhood": neighbourhood_batch[i].detach(),
                    "numeric": numeric_batch[i].detach(),
                    "action": actions_by_env[env_idx][agent_id],
                    "log_prob": log_probs_batch[i].detach(),
                    "value": values_batch[i].detach(),
                    "reward": reward,
                    "utility": result.get("utility", 0.0),
                    "original_reward": reward,
                    "action_mask": action_masks[i],
                })

    def normalize_reward(self, reward):
        # Simple clipping without the buggy normalization
        return np.clip(reward, -10.0, 10.0)
    
    def flatten_observation(self, observation):
        channels = [
            torch.tensor(observation["neighbourhood"][key], dtype=torch.float32)
            for key in ["Wood", "Stone", "Water", "Houses", "Agents"]
        ]
        neighbourhood_tensor = torch.stack(channels, dim=0)

        pos = np.array(observation["position"], dtype=np.float32)
        inv = np.array([observation["inventory"]["wood"],
                        observation["inventory"]["stone"],
                        observation["inventory"]["coins"]], dtype=np.float32)
        
        bp = np.array([observation["build_payout"]], dtype=np.float32)
        ttt = np.array([observation["time_to_tax"]], dtype=np.float32)
        incomes = np.array(observation["incomes"], dtype=np.float32)
        
        features = [pos, inv, bp, ttt, incomes]
        
        if "tax_rates" in observation and "tax_bracket" in observation:
            tax_rates = np.array(observation["tax_rates"], dtype=np.float32)
            tax_bracket = np.array([observation["tax_bracket"]], dtype=np.float32)
            features.extend([tax_rates, tax_bracket])

        if "inflation_rate" in observation and "interest_rate" in observation and "money_supply" in observation:
            inflation_rate = np.array([observation["inflation_rate"]], dtype=np.float32)
            interest_rate = np.array([observation["interest_rate"]], dtype=np.float32)
            money_supply = np.array([observation["money_supply"]], dtype=np.float32)
            target_inflation = np.array([observation["target_inflation"]], dtype=np.float32)
            features.extend([inflation_rate, interest_rate, money_supply, target_inflation])
        
        numeric = np.concatenate(features)
        numeric_tensor = torch.tensor(numeric, dtype=torch.float32)
        
        return neighbourhood_tensor, numeric_tensor
    
    def compute_gae(self, buffer):
        advantages = []
        returns = []
        gae = 0

        for t in reversed(range(len(buffer))):
            r = buffer[t]["reward"]
            v = buffer[t]["value"].item()
            
            if t == len(buffer) - 1:
                next_value = 0
            else:
                next_value = buffer[t + 1]["value"].item()
            
            delta = r + self.gamma * next_value - v
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + v)
        
        return advantages, returns

    def update_shared_policy(self, all_data, logger):
        neighbourhoods = torch.stack([t["neighbourhood"] for t in all_data]).to(self.device)
        numerics = torch.stack([t["numeric"] for t in all_data]).to(self.device)
        actions = torch.LongTensor([t["action"] for t in all_data]).to(self.device)
        old_log_probs = torch.stack([t["log_prob"] for t in all_data]).to(self.device)
        advantages = torch.FloatTensor([t["advantages"] for t in all_data]).to(self.device)
        returns = torch.FloatTensor([t["returns"] for t in all_data]).to(self.device)
        action_masks = torch.tensor(np.array([t["action_mask"] for t in all_data]), dtype=torch.bool).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(all_data)
        indices = list(range(dataset_size))

        total_policy_loss = 0
        total_value_loss = 0
        total_loss = 0
        batches = 0

        for epoch in range(self.ppo_epochs):
            random.shuffle(indices)            
            approx_kl_divs = []
            
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_idx = indices[start:end]
                batch_neigh = neighbourhoods[batch_idx]
                batch_numeric = numerics[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                self.shared_policy.train()
                logits, values, _ = self.shared_policy(batch_neigh, batch_numeric, None)
                values = values.squeeze(1)
                
                batch_action_masks = action_masks[batch_idx]
                
                masked_logits = logits.clone()
                masked_logits[~batch_action_masks] = -1e10
                
                probs = F.softmax(masked_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                batch_returns_norm = (batch_returns - batch_returns.mean()) / (batch_returns.std() + 1e-8)
                values_norm = (values - values.mean()) / (values.std() + 1e-8)
                
                value_loss = F.mse_loss(values_norm, batch_returns_norm)
                
                loss = policy_loss + self.value_loss_weight * value_loss - self.entropy_weight * entropy

                self.shared_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping as per paper
                clip_norm = self.config.get("mobile_agent_training", {}).get("gradient_clip_norm", 10.0)
                torch.nn.utils.clip_grad_norm_(self.shared_policy.parameters(), clip_norm)
                
                # Check for NaN gradients
                for param in self.shared_policy.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print("Warning: NaN gradient detected, skipping update")
                        continue
                
                self.shared_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_loss += loss.item()
                batches += 1

                log_ratio = new_log_probs - batch_old_log_probs
                approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                approx_kl_divs.append(approx_kl_div)
                
                # # Early stopping if KL divergence gets too high (policy changing too much)
                # if approx_kl_div > self.target_kl:
                #     print(f"Early stopping PPO epoch {epoch} due to high KL divergence: {approx_kl_div:.4f}")
                #     break
            
            avg_kl = sum(approx_kl_divs) / len(approx_kl_divs)
            if avg_kl > self.target_kl:
                break

        avg_policy_loss = total_policy_loss / batches
        avg_value_loss = total_value_loss / batches
        avg_total_loss = total_loss / batches

        logger.log("policy_loss", avg_policy_loss)
        logger.log("value_loss", avg_value_loss)
        logger.log("total_loss", avg_total_loss)

    def update_agents(self, logger):
        all_data = []
        utilities = []

        for buffer_key, buffer in self.rollout_buffers.items():
            if not buffer:
                continue
                
            advantages, returns = self.compute_gae(buffer)

            for i, entry in enumerate(buffer):
                entry["advantages"] = advantages[i]
                entry["returns"] = returns[i]
                
            all_data.extend(buffer)

            avg_reward = np.mean([t["reward"] for t in buffer])
            logger.log(f"{buffer_key}_reward", avg_reward)
            
            avg_utility = np.mean([t["utility"] for t in buffer])
            logger.log(f"{buffer_key}_utility", avg_utility)
            utilities.append(avg_utility)
            
            self.rollout_buffers[buffer_key] = []

        if all_data:
            avg_reward = np.mean([t["reward"] for t in all_data])
            logger.log("reward", avg_reward)
            
            if utilities:
                avg_utility = np.mean(utilities)
                logger.log("utility", avg_utility)
                print(f"Average utility: {avg_utility:.2f}")
        
        if self.remote_updates:
            metrics = rpc.rpc_sync(self.remote_updater_rref, RemoteUpdater.update_shared_policy, args=(all_data,))
            logger.log("policy_loss", metrics["policy_loss"])
            logger.log("value_loss", metrics["value_loss"])
            logger.log("total_loss", metrics["total_loss"])
            print(f"Policy loss: {metrics['policy_loss']:.4f}, Value loss: {metrics['value_loss']:.4f}, Total loss: {metrics['total_loss']:.4f}")
        else:
            self.update_shared_policy(all_data, logger)
            # Print the losses that were just logged
            if hasattr(logger, 'metrics') and logger.metrics:
                policy_loss = logger.metrics.get("policy_loss", [])
                value_loss = logger.metrics.get("value_loss", [])
                total_loss = logger.metrics.get("total_loss", [])
                reward = logger.metrics.get("reward", [])
                
                if policy_loss and value_loss and total_loss and reward:
                    print(f"Policy loss: {policy_loss[-1]:.4f}, Value loss: {value_loss[-1]:.4f}, Total loss: {total_loss[-1]:.4f}, Reward: {reward[-1]:.4f}")
        return avg_utility

    def update_labour_values(self, update_idx):
        mid_update = self.num_updates // 2
        if update_idx >= mid_update:
            fraction = 1.0
        else:
            fraction = update_idx / mid_update
        
        labour_types = ["move", "gather", "trade", "build", "no_op"]
        updated_values = {}
        
        for labour_type in labour_types:
            start_key = f"start_{labour_type}_labour"
            final_key = f"{labour_type}_labour"
            
            start_value = self.config.get(start_key, 0.0)
            final_value = self.original_labour_values.get(final_key, 0.0)
            
            current_value = start_value + fraction * (final_value - start_value)
            updated_values[final_key] = current_value
            
            self.config[final_key] = current_value
        
        for pipe in self.vec_env.pipes:
            pipe.send(("update_config", updated_values))
            try:
                pipe.recv()
            except (EOFError, BrokenPipeError):
                continue
        
        return updated_values

    def calculate_decaying_entropy_weight(self, update_idx):
        initial_entropy_weight = self.mat_config.get("entropy_weight", 0.05)
        min_entropy_weight = self.mat_config.get("min_entropy_weight", 0.001)
        
        if update_idx < self.exploration_steps:
            return initial_entropy_weight
        else:
            remaining_updates = self.num_updates - self.exploration_steps
            progress = (update_idx - self.exploration_steps) / (remaining_updates - 1) if remaining_updates > 1 else 1.0
            current_entropy_weight = min_entropy_weight + (1.0 - progress) * (initial_entropy_weight - min_entropy_weight)
            return current_entropy_weight

    def train(self):
        self.shared_policy_path_complete = os.path.join(self.network_folder, f"{self.network_name}_COMPLETE.pth")
        self.shared_policy_path_partial = os.path.join(self.network_folder, f"{self.network_name}_PARTIAL.pth")
        if os.path.exists(self.shared_policy_path_complete):
            self.shared_policy.load_state_dict(torch.load(self.shared_policy_path_complete))
            print(f"Loaded shared policy from {self.shared_policy_path_complete}")
            return
        
        print(f"Training from scratch with {self.num_workers} parallel environments")
        print(f"Starting with {self.exploration_steps} iterations of random action sampling for initial exploration")

        loss_folder = self.config.get("loss_folder_phase1", "loss_plots/phase1")
        metrics_logger = MetricsLogger(loss_folder)

        real_start_time = time.time()
        
        utility_history = []
        utility_tolerance = self.config["mobile_agent_training"].get("utility_tolerance", 0.1)
        utility_patience = self.config["mobile_agent_training"].get("utility_patience", 200)
        
        try:
            if self.remote_updates:
                print("Initializing RPC on Mac...")
                rpc.init_rpc("cpu_driver", rank=1, world_size=2,
                            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                                init_method="tcp://213.192.2.104:40190"
                            ))
                print("RPC Initialized on Mac.")

                rpc.barrier()
                print("RPC barrier passed. Creating remote updater RRef...")

                self.remote_updater_rref = rpc.remote(
                    "gpu_worker", RemoteUpdater, args=(self.config, self.num_numeric)
                )
                print("Remote updater RRef created.")
            for update in range(self.num_updates):
                print("-" * 10 + f"Update {update + 1} of {self.num_updates}" + "-" * 10)


                if update < self.exploration_steps:
                    random_sampling = True
                else:
                    random_sampling = False
                
                collect_start_time = time.time()
                self.collect_rollouts(random_sampling)
                collect_end_time = time.time()
                collect_time = collect_end_time - collect_start_time

                max_buffer_length = max([len(buffer) for buffer in self.rollout_buffers.values()])
                print(f"Max buffer length: {max_buffer_length}")

                print(f"Completed rollout collection in {collect_time:.2f} seconds")

                update_start_time = time.time()
                avg_utility = self.update_agents(metrics_logger)
                update_end_time = time.time()
                update_time = update_end_time - update_start_time
                print(f"Completed update in {update_time:.2f} seconds")
                
                utility_history.append(avg_utility)
                
                if len(utility_history) > utility_patience and update >= self.exploration_steps:
                    recent_utilities = utility_history[-utility_patience:]
                    max_diff = max([abs(recent_utilities[i] - recent_utilities[i-1]) for i in range(1, len(recent_utilities))])
                    
                    if max_diff < utility_tolerance:
                        print(f"Early stopping triggered: utility stable (±{utility_tolerance}) for {utility_patience} updates")
                        print(f"Final average utility: {avg_utility:.2f}")
                        break
                
                labour_values = self.update_labour_values(update)
                self.entropy_weight = self.calculate_decaying_entropy_weight(update)
                
                reset_start_time = time.time()
                self.vec_env.reset_all(randomize_agent_positions=False)
                reset_end_time = time.time()
                reset_time = reset_end_time - reset_start_time

                print(f"Completed reset in {reset_time:.2f} seconds")

                current_time = time.time()
                elapsed_time = current_time - real_start_time
                print(f"Time elapsed: {elapsed_time:.2f} seconds")
                avg_time_per_update = elapsed_time / (update + 1)
                eta = avg_time_per_update * (self.num_updates - update - 1)
                td = timedelta(seconds=eta)
                print(f"ETA: {str(td).split('.')[0]} hours.")
                
                if (update % 1 == 0 or update == self.num_updates - 1):
                    metrics_logger.plot_metrics()
                    torch.save(self.shared_policy.state_dict(), self.shared_policy_path_partial)       
            
            torch.save(self.shared_policy.state_dict(), self.shared_policy_path_complete)
            if os.path.exists(self.shared_policy_path_partial):
                os.remove(self.shared_policy_path_partial)
        finally:
            self.vec_env.close()
            if self.remote_updates:
                rpc.shutdown()





                

