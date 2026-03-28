import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import os
import time
from datetime import timedelta
from tqdm import tqdm
from training.bank_policy import BankPolicy
from training.metrics_logger import MetricsLogger

class BankPPOTrainer:
    def __init__(self, vec_env, config, use_gpu=True):
        self.vec_env = vec_env
        self.config = config
        self.num_workers = self.config.get("n_envs", 4)

        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        self.bank_config = config.get("bank_agent_training", {})

        self.horizon = config.get("episode_length", 1000)
        self.num_updates = self.bank_config.get("num_updates", 1000)
        self.exploration_steps = min(self.bank_config.get("exploration_steps", 10), self.horizon)
        self.gamma = self.bank_config.get("gamma", 0.99)
        self.lam = self.bank_config.get("lambda", 0.95)
        self.clip_epsilon = self.bank_config.get("clip_epsilon", 0.2)
        self.ppo_epochs = self.bank_config.get("ppo_epochs", 4)
        self.mini_batch_size = self.bank_config.get("mini_batch_size", 64)
        self.learning_rate = self.bank_config.get("learning_rate", 3e-4)
        self.value_loss_weight = self.bank_config.get("value_loss_weight", 0.5)
        self.entropy_weight = self.bank_config.get("entropy_weight", 0.01)
        self.target_kl = self.bank_config.get("target_kl", 0.01)
        self.epsilon_explore = self.bank_config.get("epsilon_explore", 0.1)
        
        observation_size = 4
        
        self.policy_net = BankPolicy(
            config=self.config,
            input_size=observation_size,
            output_size=7 
        )
        self.policy_net.to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.rollout_buffers = {f"env{i}": [] for i in range(self.vec_env.total_envs)}
        
        self.network_folder = self.config.get("network_folder", "networks")
        self.network_name = f"bank-n_agents={self.vec_env.n_agents}-" \
            f"experiment_name={self.config.get('experiment_name', 'default_experiment')}"
        
        os.makedirs(self.network_folder, exist_ok=True)
        
        self.reward_normalizer = {
            "running_mean": 0,
            "running_var": 1,
            "count": 0,
            "gamma": 0.99
        }
        
        self.lstm_states = {}

    def normalize_reward(self, reward):
        original_sign = np.sign(reward)

        self.reward_normalizer["count"] += 1
        delta = reward - self.reward_normalizer["running_mean"]
        self.reward_normalizer["running_mean"] += delta / self.reward_normalizer["count"]
        delta2 = reward - self.reward_normalizer["running_mean"]
        self.reward_normalizer["running_var"] += delta * delta2
        
        var = self.reward_normalizer["running_var"] / (self.reward_normalizer["count"] + 1e-8)
        std = np.sqrt(max(var, 1e-8))
        normalized_reward = (reward - self.reward_normalizer["running_mean"]) / std

        normalized_sign = np.sign(normalized_reward)
        if original_sign != 0 and normalized_sign != 0 and original_sign != normalized_sign:
            normalized_reward = abs(normalized_reward) * original_sign
        
        return np.clip(normalized_reward, -10.0, 10.0)
        
    def collect_rollouts(self, random_sampling=False):
        if random_sampling:
            print(f"Collecting rollouts with random sampling.")
        else:
            print(f"Collecting rollouts with network-based sampling.")
            
        self.lstm_states = {}
        
        for t in range(self.horizon):
            observations = []
            env_indices = []
            
            for env_idx in range(self.num_workers):
                pipe = self.vec_env.pipes[env_idx]
                pipe.send(("get_bank_obs", None))
                
            for env_idx in range(self.num_workers):
                try:
                    obs = self.vec_env.pipes[env_idx].recv()
                    observations.append(obs)
                    env_indices.append(env_idx)
                except (EOFError, BrokenPipeError):
                    continue
            
            if not observations:
                continue
                
            obs_tensors = []
            for obs in observations:
                flattened_obs = torch.FloatTensor(obs).to(self.device)
                obs_tensors.append(flattened_obs)
                
            obs_batch = torch.stack(obs_tensors)
            
            action_masks = []
            for env_idx in env_indices:
                pipe = self.vec_env.pipes[env_idx]
                pipe.send(("get_bank_action_mask", None))
                try:
                    mask = pipe.recv()
                    action_masks.append(mask)
                except (EOFError, BrokenPipeError):
                    action_masks.append(True)
            
            if not any(action_masks):
                self.vec_env.step_envs()
                continue
                
            lstm_states_batch = [[], []]
            per_env_lstm_h = []
            per_env_lstm_c = []
            lstm_hidden_size = self.policy_net.lstm_hidden_size
            lstm_num_layers = self.policy_net.lstm_num_layers
            for env_idx in env_indices:
                env_key = f"env{env_idx}"
                lstm_state = self.lstm_states.get(env_key, None)
                if lstm_state is not None:
                    h, c = lstm_state
                    lstm_states_batch[0].append(h)
                    lstm_states_batch[1].append(c)
                    per_env_lstm_h.append(h.detach().squeeze(1).cpu())
                    per_env_lstm_c.append(c.detach().squeeze(1).cpu())
                else:
                    per_env_lstm_h.append(torch.zeros(lstm_num_layers, lstm_hidden_size))
                    per_env_lstm_c.append(torch.zeros(lstm_num_layers, lstm_hidden_size))
            
            lstm_state_batch = None
            if lstm_states_batch[0]:
                h_batch = torch.cat(lstm_states_batch[0], dim=1)
                c_batch = torch.cat(lstm_states_batch[1], dim=1)
                lstm_state_batch = (h_batch, c_batch)

            if random_sampling or random.random() < self.epsilon_explore:
                actions_batch = []
                log_probs_batch = []
                values_batch = torch.zeros(len(observations), 1).to(self.device)
                for _ in range(len(observations)):
                    action = np.random.randint(0, 7)
                    actions_batch.append(action)
                    log_probs_batch.append(0.0)
                    
                if lstm_state_batch:
                    h_shape = lstm_state_batch[0].shape
                    c_shape = lstm_state_batch[1].shape
                    lstm_state_out = (torch.zeros_like(lstm_state_batch[0]), torch.zeros_like(lstm_state_batch[1]))
                else:
                    batch_size = len(observations)
                    h = torch.zeros(self.policy_net.lstm_num_layers, batch_size, self.policy_net.lstm_hidden_size, device=self.device)
                    c = torch.zeros(self.policy_net.lstm_num_layers, batch_size, self.policy_net.lstm_hidden_size, device=self.device)
                    lstm_state_out = (h, c)
                    
                log_probs_batch = torch.tensor(log_probs_batch, device=self.device)
                
            else:
                self.policy_net.eval()
                with torch.no_grad():
                    logits, values_batch, lstm_state_out = self.policy_net(obs_batch, lstm_state_batch)
                    
                    action_masks_tensor = torch.tensor(action_masks, dtype=torch.bool, device=self.device)
                    masked_logits = logits.clone()
                    masked_logits[~action_masks_tensor] = -1e10
                    
                    probs = F.softmax(masked_logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    actions_batch = dist.sample()
                    log_probs_batch = dist.log_prob(actions_batch)
                    
                    actions_batch = actions_batch.cpu().numpy()
            
            if lstm_state_out is not None:
                h_out, c_out = lstm_state_out
                for i, env_idx in enumerate(env_indices):
                    env_key = f"env{env_idx}"
                    h_env = h_out[:, i:i+1, :].detach()
                    c_env = c_out[:, i:i+1, :].detach()
                    self.lstm_states[env_key] = (h_env, c_env)
                
            for i, env_idx in enumerate(env_indices):
                if not action_masks[i]:
                    continue
                    
                action = actions_batch[i]
                env_key = f"env{env_idx}"
                
                pipe = self.vec_env.pipes[env_idx]
                pipe.send(("bank_step", action))
                
                try:
                    pre_utility, post_utility = pipe.recv()
                    reward = post_utility - pre_utility
                    
                    normalized_reward = self.normalize_reward(reward)
                    
                    self.rollout_buffers[env_key].append({
                        "obs": obs_tensors[i].detach(),
                        "action": action,
                        "log_prob": log_probs_batch[i].detach() if isinstance(log_probs_batch, torch.Tensor) else torch.tensor(log_probs_batch[i], device=self.device),
                        "value": values_batch[i].detach() if isinstance(values_batch, torch.Tensor) else torch.tensor(0.0, device=self.device),
                        "reward": normalized_reward,
                        "utility": post_utility,
                        "original_reward": reward,
                        "lstm_h": per_env_lstm_h[i],
                        "lstm_c": per_env_lstm_c[i],
                    })
                    
                except (EOFError, BrokenPipeError):
                    continue
                    
            self.vec_env.step_envs()
            
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
        
    def update_policy(self, all_data, logger):
        observations = torch.stack([t["obs"] for t in all_data]).to(self.device)

        actions = torch.tensor([t["action"] for t in all_data], dtype=torch.long).to(self.device)

        old_log_probs = torch.stack([t["log_prob"] for t in all_data]).to(self.device)
        advantages = torch.FloatTensor([t["advantages"] for t in all_data]).to(self.device)
        returns = torch.FloatTensor([t["returns"] for t in all_data]).to(self.device)
        all_lstm_h = torch.stack([t["lstm_h"] for t in all_data]).to(self.device)
        all_lstm_c = torch.stack([t["lstm_c"] for t in all_data]).to(self.device)
        
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
                end = min(start + self.mini_batch_size, dataset_size)
                batch_idx = indices[start:end]
                
                batch_obs = observations[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_h = all_lstm_h[batch_idx].permute(1, 0, 2).contiguous()
                batch_c = all_lstm_c[batch_idx].permute(1, 0, 2).contiguous()

                self.policy_net.train()
                logits, values, _ = self.policy_net(batch_obs, (batch_h, batch_c))
                values = values.squeeze(1)
                
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values, batch_returns)
                
                loss = policy_loss + self.value_loss_weight * value_loss - self.entropy_weight * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                clip_norm = self.config.get("bank_agent_training", {}).get("gradient_clip_norm", 10.0)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), clip_norm)

                has_nan = False
                for param in self.policy_net.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print("Warning: NaN gradient detected in bank, skipping update")
                        has_nan = True
                        break

                if not has_nan:
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_loss += loss.item()
                    batches += 1

                    log_ratio = new_log_probs - batch_old_log_probs
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                    approx_kl_divs.append(approx_kl_div)

            if approx_kl_divs:
                avg_kl = sum(approx_kl_divs) / len(approx_kl_divs)
                if avg_kl > self.target_kl:
                    break

        if batches == 0:
            batches = 1
        avg_policy_loss = total_policy_loss / batches
        avg_value_loss = total_value_loss / batches
        avg_total_loss = total_loss / batches
        
        logger.log("bank_policy_loss", avg_policy_loss)
        logger.log("bank_value_loss", avg_value_loss)
        logger.log("bank_total_loss", avg_total_loss)
        
        return avg_total_loss
    
    def update_bank(self, logger):
        all_data = []
        utilities = []
        
        for env_key, buffer in self.rollout_buffers.items():
            if not buffer:
                continue
                
            advantages, returns = self.compute_gae(buffer)
            
            for i, entry in enumerate(buffer):
                entry["advantages"] = advantages[i]
                entry["returns"] = returns[i]
                
            all_data.extend(buffer)
                        
            avg_utility = np.mean([t["utility"] for t in buffer])
            utilities.append(avg_utility)
            
            self.rollout_buffers[env_key] = []
            
        if all_data:
            avg_reward = np.mean([t["reward"] for t in all_data])
            logger.log("bank_reward", avg_reward)
            
            if utilities:
                avg_utility = np.mean(utilities)
                logger.log("bank_utility", avg_utility)
                print(f"Average bank utility: {avg_utility:.2f}")
                
        if all_data:
            self.update_policy(all_data, logger)
            return avg_utility if utilities else 0
        else:
            return 0
            
    def calculate_decaying_entropy_weight(self, update_idx):
        initial_entropy_weight = self.bank_config.get("entropy_weight", 0.05)
        min_entropy_weight = self.bank_config.get("min_entropy_weight", 0.001)
        progress = update_idx / (self.num_updates - 1) if self.num_updates > 1 else 1.0
        current_entropy_weight = min_entropy_weight + (1.0 - progress) * (initial_entropy_weight - min_entropy_weight)
        
        return current_entropy_weight
        
    def train(self):
        shared_policy_path_complete = os.path.join(self.network_folder, f"{self.network_name}_COMPLETE.pth")
        shared_policy_path_partial = os.path.join(self.network_folder, f"{self.network_name}_PARTIAL.pth")
        
        if os.path.exists(shared_policy_path_complete):
            self.policy_net.load_state_dict(torch.load(shared_policy_path_complete))
            print(f"Loaded policy from {shared_policy_path_complete}")
            return
            
        print(f"Training bank agent from scratch with {self.num_workers} parallel environments")
        print(f"Starting with {self.exploration_steps} iterations of random action sampling for initial exploration")
        
        loss_folder = self.config.get("loss_folder_phase3", "loss_plots/phase3")
        metrics_logger = MetricsLogger(loss_folder)
        
        real_start_time = time.time()
        
        utility_history = []
        utility_tolerance = self.bank_config.get("utility_tolerance", 2.0)
        utility_patience = self.bank_config.get("utility_patience", 10)
        
        try:
            for update in range(self.num_updates):
                print("-" * 10 + f"Bank Update {update + 1} of {self.num_updates}" + "-" * 10)
                self.lstm_states = {}
                
                if update < self.exploration_steps:
                    random_sampling = True
                else:
                    random_sampling = False
                    
                for env_idx in range(self.vec_env.total_envs):
                    self.vec_env.pipes[env_idx].send(("update_bank_annealing", update))
                    try:
                        self.vec_env.pipes[env_idx].recv()
                    except (EOFError, BrokenPipeError):
                        continue
                    
                collect_start_time = time.time()
                self.collect_rollouts(random_sampling)
                collect_end_time = time.time()
                collect_time = collect_end_time - collect_start_time
                
                print(f"Completed bank rollout collection in {collect_time:.2f} seconds")
                
                update_start_time = time.time()
                avg_utility = self.update_bank(metrics_logger)
                update_end_time = time.time()
                update_time = update_end_time - update_start_time
                
                print(f"Completed bank update in {update_time:.2f} seconds")
                
                utility_history.append(avg_utility)
                
                if len(utility_history) > utility_patience and update >= self.exploration_steps:
                    recent_utilities = utility_history[-utility_patience:]
                    max_diff = max([abs(recent_utilities[i] - recent_utilities[i-1]) for i in range(1, len(recent_utilities))])
                    
                    if max_diff < utility_tolerance:
                        print(f"Early stopping triggered: bank utility stable (±{utility_tolerance}) for {utility_patience} updates")
                        print(f"Final average bank utility: {avg_utility:.2f}")
                        break
                        
                self.entropy_weight = self.calculate_decaying_entropy_weight(update)
                
                reset_start_time = time.time()
                self.vec_env.reset_all(randomize_agent_positions=True)
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
                    torch.save(self.policy_net.state_dict(), shared_policy_path_partial)
                    
            torch.save(self.policy_net.state_dict(), shared_policy_path_complete)
            
            if os.path.exists(shared_policy_path_partial):
                os.remove(shared_policy_path_partial)
                
        finally:
            pass 