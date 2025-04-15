from training.mobile_agent_ppo_trainer import MultiAgentPPOTrainer
from training.planner_ppo_trainer import PlannerPPOTrainer
from training.vectorized_env import VectorizedEnv
from training.metrics_logger import MetricsLogger
from training.mobile_agent_policy import MobileAgentPolicy
from training.planner_policy import PlannerPolicy
import time
import os
import torch
from datetime import timedelta
import numpy as np

class TwoPhaseTrainer:
    def __init__(self, config):
        self.config = config
        self.original_config = config.copy()
        
        self.network_folder = config.get("network_folder", "networks")
        self.experiment_name = config.get("experiment_name", "default_experiment")
        self.loss_folder_phase1 = config.get("loss_folder_phase1", "loss_plots/phase1")
        self.loss_folder_phase2 = config.get("loss_folder_phase2", "loss_plots/phase2")

        os.makedirs(self.network_folder, exist_ok=True)
        os.makedirs(self.loss_folder_phase1, exist_ok=True)
        os.makedirs(self.loss_folder_phase2, exist_ok=True)

        self.use_gpu = config.get("use_gpu", True)
        
        self.mobile_trainer = None
        self.planner_trainer = None
        self.mobile_policy_path = None
        
    def phase_one(self):
        print("\n" + "="*50)
        print("PHASE 1: Training Mobile Agents with Zero Taxes")
        print("="*50 + "\n")
        
        phase1_config = self.config.copy()
        phase1_config["planner"] = True  
        phase1_config["bank"] = True  
        
        print("Creating vectorized environment for Phase 1...")
        vec_env = VectorizedEnv(phase1_config)

        print("Setting up mobile agent PPO trainer...")
        self.mobile_trainer = MultiAgentPPOTrainer(vec_env, phase1_config, self.use_gpu)

        print("Applying zero tax policy to planner agents...")
        self._apply_zero_tax_policy(vec_env)
        print("Applying zero policy to bank agents...")
        self._apply_zero_bank_policy(vec_env)

        phase_1_start_time = time.time()
        try:
            print("Training mobile agents with zero taxes and zero bank policy...")
            self.mobile_trainer.train()
            print("Mobile agent training complete.")
            
            self.mobile_policy_path = self._save_phase1_model()
            
        finally:
            vec_env.close()
            print("Vectorized environment closed.")
            
        phase_1_duration = time.time() - phase_1_start_time
        print(f"\nPHASE 1 COMPLETED IN {phase_1_duration:.2f} SECONDS")
        
        return phase_1_duration
    
    def phase_two(self):
        if self.mobile_policy_path is None:
            raise ValueError("Mobile policy path is not set. Phase 1 must be run first.")
            
        print("\n" + "="*50)
        print("PHASE 2: Training Planner Agents with Taxes")
        print("="*50 + "\n")

        mobile_phase2_path = os.path.join(
            self.network_folder, 
            f"mobile_agents-phase_2-n_agents={self.config.get('n_agents', 4)}-experiment_name={self.experiment_name}_COMPLETE.pth"
        )
        planner_phase2_path = os.path.join(
            self.network_folder, 
            f"planner_agent-phase_2-n_agents={self.config.get('n_agents', 4)}-experiment_name={self.experiment_name}_COMPLETE.pth"
        )

        if os.path.exists(mobile_phase2_path) and os.path.exists(planner_phase2_path):
            print("Found existing phase 2 complete models. Skipping phase 2 training.")
            print(f"Mobile agent model: {mobile_phase2_path}")
            print(f"Planner model: {planner_phase2_path}")
            return 0.0 

        joint_params = self.config.get("joint_training", {})
        num_joint_episodes = joint_params.get("num_updates", 200)
        exploration_steps = joint_params.get("exploration_steps", 100)

        episode_length = self.config.get("episode_length", 100)
        tax_period_length = self.config.get("tax_period_length", 100)

        phase2_config = self.config.copy()
        phase2_config["planner"] = True  
        phase2_config["bank"] = True  
        
        print("Creating vectorized environment for Phase 2...")
        vec_env = VectorizedEnv(phase2_config)

        print("Applying zero policy to bank agents...")
        self._apply_zero_bank_policy(vec_env)

        print("Loading pre-trained mobile agent policy...")
        mobile_policy = self._create_mobile_policy(phase2_config, vec_env)
        
        self.mobile_trainer = self._setup_phase2_mobile_trainer(vec_env, phase2_config, mobile_policy)
        self.planner_trainer = self._setup_planner_trainer(vec_env, phase2_config, exploration_steps)
        
        mobile_metrics_logger = MetricsLogger(self.loss_folder_phase2 + "/mobile")
        planner_metrics_logger = MetricsLogger(self.loss_folder_phase2 + "/planner")

        self.mobile_trainer.lstm_states = {}
        self.planner_trainer.lstm_states = {}

        mobile_utility_history = []
        planner_utility_history = []
        utility_tolerance = joint_params.get("utility_tolerance", 1.0)
        utility_patience = joint_params.get("utility_patience", 10)

        print(f"Will train phase 2 for {num_joint_episodes} episodes with utility tolerance {utility_tolerance} and patience {utility_patience}.")
        
        phase2_start_time = time.time()

        try:
            print(f"Starting joint training with {num_joint_episodes} episodes.")
            full_start_time = time.time()

            for episode in range(num_joint_episodes):
                print(f"\n--- Joint Training Episode {episode + 1}/{num_joint_episodes} ---")
                episode_start_time = time.time()
                
                vec_env.reset_all(randomize_agent_positions=True)
                self._reset_buffers_for_joint_training()
                
                print("Running joint training episode...")
                self._run_joint_training_episode(
                    vec_env,
                    episode,
                    episode_length,
                    tax_period_length,
                    exploration_steps,
                )

                print(f"Done running joint training episode in {time.time() - episode_start_time:.2f} seconds.")
                
                mobile_avg_utility, planner_avg_utility = self._update_models_after_episode(
                    episode,
                    mobile_metrics_logger,
                    planner_metrics_logger,
                    mobile_utility_history,
                    planner_utility_history
                )

                print(f"Updated models in {time.time() - episode_start_time:.2f} seconds.")

                planner_utility_history.append(planner_avg_utility)
                mobile_utility_history.append(mobile_avg_utility)
                
                self._save_interim_models_if_needed(episode, num_joint_episodes, mobile_metrics_logger, planner_metrics_logger)
                
                if self._should_stop_early(
                    planner_utility_history,
                    mobile_utility_history,
                    utility_patience,
                    utility_tolerance,
                    episode,
                    exploration_steps
                ):
                    print("Early stopping triggered - utilities have stabilized")
                    break
                
                self._update_entropy_weights(episode, exploration_steps)

                current_time = time.time()
                episode_duration = current_time - episode_start_time
                avg_time_per_episode = (current_time - full_start_time) / (episode + 1)
                remaining_episodes = num_joint_episodes - (episode + 1)
                eta_seconds = remaining_episodes * avg_time_per_episode
                eta = timedelta(seconds=eta_seconds)
                
                print(f"\nEpisode duration: {episode_duration:.2f}s")
                print(f"ETA: {str(eta).split('.')[0]}")
                
            
            self._save_final_models()
            
        finally:
            vec_env.close()
            print("Vectorized environment closed.")
            
        phase_2_duration = time.time() - phase2_start_time
        print(f"\nPHASE 2 COMPLETED IN {phase_2_duration:.2f} SECONDS")
        
        return phase_2_duration
    
    def train_two_phase(self):
        print("\n" + "="*50)
        print("Starting Two-Phase Training")
        print("="*50 + "\n")
        
        phase1_duration = self.phase_one()
        phase2_duration = self.phase_two()
        
        total_duration = phase1_duration + phase2_duration
        print(f"\nTotal training completed in {timedelta(seconds=total_duration)}")
        

    def _apply_zero_tax_policy(self, vec_env):
        for env_idx in range(vec_env.total_envs):
            vec_env.pipes[env_idx].send(("wrap_planner_with_zero_tax", None))
            try:
                vec_env.pipes[env_idx].recv()
            except (EOFError, BrokenPipeError):
                continue
                
    def _apply_zero_bank_policy(self, vec_env):
        for env_idx in range(vec_env.total_envs):
            vec_env.pipes[env_idx].send(("wrap_bank_with_zero_policy", None))
            try:
                vec_env.pipes[env_idx].recv()
            except (EOFError, BrokenPipeError):
                continue
                
    def _save_phase1_model(self):
        mobile_policy_path = os.path.join(
            self.network_folder, 
            f"mobile_agents-phase_1-n_agents={self.mobile_trainer.vec_env.n_agents}-experiment_name={self.experiment_name}_COMPLETE.pth"
        )
        torch.save(self.mobile_trainer.shared_policy.state_dict(), mobile_policy_path)
        print(f"Saved mobile agent policy to {mobile_policy_path}")
        return mobile_policy_path
        
    def _create_mobile_policy(self, config, vec_env):
        basic_numeric_size = 7 + vec_env.n_agents  # position(2) + inventory(3) + build_payout(1) + time_to_tax(1) + incomes(n_agents)
        planner_features_size = config.get("n_tax_brackets", 7) + 1  # tax_rates (n_brackets) + tax_bracket
        bank_features_size = 4  # inflation_rate, interest_rate, money_supply
        
        num_numeric = basic_numeric_size + planner_features_size + bank_features_size
        action_range = vec_env.env_ref.mobile_agents[0].action_range
        
        # Create policy and load pre-trained weights
        mobile_policy = MobileAgentPolicy(
            config=config,
            num_numeric=num_numeric,
            action_range=action_range
        )
        mobile_policy.load_state_dict(torch.load(self.mobile_policy_path))
        print(f"Loaded pre-trained mobile agent policy from {self.mobile_policy_path}")
        
        return mobile_policy
        
    def _create_planner_policy(self, config, vec_env):
        n_agents = vec_env.n_agents
        n_tax_brackets = config.get("n_tax_brackets", 7)
        observation_size = 8 + (3 * n_agents + 2) + (n_tax_brackets + n_agents + 1) + 2
        
        planner_policy = PlannerPolicy(
            config=config,
            input_size=observation_size,
            output_size=n_tax_brackets
        )
        
        if hasattr(self, 'planner_policy_path') and self.planner_policy_path:
            planner_policy.load_state_dict(torch.load(self.planner_policy_path))
            print(f"Loaded pre-trained planner policy from {self.planner_policy_path}")
            
        return planner_policy
        
    def _setup_phase2_mobile_trainer(self, vec_env, config, mobile_policy):
        mobile_trainer = MultiAgentPPOTrainer(vec_env, config, self.use_gpu)
        mobile_trainer.shared_policy = mobile_policy
        mobile_trainer.shared_policy.to(mobile_trainer.device)
        return mobile_trainer
        
    def _setup_planner_trainer(self, vec_env, config, exploration_steps):
        planner_config = config.copy()
        planner_config["planner_agent_training"] = planner_config.get("planner_agent_training", {}).copy()
        planner_config["planner_agent_training"]["exploration_steps"] = exploration_steps
        return PlannerPPOTrainer(vec_env, planner_config, self.use_gpu)
        
    def _reset_buffers_for_joint_training(self):
        self.mobile_trainer.lstm_states = {}
        self.planner_trainer.lstm_states = {}
        
        for key in list(self.mobile_trainer.rollout_buffers.keys()):
            self.mobile_trainer.rollout_buffers[key] = []
        
        for key in list(self.planner_trainer.rollout_buffers.keys()):
            self.planner_trainer.rollout_buffers[key] = []
            
    def _run_joint_training_episode(self, vec_env, episode, episode_length, tax_period_length, 
                                   exploration_steps):
        random_planner_actions = episode < exploration_steps
        
        mobile_current_buffers = {env_idx: [] for env_idx in range(vec_env.total_envs)}
        planner_current_buffers = {env_idx: [] for env_idx in range(vec_env.total_envs)}
        
        for step in range(episode_length):
            is_tax_year_start = (step > 0) and (step % tax_period_length == 0)
            
            if is_tax_year_start:
                self._handle_tax_year_start(
                    vec_env, 
                    random_planner_actions, 
                    planner_current_buffers
                )
            
            self._handle_mobile_agent_steps(vec_env, mobile_current_buffers)
            
            self._update_environment_state(vec_env)
        
        self._process_buffers_after_episode(mobile_current_buffers, planner_current_buffers)
        
    def _handle_tax_year_start(self, vec_env, random_sampling, planner_current_buffers):
        observations = []
        env_indices = []
        
        for env_idx in range(vec_env.total_envs):
            vec_env.pipes[env_idx].send(("get_planner_obs", None))
            try:
                obs = vec_env.pipes[env_idx].recv()
                if obs is not None:
                    observations.append(obs)
                    env_indices.append(env_idx)
            except (EOFError, BrokenPipeError):
                continue
        
        if not observations:
            return
            
        obs_tensors = []
        for obs in observations:
            flattened_obs = torch.FloatTensor(obs).to(self.planner_trainer.device)
            obs_tensors.append(flattened_obs)
            
        action_masks = []
        for env_idx in env_indices:
            vec_env.pipes[env_idx].send(("get_planner_action_mask", None))
            try:
                mask = vec_env.pipes[env_idx].recv()
                action_masks.append(mask)
            except (EOFError, BrokenPipeError):
                action_masks.append(False)
        
        if not (obs_tensors and any(action_masks)):
            return
            
        obs_batch = torch.stack(obs_tensors)
        
        lstm_states_batch = self._get_planner_lstm_states_batch(env_indices)
        
        if random_sampling:
            actions_batch, log_probs_batch, values_batch, lstm_state_out = self._get_random_planner_actions(
                len(obs_tensors),
                lstm_states_batch
            )
        else:
            actions_batch, log_probs_batch, values_batch, lstm_state_out = self._get_planner_network_actions(
                obs_batch,
                lstm_states_batch
            )
            
        if lstm_state_out is not None:
            self._update_planner_lstm_states(lstm_state_out, env_indices)
            
        for i, env_idx in enumerate(env_indices):
            if not action_masks[i]:
                continue
                
            action = actions_batch[i]
            
            vec_env.pipes[env_idx].send(("get_planner_utility", None))
            try:
                pre_utility = vec_env.pipes[env_idx].recv()
                
                vec_env.pipes[env_idx].send(("planner_step", action))
                post_utility, _ = vec_env.pipes[env_idx].recv()
                
                reward = post_utility - pre_utility
                normalized_reward = self.planner_trainer.normalize_reward(reward)
                
                planner_current_buffers[env_idx].append({
                    "obs": obs_tensors[i].detach(),
                    "action": action,
                    "log_prob": log_probs_batch[i].detach() if isinstance(log_probs_batch, torch.Tensor) 
                              else torch.tensor(log_probs_batch[i], device=self.planner_trainer.device),
                    "value": values_batch[i].detach() if isinstance(values_batch, torch.Tensor) 
                            else torch.tensor(0.0, device=self.planner_trainer.device),
                    "reward": normalized_reward,
                    "utility": post_utility,
                    "original_reward": reward
                })
            except (EOFError, BrokenPipeError):
                continue
                
    def _get_planner_lstm_states_batch(self, env_indices):
        h_list, c_list = [], []
        
        for env_idx in env_indices:
            env_key = f"env{env_idx}"
            if env_key in self.planner_trainer.lstm_states:
                h, c = self.planner_trainer.lstm_states[env_key]
                h_list.append(h)
                c_list.append(c)
                
        if h_list and c_list:
            h_batch = torch.cat(h_list, dim=1)
            c_batch = torch.cat(c_list, dim=1)
            return (h_batch, c_batch)
        return None
        
    def _get_random_planner_actions(self, batch_size, lstm_states_batch):
        actions_batch = []
        log_probs_batch = []
        
        for _ in range(batch_size):
            action = np.random.rand(self.planner_trainer.n_tax_brackets)
            actions_batch.append(action)
            log_probs_batch.append(0.0)
            
        values_batch = torch.zeros(batch_size, 1).to(self.planner_trainer.device)
        
        if lstm_states_batch:
            h, c = lstm_states_batch
            lstm_state_out = (torch.zeros_like(h), torch.zeros_like(c))
        else:
            h = torch.zeros(
                self.planner_trainer.policy_net.lstm_num_layers, 
                batch_size, 
                self.planner_trainer.policy_net.lstm_hidden_size, 
                device=self.planner_trainer.device
            )
            c = torch.zeros_like(h)
            lstm_state_out = (h, c)
            
        log_probs_batch = torch.tensor(log_probs_batch, device=self.planner_trainer.device)
        return actions_batch, log_probs_batch, values_batch, lstm_state_out
        
    def _get_planner_network_actions(self, obs_batch, lstm_states_batch):
        self.planner_trainer.policy_net.eval()
        with torch.no_grad():
            logits, values_batch, lstm_state_out = self.planner_trainer.policy_net(
                obs_batch, lstm_states_batch
            )
            
            actions_tensor = torch.sigmoid(logits)
            
            sigma = 0.1
            dist = torch.distributions.Normal(actions_tensor, sigma)
            
            sampled_actions = actions_tensor
            log_probs_batch = torch.sum(dist.log_prob(sampled_actions), dim=1)
            
            actions_batch = [action.cpu().numpy() for action in actions_tensor]
            
        return actions_batch, log_probs_batch, values_batch, lstm_state_out
        
    def _update_planner_lstm_states(self, lstm_state_out, env_indices):
        h_out, c_out = lstm_state_out
        for i, env_idx in enumerate(env_indices):
            env_key = f"env{env_idx}"
            h_env = h_out[:, i:i+1, :].detach()
            c_env = c_out[:, i:i+1, :].detach()
            self.planner_trainer.lstm_states[env_key] = (h_env, c_env)
            
    def _handle_mobile_agent_steps(self, vec_env, mobile_current_buffers):
        all_agents_data = vec_env.get_all_agent_observations()
        
        agents_by_env = {}
        for agent_data in all_agents_data:
            env_idx = agent_data["env_idx"]
            if env_idx not in agents_by_env:
                agents_by_env[env_idx] = []
            agents_by_env[env_idx].append(agent_data)
            
        for env_idx, agents_data in agents_by_env.items():
            for agent_data in agents_data:
                agent_id = agent_data["agent_id"]
                obs = agent_data["obs"]
                
                neighbourhood, numeric = self.mobile_trainer.flatten_observation(obs)
                neighbourhood = neighbourhood.unsqueeze(0).to(self.mobile_trainer.device)
                numeric = numeric.unsqueeze(0).to(self.mobile_trainer.device)
                
                action_mask = vec_env.get_agent_action_mask(env_idx, agent_id)
                
                agent_key = f"env{env_idx}_agent{agent_id}"
                lstm_state = self.mobile_trainer.lstm_states.get(agent_key, None)
                
                action, log_prob, value, lstm_state_out = self._get_mobile_agent_action(
                    neighbourhood, 
                    numeric, 
                    lstm_state, 
                    action_mask
                )
                
                if lstm_state_out is not None:
                    self.mobile_trainer.lstm_states[agent_key] = (
                        lstm_state_out[0].detach(),
                        lstm_state_out[1].detach()
                    )
                    
                action_int = action.item()
                reward, post_utility = vec_env.agent_step(env_idx, agent_id, action_int)
                
                if env_idx not in mobile_current_buffers:
                    mobile_current_buffers[env_idx] = []
                    
                mobile_current_buffers[env_idx].append({
                    "neighbourhood": neighbourhood.squeeze(0).detach(),
                    "numeric": numeric.squeeze(0).detach(),
                    "action": action_int,
                    "log_prob": log_prob.detach(),
                    "value": value.squeeze(1).detach(),
                    "reward": self.mobile_trainer.normalize_reward(reward),
                    "utility": post_utility,
                    "original_reward": reward,
                    "action_mask": action_mask,
                    "agent_id": agent_id,
                    "env_idx": env_idx,
                    "buffer_key": f"env{env_idx}_agent{agent_id}"
                })
                
    def _get_mobile_agent_action(self, neighbourhood, numeric, lstm_state, action_mask):
        self.mobile_trainer.shared_policy.eval()
        with torch.no_grad():
            logits, value, lstm_state_out = self.mobile_trainer.shared_policy(
                neighbourhood, numeric, lstm_state
            )
            
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.mobile_trainer.device)
            masked_logits = logits.clone()
            masked_logits[~action_mask_tensor] = -1e10
            
            probs = torch.nn.functional.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action, log_prob, value, lstm_state_out
        
    def _update_environment_state(self, vec_env):
        for env_idx in range(vec_env.total_envs):
            vec_env.pipes[env_idx].send(("update_environment", None))
            try:
                vec_env.pipes[env_idx].recv()
            except (EOFError, BrokenPipeError):
                continue
                
    def _process_buffers_after_episode(self, mobile_current_buffers, planner_current_buffers):
        for env_idx, buffer in mobile_current_buffers.items():
            if not buffer:
                continue
                
            agent_buffers = {}
            for entry in buffer:
                agent_id = entry["agent_id"]
                buffer_key = entry["buffer_key"]
                if buffer_key not in agent_buffers:
                    agent_buffers[buffer_key] = []
                    
                agent_buffers[buffer_key].append({
                    "neighbourhood": entry["neighbourhood"],
                    "numeric": entry["numeric"],
                    "action": entry["action"],
                    "log_prob": entry["log_prob"],
                    "value": entry["value"],
                    "reward": entry["reward"],
                    "utility": entry["utility"],
                    "original_reward": entry["original_reward"],
                    "action_mask": entry["action_mask"]
                })
                
            for buffer_key, agent_buffer in agent_buffers.items():
                self.mobile_trainer.rollout_buffers[buffer_key] = agent_buffer
                
        for env_idx, buffer in planner_current_buffers.items():
            if buffer:
                self.planner_trainer.rollout_buffers[f"env{env_idx}"] = buffer
                
    def _update_models_after_episode(self, episode, mobile_metrics_logger, planner_metrics_logger, 
                                    mobile_utility_history, planner_utility_history):
        print("Updating mobile agent policies...")
        mobile_avg_utility = self.mobile_trainer.update_agents(mobile_metrics_logger)
        print(f"Mobile agents average utility: {mobile_avg_utility:.4f}")
        mobile_utility_history.append(mobile_avg_utility)
        
        print("Updating planner policy...")
        planner_avg_utility = self.planner_trainer.update_planner(planner_metrics_logger)
        print(f"Planner average utility: {planner_avg_utility:.4f}")
        
        if planner_avg_utility != 0:
            planner_utility_history.append(planner_avg_utility)
            
        return mobile_avg_utility, planner_avg_utility
        
    def _save_interim_models_if_needed(self, episode, num_joint_episodes, mobile_metrics_logger, planner_metrics_logger):
        if episode % 1 == 0 or episode == num_joint_episodes - 1:
            interim_mobile_path = os.path.join(
                self.network_folder, 
                f"mobile_agents-phase_2-n_agents={self.mobile_trainer.vec_env.n_agents}-experiment_name={self.experiment_name}_PARTIAL.pth"
            )
            torch.save(self.mobile_trainer.shared_policy.state_dict(), interim_mobile_path)
            
            interim_planner_path = os.path.join(
                self.network_folder, 
                f"planner_agent-phase_2-n_agents={self.planner_trainer.vec_env.n_agents}-experiment_name={self.experiment_name}_PARTIAL.pth"
            )
            torch.save(self.planner_trainer.policy_net.state_dict(), interim_planner_path)
            
            mobile_metrics_logger.plot_metrics()
            planner_metrics_logger.plot_metrics()
            
            print(f"Saved interim models at episode {episode+1}")
            
    def _should_stop_early(self, planner_utility_history, mobile_utility_history, utility_patience, utility_tolerance, episode, exploration_steps):
        if (len(planner_utility_history) > utility_patience and 
            len(mobile_utility_history) > utility_patience and 
            episode >= exploration_steps):
            
            recent_planner_utilities = planner_utility_history[-utility_patience:]
            recent_mobile_utilities = mobile_utility_history[-utility_patience:]
            
            planner_max_diff = max([abs(recent_planner_utilities[i] - recent_planner_utilities[i-1]) 
                                  for i in range(1, len(recent_planner_utilities))])
            mobile_max_diff = max([abs(recent_mobile_utilities[i] - recent_mobile_utilities[i-1]) 
                                 for i in range(1, len(recent_mobile_utilities))])
            
            if planner_max_diff < utility_tolerance and mobile_max_diff < utility_tolerance:
                print(f"Early stopping triggered: planner and mobile utilities stable (±{utility_tolerance}) for {utility_patience} episodes")
                print(f"Final planner average utility: {planner_utility_history[-1]:.4f}")
                print(f"Final mobile average utility: {mobile_utility_history[-1]:.4f}")
                return True
        return False
        
    def _update_entropy_weights(self, episode, exploration_steps):
        if episode >= exploration_steps:
            self.mobile_trainer.entropy_weight = self.mobile_trainer.calculate_decaying_entropy_weight(episode - exploration_steps)
            self.planner_trainer.entropy_weight = self.planner_trainer.calculate_decaying_entropy_weight(episode - exploration_steps)
        else:
            self.mobile_trainer.entropy_weight = self.mobile_trainer.mat_config.get("entropy_weight", 0.05)
            self.planner_trainer.entropy_weight = self.planner_trainer.planner_config.get("entropy_weight", 0.05)
        
    def _save_final_models(self):
        final_mobile_path = os.path.join(
            self.network_folder, 
            f"mobile_agents-phase_2-n_agents={self.mobile_trainer.vec_env.n_agents}-experiment_name={self.experiment_name}_COMPLETE.pth"
        )
        torch.save(self.mobile_trainer.shared_policy.state_dict(), final_mobile_path)
        
        final_planner_path = os.path.join(
            self.network_folder, 
            f"planner_agent-phase_2-n_agents={self.planner_trainer.vec_env.n_agents}-experiment_name={self.experiment_name}_COMPLETE.pth"
        )
        torch.save(self.planner_trainer.policy_net.state_dict(), final_planner_path)

        interim_mobile_path = os.path.join(
            self.network_folder, 
            f"mobile_agents-phase_2-n_agents={self.mobile_trainer.vec_env.n_agents}-experiment_name={self.experiment_name}_PARTIAL.pth"
        )
        if os.path.exists(interim_mobile_path):
            os.remove(interim_mobile_path)

        interim_planner_path = os.path.join(
            self.network_folder, 
            f"planner_agent-phase_2-n_agents={self.planner_trainer.vec_env.n_agents}-experiment_name={self.experiment_name}_PARTIAL.pth"
        )
        if os.path.exists(interim_planner_path):
            os.remove(interim_planner_path)
        
        print("Joint training completed!")
        print(f"Saved final mobile agent policy to {final_mobile_path}")
        print(f"Saved final planner policy to {final_planner_path}")

