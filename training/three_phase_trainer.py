from training.two_phase_trainer import TwoPhaseTrainer
from training.bank_ppo_trainer import BankPPOTrainer
from training.vectorized_env import VectorizedEnv
from training.metrics_logger import MetricsLogger
from training.bank_policy import BankPolicy
import time
import os
import torch
from datetime import timedelta
import numpy as np

class ThreePhaseTrainer:
    def __init__(self, config):
        self.config = config
        self.original_config = config.copy()
        
        self.network_folder = config.get("network_folder", "networks")
        self.experiment_name = config.get("experiment_name", "default_experiment")
        self.loss_folder_phase1 = config.get("loss_folder_phase1", "loss_plots/phase1")
        self.loss_folder_phase2 = config.get("loss_folder_phase2", "loss_plots/phase2")
        self.loss_folder_phase3 = config.get("loss_folder_phase3", "loss_plots/phase3")

        os.makedirs(self.network_folder, exist_ok=True)
        os.makedirs(self.loss_folder_phase1, exist_ok=True)
        os.makedirs(self.loss_folder_phase2, exist_ok=True)
        os.makedirs(self.loss_folder_phase3, exist_ok=True)

        self.use_gpu = config.get("use_gpu", True)
        
        self.two_phase_trainer = None
        self.bank_trainer = None
        self.mobile_policy_path = None
        self.planner_policy_path = None
        
    def phase_one_and_two(self):
        print("\n" + "="*50)
        print("PHASES 1 & 2: Training Mobile Agents and Planner")
        print("="*50 + "\n")
        
        self.two_phase_trainer = TwoPhaseTrainer(self.config)
        self.two_phase_trainer.train_two_phase()
        
        self.mobile_policy_path = os.path.join(
            self.network_folder, 
            f"mobile_agents-phase_2-n_agents={self.config.get('n_agents', 4)}-experiment_name={self.experiment_name}_COMPLETE.pth"
        )
        self.planner_policy_path = os.path.join(
            self.network_folder, 
            f"planner_agent-phase_2-n_agents={self.config.get('n_agents', 4)}-experiment_name={self.experiment_name}_COMPLETE.pth"
        )
        
    def phase_three(self):
        if self.mobile_policy_path is None or self.planner_policy_path is None:
            raise ValueError("Mobile and planner policy paths are not set. Phases 1 & 2 must be run first.")
            
        print("\n" + "="*50)
        print("PHASE 3: Joint Training with Bank Agent")
        print("="*50 + "\n")

        joint_params = self.config.get("joint_training", {})
        num_joint_episodes = joint_params.get("num_updates", 200)
        exploration_steps = joint_params.get("exploration_steps", 100)

        episode_length = self.config.get("episode_length", 100)
        tax_period_length = self.config.get("tax_period_length", 100)

        phase3_config = self.config.copy()
        phase3_config["planner"] = True
        phase3_config["bank"] = True
        
        print("Creating vectorized environment for Phase 3...")
        vec_env = VectorizedEnv(phase3_config)

        print("Loading pre-trained mobile and planner policies...")
        mobile_policy = self._load_mobile_policy(phase3_config, vec_env)
        planner_policy = self._load_planner_policy(phase3_config, vec_env)
        
        self.mobile_trainer = self._setup_phase3_mobile_trainer(vec_env, phase3_config, mobile_policy)
        self.planner_trainer = self._setup_phase3_planner_trainer(vec_env, phase3_config, planner_policy)
        self.bank_trainer = self._setup_bank_trainer(vec_env, phase3_config)

        self.two_phase_trainer = TwoPhaseTrainer(phase3_config)
        self.two_phase_trainer.mobile_trainer = self.mobile_trainer
        self.two_phase_trainer.planner_trainer = self.planner_trainer
        
        mobile_metrics_logger = MetricsLogger(self.loss_folder_phase3 + "/mobile")
        planner_metrics_logger = MetricsLogger(self.loss_folder_phase3 + "/planner")
        bank_metrics_logger = MetricsLogger(self.loss_folder_phase3 + "/bank")

        self.mobile_trainer.lstm_states = {}
        self.planner_trainer.lstm_states = {}
        self.bank_trainer.lstm_states = {}

        mobile_utility_history = []
        planner_utility_history = []
        bank_utility_history = []
        utility_tolerance = joint_params.get("utility_tolerance", 2.0)
        utility_patience = joint_params.get("utility_patience", 10)

        print(f"Will train phase 3 for {num_joint_episodes} episodes with utility tolerance {utility_tolerance} and patience {utility_patience}.")
        
        phase3_start_time = time.time()

        try:
            print(f"Starting joint training with {num_joint_episodes} episodes.")
            full_start_time = time.time()

            for episode in range(num_joint_episodes):
                print(f"\n--- Joint Training Episode {episode + 1}/{num_joint_episodes} ---")
                episode_start_time = time.time()
                
                vec_env.reset_all(randomize_agent_positions=True)
                self._reset_buffers_for_joint_training()
                
                self._run_joint_training_episode(
                    vec_env,
                    episode,
                    episode_length,
                    tax_period_length,
                    exploration_steps,
                )
                
                mobile_avg_utility, planner_avg_utility, bank_avg_utility = self._update_models_after_episode(
                    episode,
                    mobile_metrics_logger,
                    planner_metrics_logger,
                    bank_metrics_logger,
                    mobile_utility_history,
                    planner_utility_history,
                    bank_utility_history
                )

                planner_utility_history.append(planner_avg_utility)
                mobile_utility_history.append(mobile_avg_utility)
                bank_utility_history.append(bank_avg_utility)
                
                self._save_interim_models_if_needed(episode, num_joint_episodes, 
                                                  mobile_metrics_logger, 
                                                  planner_metrics_logger,
                                                  bank_metrics_logger)
                
                if self._should_stop_early(
                    planner_utility_history,
                    bank_utility_history,
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
                
                episode_start_time = current_time
            
            self._save_final_models()
            
        finally:
            vec_env.close()
            print("Vectorized environment closed.")
            
        phase_3_duration = time.time() - phase3_start_time
        print(f"\nPHASE 3 COMPLETED IN {phase_3_duration:.2f} SECONDS")
        
        return phase_3_duration
    
    def train_three_phase(self):
        print("\n" + "="*50)
        print("Starting Three-Phase Training")
        print("="*50 + "\n")
        
        self.phase_one_and_two()
        
        phase3_duration = self.phase_three()
        
        print(f"\nTotal training completed in {timedelta(seconds=phase3_duration)}")
        
    def _load_mobile_policy(self, config, vec_env):
        mobile_policy = self.two_phase_trainer._create_mobile_policy(config, vec_env)
        mobile_policy.load_state_dict(torch.load(self.mobile_policy_path))
        print(f"Loaded pre-trained mobile agent policy from {self.mobile_policy_path}")
        return mobile_policy
        
    def _load_planner_policy(self, config, vec_env):
        planner_policy = self.two_phase_trainer._create_planner_policy(config, vec_env)
        planner_policy.load_state_dict(torch.load(self.planner_policy_path))
        print(f"Loaded pre-trained planner policy from {self.planner_policy_path}")
        return planner_policy
        
    def _setup_phase3_mobile_trainer(self, vec_env, config, mobile_policy):
        mobile_trainer = self.two_phase_trainer._setup_phase2_mobile_trainer(vec_env, config, mobile_policy)
        return mobile_trainer
        
    def _setup_phase3_planner_trainer(self, vec_env, config, planner_policy):
        planner_trainer = self.two_phase_trainer._setup_planner_trainer(vec_env, config, 0)
        planner_trainer.policy_net = planner_policy
        planner_trainer.policy_net.to(planner_trainer.device)
        planner_trainer._load_obs_stats(self.planner_policy_path)
        return planner_trainer
        
    def _setup_bank_trainer(self, vec_env, config):
        bank_config = config.copy()
        bank_config["bank_agent_training"] = bank_config.get("bank_agent_training", {}).copy()
        
        for env_idx in range(vec_env.total_envs):
            vec_env.pipes[env_idx].send(("get_bank_obs", None))
            try:
                obs = vec_env.pipes[env_idx].recv()
                if obs is not None:
                    input_size = len(obs)
                    break
            except (EOFError, BrokenPipeError):
                continue
        else:
            raise ValueError("Could not get bank observation to determine input size")
            
        bank_policy = BankPolicy(
            config=bank_config,
            input_size=input_size,
            output_size=7
        )
        
        bank_trainer = BankPPOTrainer(vec_env, bank_config, self.use_gpu)
        bank_trainer.policy_net = bank_policy
        bank_trainer.policy_net.to(bank_trainer.device)
        
        return bank_trainer
        
    def _reset_buffers_for_joint_training(self):
        self.mobile_trainer.lstm_states = {}
        self.planner_trainer.lstm_states = {}
        self.bank_trainer.lstm_states = {}
        
        for key in list(self.mobile_trainer.rollout_buffers.keys()):
            self.mobile_trainer.rollout_buffers[key] = []
        
        for key in list(self.planner_trainer.rollout_buffers.keys()):
            self.planner_trainer.rollout_buffers[key] = []
            
        for key in list(self.bank_trainer.rollout_buffers.keys()):
            self.bank_trainer.rollout_buffers[key] = []
            
    def _run_joint_training_episode(self, vec_env, episode, episode_length, tax_period_length, 
                                   exploration_steps):
        random_planner_actions = episode < exploration_steps
        random_bank_actions = episode < exploration_steps
        
        for env_idx in range(vec_env.total_envs):
            vec_env.pipes[env_idx].send(("update_bank_annealing", episode))
            try:
                vec_env.pipes[env_idx].recv()
            except (EOFError, BrokenPipeError):
                continue

        mobile_current_buffers = {env_idx: [] for env_idx in range(vec_env.total_envs)}
        planner_current_buffers = {env_idx: [] for env_idx in range(vec_env.total_envs)}
        bank_current_buffers = {env_idx: [] for env_idx in range(vec_env.total_envs)}
        
        for step in range(episode_length):
            is_tax_year_start = (step > 0) and (step % tax_period_length == 0)

            if is_tax_year_start:
                self._handle_tax_year_start(vec_env, random_planner_actions, planner_current_buffers)

            self._handle_mobile_agent_steps(vec_env, mobile_current_buffers)
            self._handle_bank_step(vec_env, random_bank_actions, bank_current_buffers)
            
            self._update_environment_state(vec_env)
        
        self._process_buffers_after_episode(mobile_current_buffers, planner_current_buffers, bank_current_buffers)
        
    def _handle_bank_step(self, vec_env, random_bank_sampling, bank_current_buffers):
        observations = []
        env_indices = []
        
        for env_idx in range(vec_env.total_envs):
            vec_env.pipes[env_idx].send(("get_bank_obs", None))
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
            flattened_obs = torch.FloatTensor(obs).to(self.bank_trainer.device)
            obs_tensors.append(flattened_obs)
            
        action_masks = []
        for env_idx in env_indices:
            vec_env.pipes[env_idx].send(("get_bank_action_mask", None))
            try:
                mask = vec_env.pipes[env_idx].recv()
                action_masks.append(mask)
            except (EOFError, BrokenPipeError):
                action_masks.append(False)
        
        if not (obs_tensors and any(np.any(mask) for mask in action_masks)):
            return
            
        obs_batch = torch.stack(obs_tensors)
        
        lstm_states_batch = self._get_bank_lstm_states_batch(env_indices)
        
        if random_bank_sampling:
            actions_batch, log_probs_batch, values_batch, lstm_state_out = self._get_random_bank_actions(
                len(obs_tensors),
                lstm_states_batch,
                action_masks
            )
        else:
            actions_batch, log_probs_batch, values_batch, lstm_state_out = self._get_bank_network_actions(
                obs_batch,
                lstm_states_batch,
                action_masks
            )
            
        if lstm_state_out is not None:
            self._update_bank_lstm_states(lstm_state_out, env_indices)
            
        for i, env_idx in enumerate(env_indices):
            if not np.any(action_masks[i]): 
                continue
                
            action = actions_batch[i]
            
            vec_env.pipes[env_idx].send(("get_bank_utility", None))
            try:
                pre_utility = vec_env.pipes[env_idx].recv()
                
                vec_env.pipes[env_idx].send(("bank_step", action))
                post_utility, _ = vec_env.pipes[env_idx].recv()
                
                reward = post_utility - pre_utility
                normalized_reward = self.bank_trainer.normalize_reward(reward)
                
                bank_current_buffers[env_idx].append({
                    "obs": obs_tensors[i].detach(),
                    "action": action,
                    "log_prob": log_probs_batch[i].detach() if isinstance(log_probs_batch, torch.Tensor) 
                              else torch.tensor(log_probs_batch[i], device=self.bank_trainer.device),
                    "value": values_batch[i].detach() if isinstance(values_batch, torch.Tensor) 
                            else torch.tensor(0.0, device=self.bank_trainer.device),
                    "reward": normalized_reward,
                    "utility": post_utility,
                    "original_reward": reward
                })
            except (EOFError, BrokenPipeError):
                continue

    def _handle_tax_year_start(self, vec_env, random_planner_sampling, planner_current_buffers):
        self.two_phase_trainer._handle_tax_year_start(vec_env, random_planner_sampling, planner_current_buffers)

    def _get_bank_lstm_states_batch(self, env_indices):
        h_list, c_list = [], []
        
        for env_idx in env_indices:
            env_key = f"env{env_idx}"
            if env_key in self.bank_trainer.lstm_states:
                h, c = self.bank_trainer.lstm_states[env_key]
                h_list.append(h)
                c_list.append(c)
                
        if h_list and c_list:
            h_batch = torch.cat(h_list, dim=1)
            c_batch = torch.cat(c_list, dim=1)
            return (h_batch, c_batch)
        return None
        
    def _get_random_bank_actions(self, batch_size, lstm_states_batch, action_masks):
        actions_batch = []
        log_probs_batch = []
        
        for mask in action_masks:
            valid_actions = np.where(mask)[0]
            if len(valid_actions) > 0:
                action = int(np.random.choice(valid_actions))
            else:
                action = 0
            actions_batch.append(action)
            log_probs_batch.append(0.0)
            
        values_batch = torch.zeros(batch_size, 1).to(self.bank_trainer.device)
        
        if lstm_states_batch:
            h, c = lstm_states_batch
            lstm_state_out = (torch.zeros_like(h), torch.zeros_like(c))
        else:
            h = torch.zeros(
                self.bank_trainer.policy_net.lstm_num_layers, 
                batch_size, 
                self.bank_trainer.policy_net.lstm_hidden_size, 
                device=self.bank_trainer.device
            )
            c = torch.zeros_like(h)
            lstm_state_out = (h, c)
            
        log_probs_batch = torch.tensor(log_probs_batch, device=self.bank_trainer.device)
        return actions_batch, log_probs_batch, values_batch, lstm_state_out
        
    def _get_bank_network_actions(self, obs_batch, lstm_states_batch, action_masks):
        self.bank_trainer.policy_net.eval()
        with torch.no_grad():
            logits, values_batch, lstm_state_out = self.bank_trainer.policy_net(
                obs_batch, lstm_states_batch
            )
            
            action_masks_tensor = torch.tensor(np.array(action_masks), dtype=torch.bool, device=self.bank_trainer.device)
            masked_logits = logits.clone()
            masked_logits[~action_masks_tensor] = -1e10
            
            probs = torch.nn.functional.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions_tensor = dist.sample()
            log_probs_batch = dist.log_prob(actions_tensor)
            
            actions_batch = [int(action.item()) for action in actions_tensor]
            
        return actions_batch, log_probs_batch, values_batch, lstm_state_out
        
    def _update_bank_lstm_states(self, lstm_state_out, env_indices):
        h_out, c_out = lstm_state_out
        for i, env_idx in enumerate(env_indices):
            env_key = f"env{env_idx}"
            h_env = h_out[:, i:i+1, :].detach()
            c_env = c_out[:, i:i+1, :].detach()
            self.bank_trainer.lstm_states[env_key] = (h_env, c_env)
            
    def _handle_mobile_agent_steps(self, vec_env, mobile_current_buffers):
        self.two_phase_trainer._handle_mobile_agent_steps(vec_env, mobile_current_buffers)
        
    def _update_environment_state(self, vec_env):
        self.two_phase_trainer._update_environment_state(vec_env)
        
    def _process_buffers_after_episode(self, mobile_current_buffers, planner_current_buffers, bank_current_buffers):
        self.two_phase_trainer._process_buffers_after_episode(mobile_current_buffers, planner_current_buffers)
        
        for env_idx, buffer in bank_current_buffers.items():
            if buffer:
                self.bank_trainer.rollout_buffers[f"env{env_idx}"] = buffer
                
    def _update_models_after_episode(self, episode, mobile_metrics_logger, planner_metrics_logger, 
                                    bank_metrics_logger, mobile_utility_history, planner_utility_history,
                                    bank_utility_history):
        print("Updating mobile agent policies...")
        mobile_avg_utility = self.mobile_trainer.update_agents(mobile_metrics_logger)
        print(f"Mobile agents average utility: {mobile_avg_utility:.4f}")

        print("Updating planner policy...")
        planner_avg_utility = self.planner_trainer.update_planner(planner_metrics_logger)
        print(f"Planner average utility: {planner_avg_utility:.4f}")

        print("Updating bank policy...")
        bank_avg_utility = self.bank_trainer.update_bank(bank_metrics_logger)
        print(f"Bank average utility: {bank_avg_utility:.4f}")

        return mobile_avg_utility, planner_avg_utility, bank_avg_utility
        
    def _save_interim_models_if_needed(self, episode, num_joint_episodes, mobile_metrics_logger, 
                                      planner_metrics_logger, bank_metrics_logger):
        if episode % 1 == 0 or episode == num_joint_episodes - 1:
            interim_mobile_path = os.path.join(
                self.network_folder, 
                f"mobile_agents-phase_3-n_agents={self.mobile_trainer.vec_env.n_agents}-experiment_name={self.experiment_name}_PARTIAL.pth"
            )
            torch.save(self.mobile_trainer.shared_policy.state_dict(), interim_mobile_path)
            
            interim_planner_path = os.path.join(
                self.network_folder, 
                f"planner_agent-phase_3-n_agents={self.planner_trainer.vec_env.n_agents}-experiment_name={self.experiment_name}_PARTIAL.pth"
            )
            torch.save(self.planner_trainer.policy_net.state_dict(), interim_planner_path)
            self.planner_trainer._save_obs_stats(interim_planner_path)

            interim_bank_path = os.path.join(
                self.network_folder, 
                f"bank_agent-phase_3-n_agents={self.bank_trainer.vec_env.n_agents}-experiment_name={self.experiment_name}_PARTIAL.pth"
            )
            torch.save(self.bank_trainer.policy_net.state_dict(), interim_bank_path)
            
            mobile_metrics_logger.plot_metrics()
            planner_metrics_logger.plot_metrics()
            bank_metrics_logger.plot_metrics()
            
            print(f"Saved interim models at episode {episode+1}")
            
    def _should_stop_early(self, planner_utility_history, bank_utility_history, mobile_utility_history, utility_patience, 
                          utility_tolerance, episode, exploration_steps):
        if (len(planner_utility_history) > utility_patience and 
            len(bank_utility_history) > utility_patience and 
            len(mobile_utility_history) > utility_patience and 
            episode >= exploration_steps):
            
            recent_planner_utilities = planner_utility_history[-utility_patience:]
            recent_bank_utilities = bank_utility_history[-utility_patience:]
            recent_mobile_utilities = mobile_utility_history[-utility_patience:]
            
            planner_max_diff = max([abs(recent_planner_utilities[i] - recent_planner_utilities[i-1]) 
                                  for i in range(1, len(recent_planner_utilities))])
            bank_max_diff = max([abs(recent_bank_utilities[i] - recent_bank_utilities[i-1]) 
                               for i in range(1, len(recent_bank_utilities))])
            mobile_max_diff = max([abs(recent_mobile_utilities[i] - recent_mobile_utilities[i-1]) 
                                 for i in range(1, len(recent_mobile_utilities))])
            
            if (planner_max_diff < utility_tolerance and 
                bank_max_diff < utility_tolerance and 
                mobile_max_diff < utility_tolerance):
                print(f"Early stopping triggered: all utilities stable (±{utility_tolerance}) for {utility_patience} episodes")
                print(f"Final planner average utility: {planner_utility_history[-1]:.4f}")
                print(f"Final bank average utility: {bank_utility_history[-1]:.4f}")
                print(f"Final mobile average utility: {mobile_utility_history[-1]:.4f}")
                return True
        return False
        
    def _update_entropy_weights(self, episode, exploration_steps):
        if episode >= exploration_steps:
            self.mobile_trainer.entropy_weight = self.mobile_trainer.calculate_decaying_entropy_weight(episode - exploration_steps)
            self.planner_trainer.entropy_weight = self.planner_trainer.calculate_decaying_entropy_weight(episode - exploration_steps)
            self.bank_trainer.entropy_weight = self.bank_trainer.calculate_decaying_entropy_weight(episode - exploration_steps)
        else:
            self.mobile_trainer.entropy_weight = self.mobile_trainer.mat_config.get("entropy_weight", 0.05)
            self.planner_trainer.entropy_weight = self.planner_trainer.planner_config.get("entropy_weight", 0.05)
            self.bank_trainer.entropy_weight = self.bank_trainer.bank_config.get("entropy_weight", 0.05)
        
    def _save_final_models(self):
        final_mobile_path = os.path.join(
            self.network_folder, 
            f"mobile_agents-phase_3-n_agents={self.mobile_trainer.vec_env.n_agents}-experiment_name={self.experiment_name}_COMPLETE.pth"
        )
        torch.save(self.mobile_trainer.shared_policy.state_dict(), final_mobile_path)
        
        final_planner_path = os.path.join(
            self.network_folder, 
            f"planner_agent-phase_3-n_agents={self.planner_trainer.vec_env.n_agents}-experiment_name={self.experiment_name}_COMPLETE.pth"
        )
        torch.save(self.planner_trainer.policy_net.state_dict(), final_planner_path)
        self.planner_trainer._save_obs_stats(final_planner_path)

        final_bank_path = os.path.join(
            self.network_folder, 
            f"bank_agent-phase_3-n_agents={self.bank_trainer.vec_env.n_agents}-experiment_name={self.experiment_name}_COMPLETE.pth"
        )
        torch.save(self.bank_trainer.policy_net.state_dict(), final_bank_path)

        for phase in ["mobile_agents", "planner_agent", "bank_agent"]:
            interim_path = os.path.join(
                self.network_folder, 
                f"{phase}-phase_3-n_agents={self.mobile_trainer.vec_env.n_agents}-experiment_name={self.experiment_name}_PARTIAL.pth"
            )
            if os.path.exists(interim_path):
                os.remove(interim_path)
        
        print("Joint training completed!")
        print(f"Saved final mobile agent policy to {final_mobile_path}")
        print(f"Saved final planner policy to {final_planner_path}")
        print(f"Saved final bank policy to {final_bank_path}") 