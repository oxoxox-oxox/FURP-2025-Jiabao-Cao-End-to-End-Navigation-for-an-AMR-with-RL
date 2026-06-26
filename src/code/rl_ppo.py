"""
Proximal Policy Optimization (PPO) algorithm.

Pure PyTorch implementation for continuous action spaces.
Works with any Gymnasium-compatible environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import copy
import time
from collections import defaultdict

from rl_network import ActorCritic
from rl_buffer import RolloutBuffer


class PPO:
    """
    Proximal Policy Optimization (PPO) with clipped objective.

    Suitable for continuous action spaces using a diagonal Gaussian policy.

    Usage:
        model = PPO(env=IrSimNavEnv, env_kwargs={'render_mode': None}, n_envs=4)
        model.learn(total_timesteps=500_000)
        model.save('model.pt')

        # Inference
        model = PPO.load('model.pt')
        action, _ = model.predict(obs)
    """

    def __init__(
        self,
        env,                        # Gymnasium Env class or instance
        env_kwargs=None,            # kwargs passed to env() if callable
        n_envs=1,                   # number of parallel environments
        learning_rate=3e-4,         # actor LR
        critic_lr=None,             # critic LR (defaults to learning_rate if None)
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,             # optional early stopping KL threshold
        device='cpu',
        tensorboard_log=None,
        seed=None,
    ):
        # Hyperparameters
        self.n_envs = n_envs
        self.lr = learning_rate
        self.critic_lr = critic_lr if critic_lr is not None else learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        # Device
        if device == 'cuda' and not torch.cuda.is_available():
            print("[PPO] CUDA not available, falling back to CPU")
            device = 'cpu'
        self.device = torch.device(device)

        # Seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Create environments
        self.env_kwargs = env_kwargs or {}
        self._env_fn = env  # store for save/load
        self.envs = self._create_envs(env, self.env_kwargs)

        # Infer dimensions from first env
        sample_env = self.envs[0]
        self.obs_dim = sample_env.observation_space.shape[0]
        self.act_dim = sample_env.action_space.shape[0]
        self.act_low = sample_env.action_space.low.copy()
        self.act_high = sample_env.action_space.high.copy()

        # Actor-Critic network
        self.policy = ActorCritic(self.obs_dim, self.act_dim).to(self.device)

        # Separate optimizer parameter groups: critic LR > actor LR
        # because the critic needs to fit faster to keep up with policy changes.
        actor_params = []
        critic_params = []
        for name, param in self.policy.named_parameters():
            if 'critic' in name:
                critic_params.append(param)
            else:
                actor_params.append(param)

        self.optimizer = optim.Adam([
            {'params': actor_params,  'lr': self.lr},
            {'params': critic_params, 'lr': self.critic_lr},
        ], eps=1e-5)

        # Rollout buffer
        self.buffer = RolloutBuffer(
            n_steps, n_envs, self.obs_dim, self.act_dim,
            gamma, gae_lambda, self.device
        )

        # Logging
        self.writer = None
        if tensorboard_log:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(tensorboard_log)

        # State
        self._current_obs = None
        self._total_steps = 0
        self._episode_buffer = defaultdict(list)  # per-env episode stats

    # ---- Environment management ----

    def _create_envs(self, env, env_kwargs):
        """Create n_envs environment instances."""
        envs = []
        for _ in range(self.n_envs):
            if callable(env):
                kwargs = copy.deepcopy(env_kwargs)
                e = env(**kwargs)
            elif self.n_envs > 1:
                e = copy.deepcopy(env)
            else:
                e = env
            envs.append(e)
        return envs

    def _reset_all_envs(self):
        """Reset all envs, return stacked observations."""
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        return np.stack(obs_list).astype(np.float32)

    def _step_envs(self, actions):
        """
        Step all environments.

        Handles episode termination with auto-reset.
        When an episode is truncated (timeout), computes the bootstrap value
        BEFORE resetting, so GAE uses the correct next-state value.

        Args:
            actions: (n_envs, act_dim) numpy array

        Returns:
            next_obs: (n_envs, obs_dim) -- next obs (or reset obs if episode ended)
            rewards: (n_envs,)
            dones: (n_envs,) -- terminal flags (collision/success)
            truncateds: (n_envs,) -- timeout flags
            timeout_bootstrap_vals: (n_envs,) -- V(timeout_state) or 0
            infos: list of info dicts
        """
        next_obs = np.zeros((self.n_envs, self.obs_dim), dtype=np.float32)
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=np.float32)
        truncateds = np.zeros(self.n_envs, dtype=np.float32)
        timeout_bootstrap_vals = np.zeros(self.n_envs, dtype=np.float32)
        infos = []

        for i, env in enumerate(self.envs):
            obs_i, reward, done, truncated, info = env.step(actions[i])

            rewards[i] = reward
            dones[i] = float(done)
            truncateds[i] = float(truncated)
            infos.append(info)

            if done or truncated:
                # Track episode stats
                self._episode_buffer[i].append({
                    'reward': reward,
                    'result': info.get('result', 'unknown'),
                    'length': info.get('episode_length', 0),
                })

                # For truncated episodes: compute bootstrap value BEFORE reset
                if truncated and not done:
                    with torch.no_grad():
                        obs_t = torch.FloatTensor(obs_i).unsqueeze(0).to(self.device)
                        _, _, val = self.policy.forward(obs_t)
                        timeout_bootstrap_vals[i] = val.item()

                # Auto-reset for next step
                obs_i, _ = env.reset()

            next_obs[i] = obs_i

        return next_obs, rewards, dones, truncateds, timeout_bootstrap_vals, infos

    # ---- Rollout collection ----

    def collect_rollouts(self):
        """
        Collect n_steps of experience across all parallel envs.

        Fills the RolloutBuffer with (obs, action, reward, value, log_prob, done, timeout).
        """
        if self._current_obs is None:
            self._current_obs = self._reset_all_envs()

        for step in range(self.n_steps):
            # Get actions from current policy
            obs_tensor = torch.FloatTensor(self._current_obs).to(self.device)
            with torch.no_grad():
                actions_tensor, log_probs_tensor, values_tensor = \
                    self.policy.get_action(obs_tensor)

            actions = actions_tensor.cpu().numpy()
            log_probs = log_probs_tensor.cpu().numpy().flatten()
            values = values_tensor.cpu().numpy().flatten()

            # Clip actions to valid range
            actions = np.clip(actions, self.act_low, self.act_high)

            # Step all envs
            next_obs, rewards, dones, truncateds, timeout_bootstrap_vals, infos = \
                self._step_envs(actions)

            # Store transition
            self.buffer.store(
                step, self._current_obs, actions, rewards, values, log_probs,
                dones, truncateds, timeout_bootstrap_vals
            )

            self._current_obs = next_obs
            self._total_steps += self.n_envs

    # ---- PPO Update ----

    def train(self):
        """
        Perform PPO update on the collected rollout data.

        Returns:
            dict with training metrics (policy_loss, value_loss, entropy, approx_kl)
        """
        # Compute last value for GAE bootstrapping
        obs_tensor = torch.FloatTensor(self._current_obs).to(self.device)
        with torch.no_grad():
            _, _, last_values_tensor = self.policy.forward(obs_tensor)
        last_values = last_values_tensor.cpu().numpy().flatten()

        # GAE: compute advantages and returns
        self.buffer.compute_gae(last_values)

        # PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        n_batches = 0

        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                obs_b, act_b, old_logp_b, adv_b, ret_b = batch

                # Evaluate current policy on stored data
                new_logp, entropy, values = self.policy.evaluate(obs_b, act_b)

                # ---- PPO Clipped Surrogate Loss ----
                ratio = torch.exp(new_logp - old_logp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range,
                                    1.0 + self.clip_range) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # ---- Value Loss (MSE) ----
                value_loss = nn.functional.mse_loss(values, ret_b)

                # ---- Entropy Bonus ----
                entropy_loss = -entropy.mean()

                # ---- Total Loss ----
                loss = (policy_loss
                        + self.vf_coef * value_loss
                        + self.ent_coef * entropy_loss)

                # NaN check
                if torch.isnan(loss):
                    print("[PPO] WARNING: NaN loss detected, skipping batch")
                    continue

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_approx_kl += (old_logp_b - new_logp).mean().item()
                n_batches += 1

            # Optional early stopping by KL divergence
            if self.target_kl is not None:
                avg_kl = total_approx_kl / max(n_batches, 1)
                if avg_kl > self.target_kl:
                    break

        # Compute explained variance (MUST be before buffer.clear(),
        # otherwise returns/values arrays are empty and var() produces NaN)
        explained_var = self._explained_variance()

        # Clear buffer for next rollout
        self.buffer.clear()

        n = max(n_batches, 1)
        metrics = {
            'policy_loss': total_policy_loss / n,
            'value_loss': total_value_loss / n,
            'entropy': total_entropy / n,
            'approx_kl': total_approx_kl / n,
            'explained_variance': explained_var,
        }
        return metrics

    def _explained_variance(self):
        """EV = 1 - Var[returns - values] / Var[returns]. Measures value function fit."""
        returns = self.buffer.returns[:self.buffer.ptr].flatten()
        values = self.buffer.values[:self.buffer.ptr].flatten()

        # Need at least 2 samples for sample variance (ddof=1).
        # With 0 or 1 samples, variance is undefined → return NaN.
        if len(returns) <= 1:
            return float('nan')

        var_returns = returns.var()
        if var_returns < 1e-10:
            return float('nan')
        return 1.0 - np.var(returns - values) / var_returns

    # ---- Evaluation ----

    def _evaluate(self, eval_env, n_eval_episodes=10):
        """
        Run deterministic evaluation episodes.

        Args:
            eval_env: Gymnasium environment
            n_eval_episodes: number of episodes to run

        Returns:
            mean_reward, success_rate
        """
        episode_rewards = []
        successes = 0

        for _ in range(n_eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            truncated = False
            ep_reward = 0.0

            while not (done or truncated):
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = eval_env.step(action)
                ep_reward += reward

            episode_rewards.append(ep_reward)
            if info.get('result') == 'success':
                successes += 1

        mean_reward = np.mean(episode_rewards)
        success_rate = successes / n_eval_episodes
        return mean_reward, success_rate

    # ---- Main Training Loop ----

    def learn(self, total_timesteps,
              eval_env=None,
              eval_freq=5000,
              n_eval_episodes=10,
              best_model_save_path='./models/',
              tb_log_dir=None):
        """
        Run the main PPO training loop.

        Args:
            total_timesteps: total environment steps to train for
            eval_env: optional Gymnasium env for periodic evaluation
            eval_freq: evaluate every N timesteps
            n_eval_episodes: number of episodes per evaluation
            best_model_save_path: directory to save the best model
            tb_log_dir: TensorBoard log directory (uses self.writer if set in __init__)
        """
        # Override TensorBoard writer if tb_log_dir is provided
        if tb_log_dir and self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(tb_log_dir)

        best_mean_reward = -float('inf')
        iteration = 0
        timesteps_so_far = 0

        print(f"[PPO] Starting training: {total_timesteps} timesteps, "
              f"{self.n_envs} envs, device={self.device}")
        t_start = time.time()

        while timesteps_so_far < total_timesteps:
            # Collect experience
            rollout_start = time.time()
            self.collect_rollouts()
            timesteps_so_far += self.n_steps * self.n_envs
            iteration += 1
            rollout_time = time.time() - rollout_start

            # PPO update
            train_start = time.time()
            train_info = self.train()
            train_time = time.time() - train_start

            # Logging
            fps = (self.n_steps * self.n_envs) / max(rollout_time, 0.001)
            if iteration % 5 == 0 or iteration == 1:
                print(f"[PPO] iter={iteration:4d} | "
                      f"steps={timesteps_so_far:8d} | "
                      f"fps={fps:6.0f} | "
                      f"p_loss={train_info['policy_loss']:7.4f} | "
                      f"v_loss={train_info['value_loss']:7.4f} | "
                      f"ent={train_info['entropy']:6.4f} | "
                      f"kl={train_info['approx_kl']:6.4f} | "
                      f"ev={train_info['explained_variance']:5.2f}")

            if self.writer is not None:
                self.writer.add_scalar('train/policy_loss',
                                       train_info['policy_loss'], timesteps_so_far)
                self.writer.add_scalar('train/value_loss',
                                       train_info['value_loss'], timesteps_so_far)
                self.writer.add_scalar('train/entropy',
                                       train_info['entropy'], timesteps_so_far)
                self.writer.add_scalar('train/approx_kl',
                                       train_info['approx_kl'], timesteps_so_far)
                self.writer.add_scalar('train/explained_variance',
                                       train_info['explained_variance'], timesteps_so_far)
                self.writer.add_scalar('train/fps', fps, timesteps_so_far)

            # Periodic evaluation
            if eval_env is not None and timesteps_so_far % eval_freq < self.n_steps * self.n_envs:
                mean_reward, success_rate = self._evaluate(eval_env, n_eval_episodes)
                print(f"[PPO] Eval  | steps={timesteps_so_far:8d} | "
                      f"mean_reward={mean_reward:7.2f} | "
                      f"success_rate={success_rate:.2f}")

                if self.writer is not None:
                    self.writer.add_scalar('eval/mean_reward', mean_reward,
                                           timesteps_so_far)
                    self.writer.add_scalar('eval/success_rate', success_rate,
                                           timesteps_so_far)

                # Save best model
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    if best_model_save_path:
                        os.makedirs(best_model_save_path, exist_ok=True)
                        save_path = os.path.join(best_model_save_path,
                                                 'best_model.pt')
                        self.save(save_path)
                        print(f"[PPO] New best model saved -> {save_path} "
                              f"(reward={best_mean_reward:.2f})")

        # Training complete
        elapsed = time.time() - t_start
        print(f"[PPO] Training complete: {total_timesteps} steps in "
              f"{elapsed:.0f}s ({elapsed/60:.1f}min)")

        if self.writer is not None:
            self.writer.close()
            self.writer = None

    # ---- Save / Load ----

    def save(self, filepath):
        """
        Save model weights and hyperparameters.

        Args:
            filepath: path to .pt file
        """
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'act_low': self.act_low,
            'act_high': self.act_high,
            'hyperparams': {
                'n_envs': self.n_envs,
                'lr': self.lr,
                'critic_lr': self.critic_lr,
                'n_steps': self.n_steps,
                'batch_size': self.batch_size,
                'n_epochs': self.n_epochs,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_range': self.clip_range,
                'ent_coef': self.ent_coef,
                'vf_coef': self.vf_coef,
                'max_grad_norm': self.max_grad_norm,
            },
            'total_steps': self._total_steps,
        }
        torch.save(checkpoint, filepath)
        print(f"[PPO] Model saved to {filepath}")

    @classmethod
    def load(cls, filepath, device='auto', env=None):
        """
        Load a saved PPO model.

        Args:
            filepath: path to .pt checkpoint
            device: 'auto', 'cpu', or 'cuda'
            env: optional environment (for resuming training)

        Returns:
            PPO instance ready for inference or continued training
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(filepath, map_location=device,
                                weights_only=False)

        # Create a minimal instance
        # Use a dummy env if none provided
        if env is None:
            # Create a dummy callable that returns a mock to satisfy __init__
            import gymnasium as gym
            dummy_env = gym.Env  # will be replaced below
            # We need to bypass __init__ and manually set attributes
            model = cls.__new__(cls)
            model._env_fn = None
            model.envs = []
            model.env_kwargs = {}
            model.n_envs = checkpoint['hyperparams']['n_envs']
        else:
            model = cls.__new__(cls)
            model._env_fn = env
            model.envs = [env] if not isinstance(env, list) else env
            model.env_kwargs = {}
            model.n_envs = len(model.envs)

        # Restore dimensions and bounds
        model.obs_dim = checkpoint['obs_dim']
        model.act_dim = checkpoint['act_dim']
        model.act_low = checkpoint['act_low']
        model.act_high = checkpoint['act_high']

        # Restore hyperparameters
        hp = checkpoint['hyperparams']
        model.lr = hp['lr']
        model.critic_lr = hp.get('critic_lr', hp['lr'])  # backward-compat
        model.n_steps = hp['n_steps']
        model.batch_size = hp['batch_size']
        model.n_epochs = hp['n_epochs']
        model.gamma = hp['gamma']
        model.gae_lambda = hp['gae_lambda']
        model.clip_range = hp['clip_range']
        model.ent_coef = hp['ent_coef']
        model.vf_coef = hp['vf_coef']
        model.max_grad_norm = hp['max_grad_norm']
        model.target_kl = None

        # Device
        model.device = torch.device(device)

        # Rebuild policy and load weights
        model.policy = ActorCritic(model.obs_dim, model.act_dim).to(model.device)
        model.policy.load_state_dict(checkpoint['policy_state_dict'])
        model.policy.eval()  # inference mode by default

        # Rebuild optimizer with separate actor/critic LRs
        actor_params = []
        critic_params = []
        for name, param in model.policy.named_parameters():
            if 'critic' in name:
                critic_params.append(param)
            else:
                actor_params.append(param)
        model.optimizer = optim.Adam([
            {'params': actor_params,  'lr': model.lr},
            {'params': critic_params, 'lr': model.critic_lr},
        ], eps=1e-5)
        if 'optimizer_state_dict' in checkpoint:
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Buffer (not restored -- only needed for training)
        model.buffer = RolloutBuffer(
            model.n_steps, model.n_envs or 1, model.obs_dim, model.act_dim,
            model.gamma, model.gae_lambda, model.device
        )

        # State
        model._current_obs = None
        model._total_steps = checkpoint.get('total_steps', 0)
        model._episode_buffer = defaultdict(list)
        model.writer = None

        print(f"[PPO] Model loaded from {filepath} (device={model.device}, "
              f"trained_steps={model._total_steps})")
        return model

    # ---- Inference ----

    def predict(self, observation, deterministic=True):
        """
        Predict action for a single observation.

        Args:
            observation: (obs_dim,) numpy array or (batch, obs_dim)
            deterministic: if True, use the mean action (no exploration)

        Returns:
            action: (act_dim,) or (batch, act_dim) numpy array
            _states: None (for compatibility with SB3 API)
        """
        single_input = observation.ndim == 1

        if single_input:
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        else:
            obs_tensor = torch.FloatTensor(observation).to(self.device)

        with torch.no_grad():
            action, _, _ = self.policy.get_action(obs_tensor, deterministic=deterministic)

        action_np = action.cpu().numpy()
        action_np = np.clip(action_np, self.act_low, self.act_high)

        if single_input:
            action_np = action_np[0]

        return action_np, None  # None for hidden state (MLP has none)
