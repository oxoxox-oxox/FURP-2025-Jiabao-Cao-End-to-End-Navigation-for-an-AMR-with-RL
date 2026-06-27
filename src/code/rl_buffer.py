"""
Rollout buffer for PPO.

Fixed-size buffer that stores (obs, action, reward, value, log_prob, done) from
multiple parallel environments. Supports GAE (Generalized Advantage Estimation)
with proper handling of truncated (timeout) episodes.
"""

import numpy as np
import torch


class RolloutBuffer:
    """
    Fixed-size buffer for storing rollout data from multiple parallel envs.

    Supports:
    - GAE (Generalized Advantage Estimation)
    - Proper handling of truncated episodes (timeout) with bootstrap values
    - Mini-batch sampling with advantage normalization
    """

    def __init__(self, n_steps, n_envs, obs_dim, act_dim, gamma, gae_lambda, device='cuda'):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # Storage arrays shaped (n_steps, n_envs, ...)
        self.observations = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs, act_dim), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)

        # Bootstrap values for truncated (timeout) episodes:
        # When an episode times out, the env auto-resets to a new episode.
        # We store V(s_timeout) computed BEFORE the reset, so GAE can
        # correctly bootstrap from it instead of V(s_reset).
        self.timeout_bootstrap = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.has_timeout = np.zeros((n_steps, n_envs), dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)

        self.ptr = 0

    def store(self, step, obs, actions, rewards, values, log_probs,
              dones, truncateds, timeout_bootstrap_vals):
        """
        Store one timestep of data from all envs.

        Args:
            step: current step index [0, n_steps-1]
            obs: (n_envs, obs_dim)
            actions: (n_envs, act_dim)
            rewards: (n_envs,)
            values: (n_envs,)
            log_probs: (n_envs,)
            dones: (n_envs,) -- terminal flags (collision/success)
            truncateds: (n_envs,) -- timeout flags
            timeout_bootstrap_vals: (n_envs,) -- V(timeout_state) or 0
        """
        self.observations[step] = obs
        self.actions[step] = actions
        self.rewards[step] = rewards
        self.values[step] = values
        self.log_probs[step] = log_probs
        self.dones[step] = dones.astype(np.float32)
        self.has_timeout[step] = truncateds.astype(np.float32)
        self.timeout_bootstrap[step] = timeout_bootstrap_vals
        self.ptr = step + 1

    def compute_gae(self, last_values):
        """
        Compute GAE advantages and returns.

        For terminal episodes (done=1):  mask=0, no bootstrap
        For truncated episodes (timeout): mask=1, bootstrap from pre-reset V(timeout_state)

        Args:
            last_values: (n_envs,) -- V(last_obs) from the final observation after rollout
        """
        gae = np.zeros(self.n_envs, dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            # Determine next_value for each env
            if t == self.n_steps - 1:
                next_value = last_values.copy()
            else:
                next_value = self.values[t + 1].copy()

            # Override: for timeout episodes, use the stored pre-reset value
            timeout_mask = self.has_timeout[t] > 0.5
            next_value[timeout_mask] = self.timeout_bootstrap[t][timeout_mask]

            # Terminal mask: 1 = non-terminal (bootstrap), 0 = terminal (no bootstrap)
            mask = 1.0 - self.dones[t]

            delta = (self.rewards[t] + self.gamma * next_value * mask
                     - self.values[t])
            gae = delta + self.gamma * self.gae_lambda * mask * gae

            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get_batches(self, batch_size):
        """
        Yield mini-batches of training data with advantage normalization.

        Flattens (n_steps, n_envs) -> shuffled mini-batches.

        Yields:
            obs_batch, actions_batch, old_log_probs_batch,
            advantages_batch, returns_batch
        """
        total = self.ptr * self.n_envs
        batch_size = min(batch_size, total)

        # Flatten
        obs = self.observations[:self.ptr].reshape(total, self.obs_dim)
        actions = self.actions[:self.ptr].reshape(total, self.act_dim)
        log_probs = self.log_probs[:self.ptr].reshape(total)
        advantages = self.advantages[:self.ptr].reshape(total)
        returns = self.returns[:self.ptr].reshape(total)

        # Normalize advantages (global mean/std across all rollout data)
        adv_mean = advantages.mean()
        if len(advantages) > 1:
            adv_std = advantages.std() + 1e-8
        else:
            adv_std = 1.0
        advantages = (advantages - adv_mean) / adv_std
        # Safety clamp: prevent extreme outliers when std is tiny
        # (e.g., all rewards nearly identical → normalization amplifies noise)
        advantages = np.clip(advantages, -5.0, 5.0)

        # Normalize returns to prevent value loss from dominating gradients.
        ret_mean = returns.mean()
        if len(returns) > 1:
            ret_std = returns.std() + 1e-8
        else:
            ret_std = 1.0
        returns = (returns - ret_mean) / ret_std
        returns = np.clip(returns, -5.0, 5.0)

        # Shuffle
        indices = np.random.permutation(total)

        for start in range(0, total, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            yield (
                torch.FloatTensor(obs[batch_idx]).to(self.device),
                torch.FloatTensor(actions[batch_idx]).to(self.device),
                torch.FloatTensor(log_probs[batch_idx]).unsqueeze(-1).to(self.device),
                torch.FloatTensor(advantages[batch_idx]).unsqueeze(-1).to(self.device),
                torch.FloatTensor(returns[batch_idx]).unsqueeze(-1).to(self.device),
            )

    def clear(self):
        """Reset buffer pointer (data is overwritten, not zeroed)."""
        self.ptr = 0
