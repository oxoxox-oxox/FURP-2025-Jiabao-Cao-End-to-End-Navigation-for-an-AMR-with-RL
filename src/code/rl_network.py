"""
Actor-Critic network for PPO.

MLP-based diagonal Gaussian policy with orthogonal initialization.
- Actor: outputs action mean + state-independent log_std
- Critic: outputs scalar state value V(s)
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


class ActorCritic(nn.Module):
    """
    MLP Actor-Critic with orthogonal initialization.

    Actor: diagonal Gaussian policy (mean + state-independent log_std)
    Critic: scalar value function V(s)

    Architecture: obs -> Linear(obs_dim, 64) -> Tanh -> Linear(64, 64) -> Tanh
                            |-> Linear(64, act_dim) -> mean
                            |-> Linear(64, 1) -> value
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64),
                 activation=nn.Tanh, log_std_init=0.0):
        super().__init__()

        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h
        self.feature_extractor = nn.Sequential(*layers)

        # Actor head -> action mean
        self.actor_mean = nn.Linear(in_dim, act_dim)

        # Learnable log standard deviation (state-independent)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

        # Critic head -> state value (deeper: extra 64-dim hidden layer for
        # better value prediction, reducing value loss and improving explained_variance)
        self.critic = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Orthogonal initialization
        self._init_weights()

    def _init_weights(self):
        """Orthogonal weight initialization (matching SB3 defaults)."""
        for m in self.feature_extractor:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # Actor mean: small initial gain for stable early exploration
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)

        # Critic (now nn.Sequential): orthogonal init with gain=1.0
        for m in self.critic:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs):
        """
        Forward pass.

        Args:
            obs: (batch_size, obs_dim) tensor

        Returns:
            mean: (batch_size, act_dim) -- action mean
            log_std: (batch_size, act_dim) -- log std (broadcast from parameter)
            value: (batch_size, 1) -- state value
        """
        features = self.feature_extractor(obs)
        mean = self.actor_mean(features)
        log_std = self.log_std.expand_as(mean)
        value = self.critic(features)
        return mean, log_std, value

    def get_action(self, obs, deterministic=False):
        """
        Sample action from the policy.

        Args:
            obs: (batch_size, obs_dim) tensor
            deterministic: if True, return mean (no exploration)

        Returns:
            action: (batch_size, act_dim)
            log_prob: (batch_size, 1) -- sum of log-probs across action dims
            value: (batch_size, 1)
        """
        mean, log_std, value = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.sample()

        # Sum log-prob across action dimensions
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, log_prob, value

    def evaluate(self, obs, action):
        """
        Evaluate log-prob, entropy, and value for given (obs, action).

        Used during PPO update to recompute log-probs under the current policy.

        Args:
            obs: (batch_size, obs_dim) tensor
            action: (batch_size, act_dim) tensor

        Returns:
            log_prob: (batch_size, 1)
            entropy: (batch_size, 1)
            value: (batch_size, 1)
        """
        mean, log_std, value = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)

        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, entropy, value
