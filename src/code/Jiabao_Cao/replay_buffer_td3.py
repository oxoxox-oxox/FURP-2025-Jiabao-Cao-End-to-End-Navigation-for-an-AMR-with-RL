"""
Off-policy replay buffer for TD3 (CNNTD3).

Fixed-capacity deque-based buffer with uniform random sampling.
Matches the sample_batch interface expected by CNNTD3.train().
"""

from collections import deque
import numpy as np
import random


class ReplayBuffer:
    """
    Standard experience replay buffer for off-policy RL.

    Stores (state, action, reward, done, next_state) tuples.
    Uniform random sampling — no prioritization.
    """

    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque(maxlen=buffer_size)
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        """
        Add a transition to the buffer.

        Args:
            s:  state  (np.ndarray)
            a:  action (np.ndarray)
            r:  reward (float)
            t:  done   (int/float, 1=terminal)
            s2: next_state (np.ndarray)
        """
        self.buffer.append((s, a, r, t, s2))
        self.count = min(self.count + 1, self.buffer_size)

    def sample_batch(self, batch_size):
        """
        Sample a random batch of transitions.

        Args:
            batch_size: number of transitions to sample

        Returns:
            (s_batch, a_batch, r_batch, t_batch, s2_batch) — each as np.ndarray
        """
        batch = random.sample(self.buffer, min(batch_size, self.count))
        s_batch  = np.array([_[0] for _ in batch], dtype=np.float32)
        a_batch  = np.array([_[1] for _ in batch], dtype=np.float32)
        r_batch  = np.array([_[2] for _ in batch], dtype=np.float32)
        t_batch  = np.array([_[3] for _ in batch], dtype=np.float32)
        s2_batch = np.array([_[4] for _ in batch], dtype=np.float32)
        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def size(self):
        """Return number of stored transitions."""
        return self.count

    def clear(self):
        """Empty the buffer."""
        self.buffer.clear()
        self.count = 0
