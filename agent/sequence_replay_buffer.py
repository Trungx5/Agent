"""
agent/sequence_replay_buffer.py

Replay buffer dành cho LSTM-DQN.
Mỗi transition lưu chuỗi quan sát thay vì state phẳng:
    seq       : (seq_len, feature_dim)  – chuỗi hiện tại
    action    : int
    reward    : float
    next_seq  : (seq_len, feature_dim)  – chuỗi kế tiếp
    done      : float
"""

import numpy as np
import random
from collections import deque


class SequenceReplayBuffer:
    """
    Fixed-size circular buffer for sequence-based (LSTM) transitions.
    Stores (seq, action, reward, next_seq, done).
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        seq:      np.ndarray,   # (seq_len, feature_dim)
        action:   int,
        reward:   float,
        next_seq: np.ndarray,   # (seq_len, feature_dim)
        done:     float,
    ) -> None:
        self.buffer.append((
            np.array(seq,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_seq, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        """
        Returns:
            seqs      : (B, seq_len, feature_dim)
            actions   : (B,)
            rewards   : (B,)
            next_seqs : (B, seq_len, feature_dim)
            dones     : (B,)
        """
        batch = random.sample(self.buffer, batch_size)
        seqs, actions, rewards, next_seqs, dones = zip(*batch)
        return (
            np.array(seqs,      dtype=np.float32),
            np.array(actions,   dtype=np.int64),
            np.array(rewards,   dtype=np.float32),
            np.array(next_seqs, dtype=np.float32),
            np.array(dones,     dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
