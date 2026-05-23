from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional

from agent.replay_buffer import ReplayBuffer


# ── Neural Network ────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """MLP: state_dim → hidden → hidden → action_dim (Q-values)."""

    def __init__(self, state_dim: int = 3, action_dim: int = 3, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """Dueling MLP: shared trunk + value/advantage heads."""

    def __init__(self, state_dim: int = 3, action_dim: int = 3, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + (a - a.mean(dim=1, keepdim=True))


# ── DQN Agent ─────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(
        self,
        state_dim:        int   = 6,   # (E, Q, H, ΔH, Health, Forecast)
        action_dim:       int   = 3,
        use_double:       bool  = True,
        use_dueling:      bool  = True,
        lr:               float = 1e-3,
        gamma:            float = 0.99,
        epsilon_start:    float = 1.0,
        epsilon_end:      float = 0.05,
        epsilon_decay:    float = 0.995,
        buffer_size:      int   = 10_000,
        batch_size:       int   = 64,
        target_sync_freq: int   = 100,
        grad_clip:        float = 1.0,      # NEW: gradient clipping
        device:           Optional[str] = None,
    ):
        self.action_dim       = action_dim
        self.use_double       = use_double
        self.use_dueling      = use_dueling
        self.gamma            = gamma
        self.epsilon          = epsilon_start
        self.epsilon_end      = epsilon_end
        self.epsilon_decay    = epsilon_decay
        self.batch_size       = batch_size
        self.target_sync_freq = target_sync_freq
        self.grad_clip        = grad_clip
        self.steps            = 0

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQNAgent] Using device: {self.device}")

        # Main (online) network + Target network
        net_cls = DuelingQNetwork if self.use_dueling else QNetwork
        self.main_net   = net_cls(state_dim, action_dim).to(self.device)
        self.target_net = net_cls(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.main_net.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()   # FIX: Huber loss instead of MSE
                                             # → more stable, less sensitive to outlier rewards
        self.buffer    = ReplayBuffer(buffer_size)

    # ── Action selection ──────────────────────────────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.main_net(state_t).squeeze(0).cpu().numpy()
        # Tie-breaking: if multiple actions share the max Q-value,
        # choose randomly among them (from Q-learning lab notebook pattern)
        max_q       = q_vals.max()
        best_actions = np.where(np.isclose(q_vals, max_q))[0]
        return int(np.random.choice(best_actions))

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, float(done))

    # ── Learning step ─────────────────────────────────────────────────────────
    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        s  = torch.as_tensor(states,      dtype=torch.float32, device=self.device)
        a  = torch.as_tensor(actions,     dtype=torch.int64,   device=self.device)
        r  = torch.as_tensor(rewards,     dtype=torch.float32, device=self.device)
        s_ = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        d  = torch.as_tensor(dones,       dtype=torch.float32, device=self.device)

        # Current Q(s, a)
        q_current = self.main_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Target: r + γ · Q_target(s', argmax_a Q_main(s', a))
        with torch.no_grad():
            if self.use_double:
                best_actions = self.main_net(s_).argmax(dim=1, keepdim=True)
                q_next = self.target_net(s_).gather(1, best_actions).squeeze(1)
            else:
                q_next = self.target_net(s_).max(dim=1)[0]
            q_target = r + self.gamma * q_next * (1.0 - d)

        loss = self.loss_fn(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()

        # FIX: gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(self.main_net.parameters(), self.grad_clip)

        self.optimizer.step()

        # Epsilon decay (per learn call, not per step)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Sync target network periodically
        self.steps += 1
        if self.steps % self.target_sync_freq == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())

        return loss.item()

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save(
            {
                "main_net":  self.main_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon":   self.epsilon,
                "steps":     self.steps,
            },
            path,
        )

    def load(self, path: str):
        # FIX: weights_only=True avoids PyTorch 2.x deprecation warning
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.main_net.load_state_dict(ckpt["main_net"])
        self.target_net.load_state_dict(ckpt["main_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.steps   = ckpt["steps"]
        print(f"[DQNAgent] Loaded checkpoint (step={self.steps}, epsilon={self.epsilon:.4f})")
