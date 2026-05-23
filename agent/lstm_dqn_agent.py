"""
agent/lstm_dqn_agent.py — LSTM + DQN Agent

Kiến trúc:
    Observation sequence  →  LSTM Encoder  →  DQN Head  →  Q-values
      (seq_len, feat_dim)      (lstm_hidden)   (dqn_hidden)  (action_dim)

Giải thích:
  - Thay vì nhận 1 observation phẳng (4-dim), agent nhận 1 CHUỖI
    gồm seq_len bước gần nhất (seq_len × 4).
  - LSTM xử lý chuỗi này → vector ngữ cảnh (context vector).
  - DQN head ánh xạ context → Q-values.
  - Đầu episode, lịch sử được zero-pad.
  - Sequence ngay trước khi store transition được build bằng push_obs().
  - Sequence kế tiếp được peek bằng peek_next_seq() TRƯỚC khi update history.

Usage trong vòng lặp train:
    agent.reset_history()
    obs, _ = env.reset()
    seq = agent.push_obs(obs)           # thêm obs vào history, trả seq hiện tại

    while not done:
        action   = agent.select_action(seq)
        next_obs, reward, term, trunc, info = env.step(action)
        done     = term or trunc

        next_seq = agent.peek_next_seq(next_obs)   # nhìn trước next_seq
        agent.store(seq, action, reward, next_seq, float(done))
        agent.learn()

        seq = agent.push_obs(next_obs)  # advance history
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional

from agent.sequence_replay_buffer import SequenceReplayBuffer


# ── LSTM Q-Network ────────────────────────────────────────────────────────────
class LSTMQNetwork(nn.Module):
    """
    LSTM Encoder tiếp nối với DQN Head.

    Input  : (batch, seq_len, feature_dim)
    Output : (batch, action_dim)  — Q-values
    """

    def __init__(
        self,
        feature_dim: int = 4,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        action_dim:  int = 3,
        dqn_hidden:  int = 128,
        dueling:     bool = True,
    ):
        super().__init__()

        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,        # input shape: (batch, seq, feat)
        )

        # DQN Head (optionally dueling)
        self.dueling = dueling
        if self.dueling:
            self.trunk = nn.Sequential(
                nn.Linear(lstm_hidden, dqn_hidden),
                nn.ReLU(),
                nn.Linear(dqn_hidden, dqn_hidden),
                nn.ReLU(),
            )
            self.value = nn.Linear(dqn_hidden, 1)
            self.advantage = nn.Linear(dqn_hidden, action_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(lstm_hidden, dqn_hidden),
                nn.ReLU(),
                nn.Linear(dqn_hidden, dqn_hidden),
                nn.ReLU(),
                nn.Linear(dqn_hidden, action_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, feature_dim)
        Returns Q-values: (batch, action_dim)
        """
        # LSTM: output (batch, seq, hidden), h_n (layers, batch, hidden)
        _, (h_n, _) = self.lstm(x)
        context = h_n[-1]          # lấy hidden state của layer cuối → (batch, hidden)
        if self.dueling:
            h = self.trunk(context)
            v = self.value(h)
            a = self.advantage(h)
            return v + (a - a.mean(dim=1, keepdim=True))
        return self.head(context)


# ── LSTM DQN Agent ────────────────────────────────────────────────────────────
class LSTMDQNAgent:
    """
    DQN agent với LSTM encoder để xử lý chuỗi quan sát.

    Mỗi 'state' không còn là vector phẳng mà là một tensor
    (seq_len × feature_dim) — biểu diễn lịch sử gần nhất.
    """

    def __init__(
        self,
        feature_dim:      int   = 6,       # số features mỗi bước (E, Q, H, ΔH, Health, Forecast)
        seq_len:          int   = 8,       # độ dài chuỗi lịch sử
        action_dim:       int   = 3,       # Sleep | LowTX | HighTX
        use_double:       bool  = True,
        use_dueling:      bool  = True,
        lstm_hidden:      int   = 64,
        lstm_layers:      int   = 1,
        dqn_hidden:       int   = 128,
        lr:               float = 1e-3,
        gamma:            float = 0.99,
        epsilon_start:    float = 1.0,
        epsilon_end:      float = 0.05,
        epsilon_decay:    float = 0.995,
        buffer_size:      int   = 10_000,
        batch_size:       int   = 64,
        target_sync_freq: int   = 100,
        grad_clip:        float = 1.0,
        device:           Optional[str] = None,
    ):
        self.feature_dim      = feature_dim
        self.seq_len          = seq_len
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
        print(
            f"[LSTMDQNAgent] device={self.device} | "
            f"seq_len={seq_len} | lstm_hidden={lstm_hidden} | dqn_hidden={dqn_hidden}"
        )

        # Networks
        net_kwargs = dict(
            feature_dim=feature_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            action_dim=action_dim,
            dqn_hidden=dqn_hidden,
            dueling=use_dueling,
        )
        self.main_net   = LSTMQNetwork(**net_kwargs).to(self.device)
        self.target_net = LSTMQNetwork(**net_kwargs).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.main_net.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()   # Huber loss — ổn định hơn MSE

        self.buffer = SequenceReplayBuffer(buffer_size)

        # Rolling observation history (reset mỗi episode)
        self._history: deque = deque(maxlen=seq_len)

    # ── History management ────────────────────────────────────────────────────

    def reset_history(self) -> None:
        """Xóa lịch sử – gọi đầu mỗi episode."""
        self._history.clear()

    def _padded_sequence(self, history_list: list) -> np.ndarray:
        """Chuyển danh sách obs thành tensor (seq_len, feature_dim), zero-pad đầu."""
        pad_len = self.seq_len - len(history_list)
        if pad_len > 0:
            zeros = [np.zeros(self.feature_dim, dtype=np.float32)] * pad_len
            history_list = zeros + history_list
        return np.array(history_list[-self.seq_len:], dtype=np.float32)

    def push_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Thêm obs vào history, trả về sequence hiện tại.
        Gọi ở đầu episode (sau reset) và sau mỗi bước (để advance state).
        """
        self._history.append(obs.astype(np.float32))
        return self._padded_sequence(list(self._history))

    def peek_next_seq(self, next_obs: np.ndarray) -> np.ndarray:
        """
        Xây dựng next_sequence nếu next_obs được thêm vào history,
        nhưng KHÔNG thực sự thay đổi history.

        Dùng để xây next_seq trước khi store transition.
        """
        future_list = list(self._history) + [next_obs.astype(np.float32)]
        return self._padded_sequence(future_list)

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, sequence: np.ndarray) -> int:
        """
        Epsilon-greedy selection.
        sequence: (seq_len, feature_dim)
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        t = torch.as_tensor(
            sequence, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, seq_len, feat)
        with torch.no_grad():
            q_vals = self.main_net(t).squeeze(0).cpu().numpy()

        # Tie-breaking ngẫu nhiên
        max_q = q_vals.max()
        best  = np.where(np.isclose(q_vals, max_q))[0]
        return int(np.random.choice(best))

    def store(
        self,
        seq:      np.ndarray,
        action:   int,
        reward:   float,
        next_seq: np.ndarray,
        done:     float,
    ) -> None:
        self.buffer.push(seq, action, reward, next_seq, done)

    # ── Learning step ─────────────────────────────────────────────────────────

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        seqs, actions, rewards, next_seqs, dones = self.buffer.sample(self.batch_size)

        # Chuyển sang tensor
        s  = torch.as_tensor(seqs,      dtype=torch.float32, device=self.device)  # (B, T, F)
        a  = torch.as_tensor(actions,   dtype=torch.int64,   device=self.device)  # (B,)
        r  = torch.as_tensor(rewards,   dtype=torch.float32, device=self.device)  # (B,)
        s_ = torch.as_tensor(next_seqs, dtype=torch.float32, device=self.device)  # (B, T, F)
        d  = torch.as_tensor(dones,     dtype=torch.float32, device=self.device)  # (B,)

        # Q(s, a) từ main network
        q_current = self.main_net(s).gather(1, a.unsqueeze(1)).squeeze(1)   # (B,)

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
        nn.utils.clip_grad_norm_(self.main_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Epsilon decay (mỗi lần learn)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Đồng bộ target network
        self.steps += 1
        if self.steps % self.target_sync_freq == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())

        return loss.item()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "main_net":    self.main_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "epsilon":     self.epsilon,
            "steps":       self.steps,
            "seq_len":     self.seq_len,
            "feature_dim": self.feature_dim,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.main_net.load_state_dict(ckpt["main_net"])
        self.target_net.load_state_dict(ckpt["main_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.steps   = ckpt["steps"]
        print(f"[LSTMDQNAgent] Loaded checkpoint (step={self.steps}, eps={self.epsilon:.4f})")

    # ── Model summary ─────────────────────────────────────────────────────────

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.main_net.parameters() if p.requires_grad)
