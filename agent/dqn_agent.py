import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from typing import Optional

try:
    from agent.q_network import QNetwork
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.q_network import QNetwork

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, n):
        return random.sample(self.buf, n)

    def __len__(self):
        return len(self.buf)


# eski isim uyumluluğu
ReplayMemory = ReplayBuffer


class DQNAgent:
    """
    Double DQN + CNN (Dueling head).
    Kural tabanlı çözücünün veremediği durumlarda devreye girer.
    """

    def __init__(
        self,
        state_size,
        action_size,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=30_000,
        batch_size=64,
        target_update=2_000,
        memory_capacity=50_000,
        device=None,
        use_per=False,        # eski parametre — yok sayılır
    ):
        self.n_actions     = action_size
        self.gamma         = gamma
        self.eps_start     = epsilon_start
        self.eps_end       = epsilon_end
        self.eps_decay     = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device is None else torch.device(device)
        print(f"[DQNAgent] device={self.device}")

        self.policy = QNetwork(state_size, action_size).to(self.device)
        self.target = QNetwork(state_size, action_size).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.opt     = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory  = ReplayBuffer(memory_capacity)

        self.steps_done    = 0
        self.episodes_done = 0

    # aliases
    @property
    def policy_net(self):
        return self.policy

    @property
    def target_net(self):
        return self.target

    # ── Epsilon ───────────────────────────────────────────────────────────────

    @property
    def epsilon(self):
        t = min(self.episodes_done / self.eps_decay, 1.0)
        return self.eps_start + t * (self.eps_end - self.eps_start)

    def set_epsilon(self, val):
        if val <= self.eps_end:
            self.episodes_done = int(1e9)
        else:
            t = (self.eps_start - val) / (self.eps_start - self.eps_end)
            self.episodes_done = int(t * self.eps_decay)

    def episode_done(self):
        self.episodes_done += 1

    # ── Aksiyon seçimi ────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, valid_actions: list) -> int:
        """
        Epsilon-greedy. Geçersiz hücreler (zaten açık) -inf ile maskelenir.
        """
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.policy(s).squeeze(0)
            mask = torch.full((self.n_actions,), float("-inf"), device=self.device)
            mask[valid_actions] = 0.0
            return int((q + mask).argmax().item())

    # ── Bellek ────────────────────────────────────────────────────────────────

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    # ── Öğrenme ───────────────────────────────────────────────────────────────

    def learn(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)

        states      = torch.FloatTensor(np.stack([t.state      for t in batch])).to(self.device)
        actions     = torch.LongTensor( np.array( [t.action    for t in batch])).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(np.array( [t.reward    for t in batch])).to(self.device)
        next_states = torch.FloatTensor(np.stack([t.next_state for t in batch])).to(self.device)
        dones       = torch.FloatTensor(np.array( [float(t.done) for t in batch])).to(self.device)

        cur_q = self.policy(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            # Double DQN: aksiyon → policy, değer → target
            nq_p = self.policy(next_states)
            # Açık hücreler geçersiz aksiyon (normalize'da > -0.5)
            nq_p = nq_p.masked_fill(next_states > -0.5, float("-inf"))
            all_open  = (next_states > -0.5).all(dim=1)
            best_acts = nq_p.argmax(1, keepdim=True)
            best_acts[all_open] = 0

            nq_t = self.target(next_states).gather(1, best_acts).squeeze(1)
            nq_t[all_open] = 0.0

            target_q = rewards + self.gamma * nq_t * (1.0 - dones)

        loss = self.loss_fn(cur_q, target_q)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.opt.step()

        if self.steps_done % self.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())

        return loss.item()

    # ── Kaydet / Yükle ────────────────────────────────────────────────────────

    def save(self, path):
        torch.save({
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict(),
            "opt":    self.opt.state_dict(),
            "steps":  self.steps_done,
            "eps":    self.episodes_done,
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck["policy"])
        self.target.load_state_dict(ck["target"])
        self.opt.load_state_dict(ck["opt"])
        self.steps_done    = ck.get("steps", 0)
        self.episodes_done = ck.get("eps",   0)
        print(f"[DQNAgent] loaded | eps={self.epsilon:.3f}")