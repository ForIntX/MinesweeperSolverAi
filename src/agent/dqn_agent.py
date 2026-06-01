import random

import torch
import torch.nn as nn
import torch.optim as optim

from .q_network import QNetwork
from .replay_memory import ReplayMemory, Transition


class DQNAgent:
    def __init__(
        self,
        input_dim,
        output_dim,
        device="cpu",
        hidden_dim=128,
        lr=1e-3,
        memory_capacity=10000,
        batch_size=64,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    ):
        self.device = torch.device(device)
        self.output_dim = output_dim
        
        self.policy_net = QNetwork(input_dim, output_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_net = QNetwork(input_dim, output_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_capacity)
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_action(self, state_tensor, valid_actions):
        if not valid_actions:
            raise ValueError("select_action icin en az bir gecerli aksiyon gerekli")

        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor.to(self.device))
            q_values = q_values.squeeze(0)
            masked_q_values = torch.full((self.output_dim,), float('-inf')).to(self.device)

            for action in valid_actions:
                masked_q_values[action] = q_values[action]
                
            return torch.argmax(masked_q_values).item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(self.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
        
        expected_state_action_values = reward_batch + (self.gamma * next_state_values * (1 - done_batch))

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon
