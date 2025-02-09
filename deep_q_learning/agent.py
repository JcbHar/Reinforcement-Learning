import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from models import DQN
from buffer import ReplayBuffer


class Agent():
    def __init__(self, num_states, num_actions, hidden_size, lr=1e-4, gamma=0.99, min_epsilon=0.001, max_epsilon=1.0, total_episodes=500, buffer_capacity=100000, batch_size=64, target_update=10, seed=42):
        self.num_states = num_states
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed(seed)

        self.total_episodes = total_episodes
        
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.epsilon = max_epsilon
        self.epsilon_decay = np.exp(np.log(self.min_epsilon / self.max_epsilon) / self.total_episodes)
        self.gamma = gamma

        self.action_space = np.linspace(-2, 2, num_actions)

        self.replay_buffer = ReplayBuffer(buffer_capacity, batch_size, self.device, seed)

        self.policy_net = DQN(num_states, num_actions, hidden_size).to(self.device)
        self.target_net = DQN(num_states, num_actions, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.q_loss = 0.0
        self.q_values = None

    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.num_actions-1)
    
        state = np.array(state, dtype=np.float32).flatten()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
    
        with torch.no_grad():
            q_values = self.policy_net(state)
    
        return q_values.argmax().item()

        
    def learn(self):
        if self.replay_buffer.size() < self.batch_size:
            return
    
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
    
        q_values = self.policy_net(states).gather(1, actions)
    
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones.float())
    
        loss = nn.MSELoss()(q_values, target_q_values)
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        self.q_loss = loss.item()
        self.q_values = q_values.detach().cpu().numpy()

    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    
    def save(self, filename="dqn_checkpoint.pth"):
        checkpoint = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }
        torch.save(checkpoint, filename)

    
    def load(self, filename="dqn_checkpoint.pth"):
        try:
            checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
            
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = checkpoint["epsilon"]
            
        except FileNotFoundError:
            print(f"No checkpoint found at '{filename}'")

    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        
    def get_q_loss(self):
        return self.q_loss if self.q_loss is not None else None

    
    def get_q_values(self):
        return np.mean(self.q_values) if self.q_values is not None else None

