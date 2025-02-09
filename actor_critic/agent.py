import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from buffer import ReplayBuffer
from models import Actor, Critic

class Agent():
    def __init__(self, num_states, num_actions, hidden_size, actor_lr=1e-4, critic_lr=1e-4, alpha_lr=1e-4, gamma=0.99 ,buffer_capacity=100000, batch_size=64, tau=0.005, seed=42):
        self.num_states = num_states
        self.num_actions = num_actions
        
        self.gamma = gamma
        self.tau = tau
        
        self.action = None
        
        self.hidden_size = hidden_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.set_seed(seed)

        self.replay_buffer = ReplayBuffer(buffer_capacity, batch_size, self.device, seed)

        self.actor = Actor(self.num_states, num_actions, hidden_size, seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(self.num_states, self.num_actions, hidden_size, seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.critic_target = Critic(self.num_states, self.num_actions, hidden_size, seed).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.log_alpha = torch.tensor(np.log(1.0), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)
        
        self.target_entropy = -self.num_actions
        self.alpha = self.log_alpha.exp().detach().item()

        self.actor_loss = None
        self.critic_loss = None
        self.alpha_loss = None
        self.states_values = None
        self.target_values = None
        self.next_states_values = None
        self.log_prob = None

    
    def choose_action(self, state, action_space_low, action_space_high):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        self.actor.eval()
        
        with torch.no_grad():
            distribution = self.actor.forward(state)
            self.action = torch.tanh(distribution.rsample() * torch.exp(self.log_alpha))
            
            
        self.actor.train()
        
        return self.action.squeeze().cpu().numpy()

        
    def learn(self):
        if self.replay_buffer.size() < self.replay_buffer.batch_size:
            return
    
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        rewards = rewards / 50.0
        rewards = torch.clamp(rewards, min=-10, max=10)
    
        state_action = torch.cat([states, actions], dim=1).to(self.device)
        self.states_values = self.critic(state_action)
    
        with torch.no_grad():
            next_action_distr = self.actor(next_states)
            next_actions = next_action_distr.sample()
    
            next_state_action = torch.cat([next_states, next_actions], dim=1).to(self.device)
            self.next_states_values = self.critic(next_state_action)
    
            self.target_values = rewards + self.gamma * self.next_states_values * (1 - dones)
            self.target_values = torch.clamp(self.target_values, min=-100, max=100)

        self.critic_criterion = nn.SmoothL1Loss()
        self.critic_loss = self.critic_criterion(self.states_values, self.target_values.detach())

        self.critic_optimizer.zero_grad()
        self.critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        action_distribution = self.actor(states)
        entropy = action_distribution.entropy().mean()
        self.log_prob = action_distribution.log_prob(actions).sum(dim=-1, keepdim=True)
        self.log_prob -= (2 * (np.log(2) - actions - torch.nn.functional.softplus(-2 * actions))).sum(dim=-1, keepdim=True)
        self.log_prob = torch.clamp(self.log_prob, min=-20, max=2)

        self.actor_loss = (self.alpha * self.log_prob - self.critic(state_action)).mean()

        self.actor_optimizer.zero_grad()
        self.actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        self.alpha_loss = - (self.log_alpha * (entropy + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        self.alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().clamp(min=0.001, max=1.0)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            
    def save(self, filename="agent_checkpoint.pth"):
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "alpha": self.alpha,
            "tau": self.tau,
            "gamma": self.gamma,
        }
        torch.save(checkpoint, filename)

        
    def load(self, filename="agent_checkpoint.pth"):
        try:
            checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
            
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
            self.log_alpha = torch.tensor(checkpoint["log_alpha"], requires_grad=True, device=self.device)
            self.alpha = checkpoint["alpha"]
            self.tau = checkpoint["tau"]
            self.gamma = checkpoint["gamma"]

        except FileNotFoundError:
            print(f"No checkpoint found: '{filename}'")
            
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # slows down training but ensures determinism

    
    def get_alpha(self):
        return self.alpha if isinstance(self.alpha, float) else self.alpha.cpu().item()

    
    def get_actor_loss(self):
        return self.actor_loss.cpu().item() if self.actor_loss is not None else None

    
    def get_critic_loss(self):
        return self.critic_loss.cpu().item() if self.critic_loss is not None else None

    
    def get_alpha_loss(self):
        return self.alpha_loss.cpu().item() if self.alpha_loss is not None else None

    
    def get_q_values(self):
        return self.states_values.mean().cpu().item() if self.states_values is not None else None

    
    def get_target_q_values(self):
        return self.target_values.mean().cpu().item() if self.target_values is not None else None

    
    def get_next_state_q_values(self):
        return self.next_states_values.mean().cpu().item() if self.next_states_values is not None else None

    
    def get_log_prob(self):
        return self.log_prob.mean().cpu().item() if self.log_prob is not None else None

