import torch
import random
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity, batch_size, device, seed):
        self.capacity = capacity 
        self.batch_size = batch_size 
        self.device = device
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.buffer = deque(maxlen=capacity)

    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones

    
    def size(self):
        return len(self.buffer)