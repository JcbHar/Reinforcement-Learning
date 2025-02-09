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
        state = np.array(state, dtype=np.float32).flatten()
        next_state = np.array(next_state, dtype=np.float32).flatten()
        
        self.buffer.append((state, action, reward, next_state, done))

        
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
    
        states = torch.FloatTensor(np.stack(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).unsqueeze(1).to(self.device)
    
        return states, actions, rewards, next_states, dones


    def size(self):
        return len(self.buffer)
