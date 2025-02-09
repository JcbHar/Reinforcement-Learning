import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, num_states, num_actions, hidden_size, seed=42):
        super(Actor, self).__init__()
        
        self.num_states = num_states
        self.num_actions = num_actions
        
        self.linear1 = nn.Linear(self.num_states, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        self.mean_layer = nn.Linear(hidden_size, self.num_actions)
        self.log_std_layer = nn.Linear(hidden_size, self.num_actions)

    
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_layer(x)
        
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-5, max=2)
        std = torch.exp(log_std)

        distribution = torch.distributions.Normal(mean, std)

        return distribution



class Critic(nn.Module):
    def __init__(self, num_states, num_actions, hidden_size, seed=42):
        super(Critic, self).__init__()
        
        self.num_states = num_states
        self.num_actions = num_actions
        
        output_size = 1
        
        self.input_size = self.num_states+self.num_actions
        
        self.linear1 = nn.Linear(self.input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    
    def forward(self, state):
        state = state.to(next(self.parameters()).device)
        
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        
        value = self.linear3(output)
        
        return value

        