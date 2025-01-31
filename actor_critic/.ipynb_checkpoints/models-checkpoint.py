import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, num_states, num_actions, hiddenl1_size, hiddenl2_size):
        super(Actor, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.linear1 = nn.Linear(self.num_states, hiddenl1_size)
        self.linear2 = nn.Linear(hiddenl1_size, hiddenl2_size)
        
        self.mean_layer = nn.Linear(hiddenl2_size, self.num_actions)
        self.log_std_layer = nn.Linear(hiddenl2_size, self.num_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        std = torch.exp(log_std)
        
        #print("models, log_std:", log_std)
        #print("models, mean:", mean)
        #print("models, std:", std)
        #std = torch.exp(log_std.clamp(-20, 2))

        distribution = torch.distributions.Normal(mean, std)

        return distribution


class Critic(nn.Module):
    def __init__(self, num_states, num_actions, hiddenl1_size, hiddenl2_size, output_size):
        super(Critic, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.input_size = self.num_states+self.num_actions
        self.linear1 = nn.Linear(self.input_size, hiddenl1_size)
        self.linear2 = nn.Linear(hiddenl1_size, hiddenl2_size)
        self.linear3 = nn.Linear(hiddenl2_size, output_size)

    def forward(self, state):
        state = state.to(next(self.parameters()).device)
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        
        return value