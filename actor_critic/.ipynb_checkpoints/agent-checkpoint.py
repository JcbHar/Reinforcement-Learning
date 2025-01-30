import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models import Actor, Critic

class Agent():
    def __init__(self, num_states, num_actions, hidden_size, actor_learning_rate=1e-4, 
                 critic_learning_rate=1e-3, gamma=0.99, device='cuda', learning_rate=0.001, beta=0.001):
        """
        Parameters:
        num_states (int): Is the number of states in the observation space
        num_actions (int): Is the number actions in the action space
        hidden_size (int): Is the number of nodes in the hidden layer of the NN
        actor_learning_rate (float): Is the learning rate for the actor NN (using Adam)
        critic_learning_rate (float): Is the learning rate for the critic NN (using Adam)
        gamma (float): Is the discount factor of future rewards
        device (string): Is the device selected for the NN training
        learning_rate (float): learning rate for 
        beta (float):
        debug (bool):
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.action = None
        self.hidden_size = hidden_size
        output_size = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.learning_rate = learning_rate
        self.beta = beta

        self.actor = Actor(self.num_states, num_actions, hidden_size, hidden_size).to(self.device)
        self.critic = Critic(self.num_states, self.num_actions, hidden_size, hidden_size, output_size).to(self.device)
        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
   
    def choose_action(self, state, action_space_low, action_space_high, debug=False):
        if debug:
            print("Agent, choose action: state=", state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        #state = torch.from_numpy(state).float().unsqueeze(0)
        state = state.to(self.device)
        self.actor.eval()
            
        with torch.no_grad():
            distribution = self.actor.forward(state)
            self.action = distribution.sample()
            
        self.actor.train()
        
        self.action = torch.clamp(self.action, action_space_low, action_space_high)
        self.action = self.action.squeeze().cpu().numpy()
        return self.action

        
    def learn(self, states, actions, rewards, next_states, dones, debug=False):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device) 
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device) 
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        #state_action = torch.cat([states, actions], dim=0)
        state_action = torch.cat([states, actions], dim=0).to(self.device)
        states_values = self.critic(state_action)
        #next_state_action = torch.cat([next_states, actions], dim=0)
        next_state_action = torch.cat([next_states, actions], dim=0).to(self.device)
        next_states_values = self.critic(next_state_action)

        with torch.no_grad():
            probs = self.actor(states)
        log_prob = probs.log_prob(actions)

        delta = rewards + self.gamma*next_states_values*(1-int(dones)) - states_values

        actor_loss = -log_prob*delta
        actor_loss = actor_loss.mean()
        
        critic_loss = (delta**2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Extracts the weights and biases of a layer of the NNs at a time (represented as a tensor)
        for actor_param, critic_param in zip(self.actor.parameters(), self.critic.parameters()):
            # Checks if parameters were changed on the forward pass (to calculate a gradient)
            if actor_param.grad is not None:
                # Applies the learning rate to the current parameters to control the step size using gradient descent             
                actor_param.data.add_(self.learning_rate * actor_param.grad)
            # Checks if parameters were changed on the forward pass (to calculate a gradient)
            if critic_param.grad is not None:
                # Applies the learning rate (beta) to the current parameters to control the step size using gradient descent             
                critic_param.data.add_(self.beta * critic_param.grad)

    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'beta': self.beta
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.gamma = checkpoint['gamma']
        self.learning_rate = checkpoint['learning_rate']
        self.beta = checkpoint['beta']
        print(f"Agent loaded from {filepath}")