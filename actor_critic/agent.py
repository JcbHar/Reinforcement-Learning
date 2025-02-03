import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models import Actor, Critic

class Agent():
    def __init__(self, num_states, num_actions, hidden_size, actor_learning_rate=1e-4, 
                 critic_learning_rate=1e-3, gamma=0.99):
        """
        Parameters:
            num_states (int): Is the number of states in the observation space
            num_actions (int): Is the number actions in the action space
            hidden_size (int): Is the number of nodes in the hidden layer of the NN
            actor_learning_rate (float): Is the learning rate for the actor NN (using Adam)
            critic_learning_rate (float): Is the learning rate for the critic NN (using Adam)
            gamma (float): Is the discount factor of future rewards
            device (string): Is the device selected for the NN training

            actor (nn.Module): Is the actor NN
            critic (nn.Module): Is the critic NN

            critic_criterion (nn.f): Is criterion for the critic for measuring loss
            actor_optimizer (optim.f): Is the optimizer used for gradient descent on actor
            critic_optimizer (optim.f): Is the optimizer used for gradient descent on critic
        """
        
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.action = None
        self.hidden_size = hidden_size
        output_size = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(self.num_states, num_actions, hidden_size, hidden_size).to(self.device)
        self.critic = Critic(self.num_states, self.num_actions, hidden_size, hidden_size, output_size).to(self.device)
        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self.log_alpha = torch.tensor(np.log(0.001), requires_grad=True, device=self.device) 
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)  
        self.target_entropy = -0.1*self.num_actions 
        self.alpha = 0.001

   
    def choose_action(self, state, action_space_low, action_space_high):
        # Converts the state into a pytorch tensor, unequeeze(0) adds a batch dimension for pytorch efficiency
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Stabalises NN behaviour
        self.actor.eval()

        # Disables gradient tracking, you don't need to for choosing an action
        with torch.no_grad():
            # Gets the probability distribution from forward passing the actor
            distribution = self.actor.forward(state)
            
            # Samples a random action from that distribution
            #self.action = distribution.sample()
            self.action = (distribution.rsample() * torch.exp(self.log_alpha)).clamp(-1, 1)
            
        # ? Puts actor in training mode ?
        self.actor.train()

        # Forces the action to be within the range of the action space
        self.action = torch.clamp(self.action, action_space_low, action_space_high)
        # Squeeze removes the extra dimensions added my unsqueeze, cpu processes 
        # the tensor  correctly, then is converted to a numpy array
        self.action = self.action.squeeze().cpu().numpy()
        return self.action

        
    def learn(self, states, actions, rewards, next_states, dones):
        # Formats properly into nparray and then pytorch tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device) 
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device) 
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        if states.dim() == 1:
            states = states.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
            
        # Concats states and actions into a singular array [[ state | action ]*]
        state_action = torch.cat([states, actions], dim=1).to(self.device)
        #print(f"state_action shape: {state_action.shape}")
        # Passes state action pair into critic to recieve Q values
        states_values = self.critic(state_action)

        with torch.no_grad():
            next_action_distr = self.actor(next_states)
            next_actions = next_action_distr.sample()

            if next_states.dim() == 1:
                next_states = next_states.unsqueeze(0)
            if next_actions.dim() == 1:
                next_actions = next_actions.unsqueeze(0)
                
            # Concats next states and actions into a singular array [[ state | action ]*]
            next_state_action = torch.cat([next_states, next_actions], dim=1).to(self.device)
            #print(f"next_state_action shape: {next_state_action.shape}")
            # Passes state action pair into critic to recieve Q values
            next_states_values = self.critic(next_state_action)

        # Disables gradient tracking, you don't need to for action probabilities
        with torch.no_grad():
            target_values = rewards + self.gamma * next_states_values * (1 - dones)

        # Computes temporal difference (TD) error, stops calculating if episode has ended
        delta = target_values - states_values

        #print("agent, states input to actor:", states)
            #assert not torch.isnan(states).any(), "State contains NaN!"
        print("Alpha:", self.alpha)

        action_distribution = self.actor(states)
        log_prob = action_distribution.log_prob(actions)
        entropy = action_distribution.entropy().mean()
        actor_loss = (-log_prob * delta - self.alpha * entropy).mean()
    
        critic_loss = self.critic_criterion(states_values, target_values)

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
    
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
        alpha_loss = - (self.log_alpha * (entropy + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().clamp(min=0.01)

    
    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'gamma': self.gamma,
        }, filepath)

    
    def load(self, filepath):
        # weights_only=False loads with hyperparameters, optimizer state and model parameters
        checkpoint = torch.load(filepath, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.gamma = checkpoint['gamma']