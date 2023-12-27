import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


# Define the Policy Network class, which determines the agent's behavior
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, actor_lr):
        super(PolicyNetwork, self).__init__()
        # Define the network layers
        self.fc_1 = nn.Linear(state_dim, 64)  # First fully connected layer
        self.fc_2 = nn.Linear(64, 64)  # Second fully connected layer
        self.fc_mu = nn.Linear(64, action_dim)  # Output layer for action means
        self.fc_std = nn.Linear(64, action_dim)  # Output layer for action log standard deviations

        self.lr = actor_lr  # Learning rate for the actor (policy network)

        # Define the range for the log standard deviation of actions
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

        # Define the action scale and bias for converting network output to an action
        self.max_action = 2
        self.min_action = -2
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0

        # Set up the optimizer for the policy network
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        # Forward pass through the network to obtain action means and log standard deviations
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        mu = self.fc_mu(x)
        log_std = self.fc_std(x)
        # Clamp the log_std to be within the specified range for numerical stability
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        # Sample an action from the policy given the current state
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        # Create a normal distribution and sample an action
        reparameter = Normal(mean, std)
        x_t = reparameter.rsample()
        y_t = torch.tanh(x_t)
        # Scale and bias the action to be within the appropriate range
        action = self.action_scale * y_t + self.action_bias

        # Calculate the log probability of the action, including the correction for the tanh squashing
        log_prob = reparameter.log_prob(x_t)
        log_prob = log_prob - torch.sum(torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6), dim=-1, keepdim=True)

        return action, log_prob


# Define the Q Network class, which approximates the state-action value function
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, critic_lr):
        super(QNetwork, self).__init__()
        # Define the network layers
        self.fc_s = nn.Linear(state_dim, 32)  # Layer for state
        self.fc_a = nn.Linear(action_dim, 32)  # Layer for action
        self.fc_1 = nn.Linear(64, 64)  # Fully connected layer after concatenation
        self.fc_out = nn.Linear(64, action_dim)  # Output layer for Q-value

        self.lr = critic_lr  # Learning rate for the critic (Q-network)

        # Set up the optimizer for the Q-network
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, a):
        # Forward pass through the network to obtain Q-value given state and action
        h1 = F.leaky_relu(self.fc_s(x))
        h2 = F.leaky_relu(self.fc_a(a))
        # Concatenate the outputs from the state and action layers
        cat = torch.cat([h1, h2], dim=-1)
        # Pass through another layer and output the Q-value
        q = F.leaky_relu(self.fc_1(cat))
        q = self.fc_out(q)
        return q