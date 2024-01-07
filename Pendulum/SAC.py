import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
import os
import gym
import argparse
from tensorboardX import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()

parser.add_argument("--env_name", default="Pendulum-v1")  # OpenAI gym environment name
parser.add_argument('--tau',  default=0.01, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--gradient_steps', default=1, type=int)


parser.add_argument('--lr_p', default=1e-3, type=float)
parser.add_argument('--lr_q', default=1e-3, type=float)
parser.add_argument('--lr_alpha', default=5e-3, type=float)
parser.add_argument('--gamma', default=0.98, type=float) # discount gamma
parser.add_argument('--capacity', default=10000, type=int) # replay buffer size
parser.add_argument('--max_episode', default=150, type=int) #  num of games
parser.add_argument('--max_step', default=200, type=int) # num of steps per gameparser.add_argument('--batch_size', default=128, type=int) # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
parser.add_argument('--batch_size', default=128, type=int)
# optional parameters
parser.add_argument('--exploration_noise', default=0.3, type=float)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=1000, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
args = parser.parse_args()

env = gym.make(args.env_name)

# Set seeds
if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_action = float(env.action_space.low[0])

directory = 'expSAC_Pendulum/'
if not os.path.isdir(directory):
    os.mkdir(directory)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, actor_lr, max_action, min_action, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        # Define the network layers
        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_std1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)  # Output layer for action means
        self.fc_std = nn.Linear(hidden_dim, action_dim)  # Output layer for action log standard deviations

        self.lr = actor_lr  # Learning rate for the actor (policy network)

        # Define the range for the log standard deviation of actions
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

        # Define the action scale and bias for converting network output to an action
        self.max_action = max_action
        self.min_action = min_action
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0

        # Set up the optimizer for the policy network
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        # Forward pass through the network to obtain action means and log standard deviations
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        x = F.leaky_relu(self.fc_3(x))
        mu = self.fc_mu1(x)
        mu = self.fc_mu(mu)
        log_std = self.fc_std1(x)
        log_std = self.fc_std(log_std)
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


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, critic_lr, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc_s = nn.Linear(state_dim, hidden_dim)  # Layer for state
        self.fc_a = nn.Linear(action_dim, hidden_dim)  # Layer for action
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)  # Output layer for Q-value

        self.lr = critic_lr  # Learning rate for the critic (Q-network)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, a):
        # Forward pass through the network to obtain Q-value given state and action
        h1 = F.leaky_relu(self.fc_s(x))
        h2 = F.leaky_relu(self.fc_a(a))
        # Concatenate the outputs from the state and action layers
        cat = torch.cat([h1, h2], dim=-1)
        # Pass through another layer and output the Q-value
        q = F.leaky_relu(self.fc_1(cat))
        q = F.leaky_relu(self.fc_2(q))
        q = self.fc_out(q)
        return q

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)  # fixed-size buffer to hold experience tuples
        self.dev = device  # device (CPU or GPU) to perform computations

    def put(self, transition):
        self.buffer.append(transition)  # add an experience tuple to the buffer

    def sample(self, n):
        # Randomly sample a batch of experiences from the buffer
        mini_batch = random.sample(self.buffer, n)
        # Unzip the batch into separate arrays for each component
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            # Unpack the experience tuple
            s, a, r, s_prime, done = transition
            # Append each component to its corresponding list
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        # Convert the lists of components into tensors and move to the specified device
        s_batch = torch.tensor(s_lst, dtype=torch.float).to(self.dev)
        a_batch = torch.tensor(a_lst, dtype=torch.float).to(self.dev)
        r_batch = torch.tensor(r_lst, dtype=torch.float).to(self.dev)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float).to(self.dev)
        done_batch = torch.tensor(done_mask_lst, dtype=torch.float).to(self.dev)

        # Return the batch of experiences as tensors
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def size(self):
        return len(self.buffer)  # Return the current size of the replay buffer



class SAC_Agent:
    def __init__(self, state_dim=state_dim, action_dim=action_dim, 
                 lr_pi=args.lr_p, lr_q=args.lr_q, gamma=args.gamma,
                 batch_size=args.batch_size, capacity=args.capacity,
                 tau=args.tau, init_alpha=0.01, lr_alpha=args.lr_alpha,
                 device=device
                 ):
        self.state_dim = state_dim  # example shape: [cos(theta), sin(theta), theta_dot]
        self.action_dim = action_dim  # example action: [torque] with range in [-2,2]
        self.lr_pi = lr_pi  # learning rate for policy network
        self.lr_q = lr_q  # learning rate for Q-network
        self.gamma = gamma  # discount factor for reward
        self.batch_size = batch_size  # size of the sampled mini-batch from the buffer
        self.capacity = capacity  # limit of the replay buffer
        self.tau = tau  # soft-update hyperparameter for Q-target
        self.init_alpha = init_alpha  # initial value for the entropy coefficient alpha
        self.target_entropy = -self.action_dim  # target entropy for alpha optimization
        self.lr_alpha = lr_alpha  # learning rate for entropy coefficient alpha
        self.DEVICE = device  # set the device
        self.writer = SummaryWriter(directory)  # initialize the tensorboard writer
        self.num_training = 1

        # Initialize replay buffer, policy network, Q-networks, and their target networks
        self.memory = ReplayBuffer(capacity=capacity, device=self.DEVICE)

        # Initialize the log_alpha parameter and its optimizer
        self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.DEVICE)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)

        # Initialize the policy network (π) and two Q-networks with optimizers
        self.PI = PolicyNetwork(self.state_dim, self.action_dim, self.lr_pi, max_action=max_action, min_action=min_action).to(self.DEVICE)
        self.Q1 = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q1_target = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q2 = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q2_target = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)

        # Initialize Q-target networks by copying the weights from the Q-networks
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

    def choose_action(self, s):
        # Chooses an action given a state 's' based on the current policy
        with torch.no_grad():
            action, log_prob = self.PI.sample(s.to(self.DEVICE))
        return action, log_prob

    def calc_target(self, mini_batch):
        s, a, r, s_prime, done = mini_batch
        with torch.no_grad():
            # Sample new actions and their log probabilities for next states
            a_prime, log_prob_prime = self.PI.sample(s_prime)
            # Calculate entropy term
            entropy = - self.log_alpha.exp() * log_prob_prime
            # Calculate Q-targets by using the minimum of both Q-networks
            q1_target, q2_target = self.Q1_target(s_prime, a_prime), self.Q2_target(s_prime, a_prime)
            q_target = torch.min(q1_target, q2_target)
            # Calculate the final target value
            target = r + self.gamma * done * (q_target + entropy)
            # target = r + self.gamma * (q_target + entropy)
        return target

    def train_agent(self):
        mini_batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = mini_batch

        td_target = self.calc_target(mini_batch)

        # Update the Q1-network and its optimizer
        q1_loss = F.smooth_l1_loss(self.Q1(s_batch, a_batch), td_target)
        self.Q1.optimizer.zero_grad()
        q1_loss.mean().backward()
        self.Q1.optimizer.step()
        self.writer.add_scalar('Loss/q1_loss', q1_loss.mean(), global_step=self.num_training)

        # Update the Q2-network and its optimizer
        q2_loss = F.smooth_l1_loss(self.Q2(s_batch, a_batch), td_target)
        self.Q2.optimizer.zero_grad()
        q2_loss.mean().backward()
        self.Q2.optimizer.step()
        self.writer.add_scalar('Loss/q2_loss', q2_loss.mean(), global_step=self.num_training)

        # Update the policy network (PI) and its optimizer
        a, log_prob = self.PI.sample(s_batch)
        entropy = -self.log_alpha.exp() * log_prob
        q1, q2 = self.Q1(s_batch, a), self.Q2(s_batch, a)
        q = torch.min(q1, q2)
        pi_loss = -(q + entropy)  # for gradient ascent
        self.PI.optimizer.zero_grad()
        pi_loss.mean().backward()
        self.PI.optimizer.step()
        self.writer.add_scalar('Loss/actor_loss', pi_loss.mean(), global_step=self.num_training)


        # Update the log_alpha parameter for entropy tuning
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # Soft-update the Q-target networks
        for param_target, param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        for param_target, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        self.num_training += 1

    def load(self, dir):
        self.PI.load_state_dict(torch.load(dir + "/sac_actor.pt"))
        self.Q1.load_state_dict(torch.load(dir + "/sac_critic1.pt"))
        self.Q1_target.load_state_dict(torch.load(dir + "/sac_critic1.pt"))
        self.Q2.load_state_dict(torch.load(dir + "/sac_critic2.pt"))
        self.Q2_target.load_state_dict(torch.load(dir + "/sac_critic2.pt"))
        self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.DEVICE)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)
        print("====================================")
        print("Model has been loaded...")
        print("====================================")

    def save(self, dir):
        torch.save(self.PI.state_dict(), dir + "/sac_actor.pt")
        torch.save(self.Q1.state_dict(), dir + "/sac_critic1.pt")
        print("====================================")
        print("Model has been saved...")
        print("====================================")
    

# Main function to train the agent
if __name__ == '__main__':
    agent = SAC_Agent()

    print_once = True  # A flag to control print statements

    # Training loop over episodes
    for ep in range(args.max_episode):
        state = env.reset()  # Reset the environment and get the initial state
        score, done = 0.0, False

        # Loop for each step of the episode
        for t in range(args.max_step):
            # Choose an action using the policy network
            action, log_prob = agent.choose_action(torch.FloatTensor(state))
            action = action.detach().cpu().numpy()  # Convert action to numpy array and detach from graph
            action = (action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)

            # Execute the action in the environment
            state_prime, reward, done, _ = env.step(action)
            # Store the experience in the replay buffer
            agent.memory.put((state, action, reward, state_prime, done))
            score += reward  # Update the cumulative reward
            state = state_prime  # Update the state

            # If enough experiences are collected, start training the agent
            if agent.memory.size()> 1000:
                print_once = False
                agent.train_agent()
            
            if ep==0: running_reward = score
            running_reward = 0.05 * score + (1 - 0.05) * running_reward
            agent.writer.add_scalar('Reward/train', score, global_step=ep)
            agent.writer.add_scalar('Running_Reward/train', running_reward, global_step=ep)

        # Print the average score of the episode
        print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(t, ep, score))

        if ep % args.log_interval == 0:
            agent.save(directory)
