import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from networks import PolicyNetwork, QNetwork


# A class to implement the replay buffer for storing experience tuples
class ReplayBuffer:
    def __init__(self, buffer_limit, device):
        self.buffer = deque(maxlen=buffer_limit)  # fixed-size buffer to hold experience tuples
        self.dev = device  # device (CPU or GPU) to perform computations

    def put(self, transition):
        self.buffer.append(transition)  # add an experience tuple to the buffer

    def sample(self, n):
        # Randomly sample a batch of experiences from the buffer
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        # Convert list of numpy arrays to a single numpy array before conversion to a tensor
        s_batch = np.array(s_lst, dtype=np.float32)
        a_batch = np.array(a_lst, dtype=np.float32)
        r_batch = np.array(r_lst, dtype=np.float32)
        s_prime_batch = np.array(s_prime_lst, dtype=np.float32)
        done_batch = np.array(done_mask_lst, dtype=np.float32)

        # Then convert numpy arrays to tensors and move to the specified device
        s_batch = torch.from_numpy(s_batch).to(self.dev)
        a_batch = torch.from_numpy(a_batch).to(self.dev)
        r_batch = torch.from_numpy(r_batch).to(self.dev)
        s_prime_batch = torch.from_numpy(s_prime_batch).to(self.dev)
        done_batch = torch.from_numpy(done_batch).to(self.dev)

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def size(self):
        return len(self.buffer)  # Return the current size of the replay buffer

# A class to represent a Soft Actor-Critic agent
class SAC_Agent:
    def __init__(self):
        # Define the dimensions of state and action as well as hyperparameters
        self.state_dim = 2  # MountainCarContinuous-v0 state space: position and velocity
        self.action_dim = 1  # MountainCarContinuous-v0 action space: force applied
        self.lr_pi = 0.01  # learning rate for policy network
        self.lr_q = 0.01  # learning rate for Q-network
        self.gamma = 0.999  # discount factor for reward
        self.batch_size = 200  # size of the sampled mini-batch from the buffer
        self.buffer_limit = 100000  # limit of the replay buffer
        self.tau = 0.01  # soft-update hyperparameter for Q-target
        self.init_alpha = 0.01  # initial value for the entropy coefficient alpha
        self.target_entropy = -self.action_dim  # target entropy for alpha optimization
        self.lr_alpha = 0.005  # learning rate for entropy coefficient alpha
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set the device

        # Initialize replay buffer, policy network, Q-networks, and their target networks
        self.memory = ReplayBuffer(self.buffer_limit, self.DEVICE)

        # Initialize the log_alpha parameter and its optimizer
        self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.DEVICE)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)

        # Initialize the policy network (π) and two Q-networks with optimizers
        self.PI = PolicyNetwork(self.state_dim, self.action_dim, self.lr_pi).to(self.DEVICE)
        self.Q1 = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q1_target = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q2 = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q2_target = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)

        # Initialize Q-target networks by copying the weights from the Q-networks
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

    def choose_action(self, s):
        # Chooses an action given a state 's' based on the current policy
        # No gradient computation is needed here
        with torch.no_grad():
            action, log_prob = self.PI.sample(s.to(self.DEVICE))
        return action, log_prob

    def calc_target(self, mini_batch):
        # Calculate the target value for training Q-networks
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
        return target

    def train_agent(self):
        # Main training loop for the SAC agent
        mini_batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = mini_batch

        # Calculate the temporal difference target
        td_target = self.calc_target(mini_batch)

        # Update the Q1-network and its optimizer
        q1_loss = F.smooth_l1_loss(self.Q1(s_batch, a_batch), td_target)
        self.Q1.optimizer.zero_grad()
        q1_loss.mean().backward()
        self.Q1.optimizer.step()

        # Update the Q2-network and its optimizer
        q2_loss = F.smooth_l1_loss(self.Q2(s_batch, a_batch), td_target)
        self.Q2.optimizer.zero_grad()
        q2_loss.mean().backward()
        self.Q2.optimizer.step()

        # Update the policy network (PI) and its optimizer
        a, log_prob = self.PI.sample(s_batch)
        entropy = -self.log_alpha.exp() * log_prob
        q1, q2 = self.Q1(s_batch, a), self.Q2(s_batch, a)
        q = torch.min(q1, q2)
        pi_loss = -(q + entropy)  # for gradient ascent
        self.PI.optimizer.zero_grad()
        pi_loss.mean().backward()
        self.PI.optimizer.step()

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

