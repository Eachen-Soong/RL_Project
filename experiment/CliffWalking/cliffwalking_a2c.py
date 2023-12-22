import argparse # 1
import gym
import numpy as np
from itertools import count # 2
from collections import namedtuple # 3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import sys


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 2)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        return action_prob, state_values


class A2CAgent(object):
    def __init__(self, obs_shape, action_shape, model, gamma, device, name='A2C'):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.model = model
        self.SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.gamma = gamma
        self.name = name

    def choose_action(self, state):
        'Choose an action based on observation input'
        state = torch.from_numpy(state).float()
        probs, state_value = self.model(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        self.saved_actions.append(self.SavedAction(m.log_prob(action), state_value))

        # the action to take (left or right)
        return action.item()
    
    def update(self, optimizer, eps):
        'Update the parameter'
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # reset gradients
        optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        optimizer.step()

        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

    def save(self, model_dir, step):
        'Save the model'
        torch.save(self.model.state_dict(), model_dir + '/' + self.name + '_ep' + str(step)+'.pth')

    def load(self, model_path):
        'Load the model'
        self.model = torch.load(model_path)

# Cart Pole

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument('--train_eps', default=9999, type=int)
    parser.add_argument('--max_t', default=9999, type=int)
    parser.add_argument('--task', default='CartPole-v1', type=str)
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=bool, default=False, metavar='N',
                        help='whether to use fixed seed')
    parser.add_argument('--random_seed', default=9527, type=int)
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--device', default='cpu', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    env = gym.make(args.task)
    env.reset(seed=args.seed)
    if args.seed:
        env.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=3e-2)
    eps = np.finfo(np.float32).eps.item()
    running_reward = 10
    agent = A2CAgent(env.observation_space.shape, env.action_space.n, model, args.gamma, args.device)

    for i_episode in range(args.train_eps):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(args.max_t):
            action = agent.choose_action(state)
            state, reward, done, _, _ = env.step(action)
            if args.render:
                env.render()
            agent.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        agent.update(optimizer, eps)

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

    agent.save('./ckpt/CartPole', i_episode)

if __name__ == '__main__':
    main()