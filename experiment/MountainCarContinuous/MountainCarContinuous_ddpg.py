import argparse
from typing import Any
import gym
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from utils import *
from networks import MLP


# this policy network only returns action, no value
class DefaultPolicy(nn.Module):
    def __init__(self, state_space_dim, action_dim, num_layers=4, hidden_dim=128) -> None:
        super(DefaultPolicy, self).__init__()
        self.mlp = MLP(state_space_dim, action_dim, num_layers, hidden_dim)
    
    def forward(self, x):
        return self.mlp(x)


class DefaultQNetwork(nn.Module):
    def __init__(self, state_space_dim, action_dim, num_layers=4, hidden_dim=128) -> None:
        super(DefaultQNetwork, self).__init__()
        self.mlp = MLP(state_space_dim + action_dim, 1, num_layers, hidden_dim)
    
    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        return self.mlp(x)
    
    def get_action_value(self, x, a):
        x = torch.cat([x, a], dim=1)
        return self.mlp(x)

    def get_action_value_batch(self, x, a):
        x = torch.cat([x, a], dim=1)
        return self.mlp(x)
    
    def get_action_value_batch_2(self, x, a):
        x = torch.cat([x, a], dim=1)
        return self.mlp(x)

# save, load, eval_step, eval_episode, eval_episodes, record_online_return, switch_task

class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0

    def close(self):
        close_obj(self.task)

    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        while True:
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            if ret is not None:
                break
        return ret

    def eval_episodes(self):
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
        self.logger.info('steps %d, episodic_return_test %.2f(%.2f)' % (
            self.total_steps, np.mean(episodic_returns), np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        ))
        self.logger.add_scalar('episodic_return_test', np.mean(episodic_returns), self.total_steps)
        return {
            'episodic_return_test': np.mean(episodic_returns),
        }

    def record_online_return(self, info, offset=0):
        if isinstance(info, dict):
            ret = info['episodic_return']
            if ret is not None:
                self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
                self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps + offset, ret))
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError

    def switch_task(self):
        config = self.config
        if not config.tasks:
            return
        segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
        if self.total_steps > segs[self.task_ind + 1]:
            self.task_ind += 1
            self.task = config.tasks[self.task_ind]
            self.states = self.task.reset()
            self.states = config.state_normalizer(self.states)

    def record_episode(self, dir, env):
        mkdir(dir)
        steps = 0
        state = env.reset()
        while True:
            self.record_obs(env, dir, steps)
            action = self.record_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            steps += 1
            if ret is not None:
                break

    def record_step(self, state):
        raise NotImplementedError

    # For DMControl
    def record_obs(self, env, dir, steps):
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave('%s/%04d.png' % (dir, steps), obs)


class DDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        self.replay.feed(dict(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=1-np.asarray(done, dtype=np.int32),
        ))

        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.replay.size() >= config.warm_up:
            transitions = self.replay.sample()
            states = tensor(transitions.state)
            actions = tensor(transitions.action)
            rewards = tensor(transitions.reward).unsqueeze(-1)
            next_states = tensor(transitions.next_state)
            mask = tensor(transitions.mask).unsqueeze(-1)

            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next)
            q_next = self.target_network.critic(phi_next, a_next)
            q_next = config.discount * mask * q_next
            q_next.add_(rewards)
            q_next = q_next.detach()
            phi = self.network.feature(states)
            q = self.network.critic(phi, actions)
            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            phi = self.network.feature(states)
            action = self.network.actor(phi)
            policy_loss = -self.network.critic(phi.detach(), action).mean()

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)


class DDPGAgent(object):
    def __init__(self, obs_shape, action_shape, gamma, device, model_type='default', name='DDPG'):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.gamma = gamma
        self.name = name
        self.model_type = model_type
        self.model = None
        if model_type == 'default':
            self.model = DefaultPolicy(obs_shape, action_shape).to(device)

    def choose_action(self, state):
        'Choose an action based on observation input'
        state = torch.from_numpy(state).float()
        action, state_value = self.model(state)

        # save to action buffer
        self.saved_actions.append(self.SavedAction(action, state_value))

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

        for (action, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(- action * advantage)

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
        return loss.item()

    def save(self, model_dir, step):
        'Save the model'
        torch.save(self.model.state_dict(), model_dir + '/' + self.name + '_ep' + str(step)+'.pth')

    def load(self, model_path):
        'Load the model'
        self.model = torch.load(model_path)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument('--train_eps', default=9999, type=int)
    parser.add_argument('--max_t', default=9999, type=int)
    parser.add_argument('--task', default='MountainCarContinuous-v0', type=str)
    parser.add_argument('--model_type', default='default', type=str)
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
    
    eps = np.finfo(np.float32).eps.item()
    running_reward = 10
    agent = DDPGAgent(obs_shape=env.observation_space.shape[0], action_shape=env.action_space.shape[0], gamma=args.gamma, model_type=args.model_type, device=args.device)
    optimizer = optim.Adam(agent.model.parameters(), lr=3e-2)
    logger = Logger(log_dir='./logs/'+args.model_type+'/'+args.task+'/a2c')

    for i_episode in range(args.train_eps):
        state = env.reset()
        ep_reward = 0
        for t in range(args.max_t):
            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            agent.rewards.append(reward)
            ep_reward += reward
            if done:
                break
            
        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        logger.log('train/episode_reward', ep_reward, i_episode)
        logger.log('train/running_reward', running_reward, i_episode)
        # perform backprop
        loss = agent.update(optimizer, eps)
        logger.log('train/loss', loss, i_episode)

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

    agent.save('./ckpt/'+args.task, i_episode)

if __name__ == '__main__':
    main()