from cliffenv import CliffWalkingEnv_black, print_agent2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Sarsa:
    """ Sarsa算法 """
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])     #* Only the first occurrence is returned
        return action

    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error   # 增量更新


if __name__ == "__main__":
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv_black(ncol, nrow)
    np.random.seed(0)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
    num_episodes = 2000

    return_list = []  # 记录每一条序列的回报
    for i in range(10): 
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                state = env.reset()
                action = agent.take_action(state)
                done = False
                while not done:
                    next_state, reward, done = env.step(action)
                    next_action = agent.take_action(next_state)
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state, next_action)
                    state = next_state
                    action = next_action
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Sarsa on {}'.format('Cliff Walking'))
    plt.show()

    action_meaning = ['↑', '↓', '←', '→']
    print('The optimal policy by Sarsa:')
    print_agent2(agent, env, action_meaning, list(range(37, 47)), [47])