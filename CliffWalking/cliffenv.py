import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

class CliffWalkingEnv_white:
    """ CliffWalking 白盒环境, 已知 P_sa"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # colomn of grid world 
        self.nrow = nrow  # row of grid world 
        # 转移矩阵 P[state][action] = [(p, next_state, reward, done)] 包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # P 用双层列表表示
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4 actions. change[0]:上,change[1]:下,change[2]:左,change[3]:右   坐标轴原点在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # 其他位置,计算下一个 state
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1    # 每一步的 reward 为 -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P
    
    
class CliffWalkingEnv_black:
    """ CliffWalking 黑盒环境, 未知 P_sa"""
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1  # 记录当前智能体位置的纵坐标

    def step(self, action):  # 外部调用这个函数来改变当前位置
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1     #  reward function 也是已知的
        done = False
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):  # 回归初始状态
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x


def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("V(s):")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()
    # TODO: show V(s) with pandas
    # v = np.array(agent.v).reshape(agent.env.nrow, agent.env.ncol)
    # df = pd.DataFrame(v)
    # print(df)


    print("Policy:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


def print_agent2(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


if __name__ == '__main__':
    env = CliffWalkingEnv_white()
    print("P_sa:")
    for i in range(env.nrow):
        for j in range(env.ncol):
            print(i * env.ncol + j, end=': ')
            for a in range(4):
                print(env.P[i * env.ncol + j][a], end='; ')
            print()
