import copy
from cliffenv import CliffWalkingEnv_white, print_agent
import pandas as pd

class PolicyIteration:
    """ 策略迭代算法 """
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow  # self.v 是一个列表,初始化 v=0 
        self.pi = [[0.25, 0.25, 0.25, 0.25]
                   for i in range(self.env.ncol * self.env.nrow)]  # 用双层列表表示策略,初始化为均匀随机策略
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 折扣因子

    def policy_evaluation(self):  # 策略评估
        cnt = 1  # 迭代次数
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []  # 开始计算状态 s 下各 action 对应的 Q(s,a)
                for a in range(4):
                    qsa = 0
                    for data in self.env.P[s][a]:   
                        p, next_state, r, done = data
                        #* 在 cliffwalking 环境下,动作会导致确定的结果(下一个状态),p=1,所以只有一次循环,但在更复杂的环境下可能有多次
                        #* Anyway, sum(p)=1
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))   # bellman
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)  # V(s) 和 Q(s,a) 之间的关系
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v  # 遍历所有 state 后,统一更新 V(s)
            if max_diff < self.theta: 
                break  # 收敛
            cnt += 1
            # TODO: show V(s) after each iteration
            # for i in range(self.env.nrow):
            #     for j in range(self.env.ncol):
            #         print('%6.6s' % ('%.3f' % self.v[i * self.env.ncol + j]), end=' ')
            #     print()
            # print("---")
        print("Policy evaluation done after %d iterations" % cnt)

    def policy_improvement(self):  # 策略提升
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for data in self.env.P[s][a]:
                    p, next_state, r, done = data
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大 Q 值,让这些动作均分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print("Policy improvement done!")
        return self.pi

    def policy_iteration(self):  # 策略迭代
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)  # 将列表进行深拷贝,方便接下来进行比较
            new_pi = self.policy_improvement()
            if old_pi == new_pi: 
                break


if __name__ == '__main__':
    env = CliffWalkingEnv_white()
    # env = gym.make("CliffWalking-v0", render_mode="human")
    action_meaning = ['↑', '↓', '←', '→']
    theta = 1e-3
    gamma = 0.9
    agent = PolicyIteration(env, theta, gamma)
    agent.policy_iteration()
    print_agent(agent, action_meaning, list(range(37, 47)), [47])