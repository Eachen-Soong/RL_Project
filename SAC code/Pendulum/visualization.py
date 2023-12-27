import matplotlib.pyplot as plt
import gym
import torch
from model import SAC_Agent
import numpy as np

# File path for the log directory
log_dir = 'log/'
# Open the text file containing the logged scores
with open(log_dir + 'pendulum_score.txt', 'r') as score_txt:
    # Read the score data from the file
    data = score_txt.readlines()

# Convert the string scores to floating point values
score = [float(s.strip()) for s in data]  # Using strip() to remove possible whitespace

# Plot the scores using matplotlib
plt.plot(score)
plt.title('Pendulum-v1 Task Performance')  # Set the title of the plot
plt.xlabel('Episode')  # Set the x-axis label
plt.ylabel('Score')  # Set the y-axis label
plt.savefig('./Pendulum-v1 Task Performance.png')  # Save the plot as an image
plt.show()  # Display the plot

# 创建Pendulum环境
env = gym.make('Pendulum-v1', render_mode='human')

# 初始化训练过的SAC智能体
agent = SAC_Agent()

episode_number = 190
model_save_dir = 'saved_model/'
weights_filename = model_save_dir + "/sac_actor_EP" + str(episode_number) + ".pt"

agent.PI.load_state_dict(torch.load(weights_filename))

# 初始化环境
state = env.reset()

# 设置要运行的总时间步数
time_steps = 1000

for t in range(time_steps):
    env.render()  # 显示环境

    # 将状态转为tensor，并增加一个batch维度，因为模型期望批量输入
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.DEVICE)

    # 使用智能体选择动作
    action, _ = agent.choose_action(state_tensor)
    # 将动作转换为cpu上的numpy数组，并确保它是一个单值数组
    action = action.cpu().numpy().squeeze()
    # 如果动作变成了一个标量，将其转换回单值数组
    action = np.array([action])
    next_state, reward, done, _ = env.step(action)

    # 更新状态
    state = next_state

    # 如果环境已完成，则重置环境
    if done:
        state = env.reset()

# 关闭环境
env.close()