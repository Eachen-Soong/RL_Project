import matplotlib.pyplot as plt
import gym
import torch
from model import SAC_Agent
import numpy as np

# File path for the log directory
log_dir = 'log/'
# Open the text file containing the logged scores
with open(log_dir + 'mountaincar_score.txt', 'r') as score_txt:
    # Read the score data from the file
    data = score_txt.readlines()

# Convert the string scores to floating point values
score = [float(s.strip()) for s in data]  # Using strip() to remove possible whitespace

# Plot the scores using matplotlib
plt.plot(score)
plt.title('MountainCarContinuous-v0 Task Performance')  # Set the title of the plot to match the environment
plt.xlabel('Episode')  # Set the x-axis label
plt.ylabel('Score')  # Set the y-axis label
plt.savefig('./MountainCarContinuous-v0 Task Performance.png')  # Save the plot with an updated file name
plt.show()  # Display the plot

# Create the MountainCarContinuous-v0 environment with visualization enabled
env = gym.make('MountainCarContinuous-v0')

# Initialize a trained SAC agent
agent = SAC_Agent()

# Load the trained weights for the agent (ensure the weights file is correct and exists)
model_save_dir = 'saved_model/'
episode_number = 190  # Update this number based on your saved models
weights_filename = model_save_dir + "sac_actor_EP" + str(episode_number) + ".pt"
agent.PI.load_state_dict(torch.load(weights_filename, map_location=agent.DEVICE))

# Evaluate the policy
state = env.reset()
done = False

while not done:
    env.render()  # Render the environment for visualization
    
    # Convert the state to tensor and add a batch dimension
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.DEVICE)
    
    # Get action from the policy
    action, _ = agent.choose_action(state_tensor)
    action = action.detach().cpu().numpy().squeeze()
    action = np.array([action])

    # Perform action in the environment
    next_state, reward, done, _ = env.step(action)

    # 更新状态
    state = next_state

    # 如果环境已完成，则重置环境
    if done:
        state = env.reset()

env.close()  # Close the environment when done
