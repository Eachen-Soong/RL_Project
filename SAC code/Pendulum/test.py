import os
import gym
import torch
from model import SAC_Agent

# The script to test the SAC agent using a pretrained model
if __name__ == '__main__':

    # Define the file path for logging and saving the trained model's weights
    log_path = 'sac_actor.pt'
    weight_file_path = 'saved_model/' + log_path
    # Ensure the directory for the saved model exists, or create it
    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')

    # Initialize the SAC agent with the path to the trained weights
    agent = SAC_Agent(weight_file_path)

    # Create the environment
    env = gym.make('Pendulum-v1')

    # Reset the environment to get the initial state
    state = env.reset()
    step = 0  # Initialize the step count

    # Simulation loop
    while True:
        env.render()  # Render the environment to visualize the agent's behavior

        # Choose an action using the policy network
        action, log_prob = agent.choose_action(torch.FloatTensor(state))
        action = action.detach().cpu().numpy()  # Convert action to numpy array and detach from the graph

        # Take the action in the environment and observe the next state and reward
        state_prime, reward, done, _ = env.step(action)
        step += 1  # Increment the step count

        state = state_prime  # Update the state

        # Reset the environment if a certain number of steps have been taken
        if step % 200 == 0:
            state = env.reset()
            print("step: ", step)