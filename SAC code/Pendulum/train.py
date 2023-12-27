import os
import gym
import torch
import numpy as np
from model import SAC_Agent

# Main function to train the agent
if __name__ == '__main__':

    # Directories to save the trained model and log files
    model_save_dir = 'saved_model/'
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)  # Create directory for saving models if it doesn't exist
    log_save_dir = 'log/'
    if not os.path.isdir(log_save_dir):
        os.mkdir(log_save_dir)  # Create directory for saving logs if it doesn't exist

    # Initialize the environment and the SAC agent
    env = gym.make('Pendulum-v1')
    agent = SAC_Agent()

    # Set the number of episodes for training
    EPISODE = 200
    print_once = True  # A flag to control print statements
    score_list = []  # List to store scores for each episode

    # Training loop over episodes
    for EP in range(EPISODE):
        state = env.reset()  # Reset the environment and get the initial state
        score, done = 0.0, False

        # Loop for each step of the episode
        while not done:
            # Choose an action using the policy network
            action, log_prob = agent.choose_action(torch.FloatTensor(state))
            action = action.detach().cpu().numpy()  # Convert action to numpy array and detach from graph

            # Execute the action in the environment
            state_prime, reward, done, _ = env.step(action)
            # Store the experience in the replay buffer
            agent.memory.put((state, action, reward, state_prime, done))
            score += reward  # Update the cumulative reward
            state = state_prime  # Update the state

            # If enough experiences are collected, start training the agent
            if agent.memory.size() > 1000:
                print_once = False
                agent.train_agent()

        # Print the average score of the episode
        print("EP:{}, Avg_Score:{:.1f}".format(EP, score))
        score_list.append(score)  # Append the score to the list

        # Save the actor model every 10 episodes
        if EP % 10 == 0:
            torch.save(agent.PI.state_dict(), model_save_dir + "/sac_actor_EP"+str(EP)+".pt")

    # Save the scores to a text file
    np.savetxt(log_save_dir + '/pendulum_score.txt', score_list)