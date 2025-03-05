import os
import sys
import gymnasium as gym
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent


def output_to_terminal(output=True):
    if output:
        sys.stdout = open('CON', 'w')
    else:
        sys.stdout = sys.__stdout__


def train(episodes, hyperparameters, env_name, run_number):
    env = gym.make(env_name)
    
    agent = Agent(**hyperparameters)

    save_path = "agent_checkpoint.pth"

    rewards_per_episode = []
    q_values_per_episode = []
    q_loss_per_episode = []
    epsilon_per_episode = []

    for i in range(episodes):
        print("\n" * 2)
        print("=" * 57)
        print("Run Number:", run_number+1)
        print("Episode:", i)
        print("-" * 57)
        
        state, _ = env.reset()
        
        done = False
        
        total_reward = 0

        while not done:
            action_idx = agent.choose_action(state)
            action = np.array([agent.action_space[action_idx]])
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated 

            agent.replay_buffer.add(state, action_idx, reward, next_state, done)

            agent.learn()

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        q_values_per_episode.append(agent.get_q_values())
        q_loss_per_episode.append(agent.get_q_loss())
        epsilon_per_episode.append(agent.epsilon)

        print(f"Total Reward: {total_reward}")
        print(f"Q-values: {agent.get_q_values()}")
        print(f"Q Loss: {agent.get_q_loss()}")
        print(f"Epsilon: {agent.epsilon:.3f}")
        print()
        print("-" * 57)

        if i % agent.target_update == 0:
            agent.update_target_network()

        agent.decay_epsilon()

    agent.save(save_path)

    save_plot(rewards_per_episode, hyperparameters, "rewards_per_episode", run_number)
    save_plot(q_values_per_episode, hyperparameters, "q_values_per_episode", run_number)
    save_plot(q_loss_per_episode, hyperparameters, "q_loss_per_episode", run_number)
    save_plot(epsilon_per_episode, hyperparameters, "epsilon_per_episode", run_number)
    
    env.close()


def simulate(episodes, hyperparameters, env_name):
    checkpoint_path = "agent_checkpoint.pth"

    env = gym.make(env_name, render_mode="human")

    agent = Agent(**hyperparameters)
    agent.load(checkpoint_path)

    state, _ = env.reset()

    start_simulation_at = max(0, episodes-1)

    for i in range(episodes):
        if i < start_simulation_at:
            continue

        print("\n" * 2)
        print("=" * 57)
        print(f"Simulation Episode: {i + 1}")
        print("=" * 57)

        done = False

        while not done:
            with torch.no_grad():
                action_idx = agent.choose_action(state)
                action = np.array([agent.action_space[action_idx]])

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            env.render()

            state = next_state if not done else env.reset()[0]

    env.close()



def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def save_plot(data, hyperparameters, data_name, number, smooth=True, window_size=50):
    directory = os.path.join("plots", data_name)
    os.makedirs(directory, exist_ok=True)

    number += 1
    filename = f"{data_name}_{number}"

    xlabel = "Episodes"
    ylabel = "Values"

    label = f"hs={hyperparameters['hidden_size']}, lr={hyperparameters['lr']}, γ={hyperparameters['gamma']}"

    plt.figure(figsize=(7, 5))

    if smooth:
        smoothed_data = moving_average(data, window_size)
        plt.plot(range(len(smoothed_data)), smoothed_data, label=label)
    else:
        plt.plot(range(len(data)), data, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(data_name)
    plt.legend(loc="best", fontsize=8)
    plt.grid(True)

    filepath = os.path.join(directory, f"{filename}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

