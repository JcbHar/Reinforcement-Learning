import os
import sys
import gym
import torch
import time
import numpy as np
import pybullet as p
import pybullet_envs
import matplotlib.pyplot as plt

from agent import Agent
from models import Actor, Critic


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
    alpha_per_episode = []
    actor_loss_per_episode = []
    critic_loss_per_episode = []
    alpha_loss_per_episode = []
    q_values_per_episode = []
    target_q_values_per_episode = []
    next_state_q_values_per_episode = []
    log_prob_per_episode = []

    for i in range(episodes):
        print("\n" * 2)
        print("=" * 57)
        print("Run Number:", run_number+1)
        print("Episode:", i)
        print("-" * 57)
        
        states = env.reset()
        
        done = False
        
        total_reward = 0
        
        while not done:
            action = agent.choose_action(states, env.action_space.low[0], env.action_space.high[0])

            next_states, reward, done, info = env.step(action)

            agent.replay_buffer.add(states, action, reward, next_states, done)

            agent.learn()            
            
            states = next_states
            total_reward += reward
        
        rewards_per_episode.append(total_reward)
        alpha_per_episode.append(agent.get_alpha())
        actor_loss_per_episode.append(agent.get_actor_loss())
        critic_loss_per_episode.append(agent.get_critic_loss())
        alpha_loss_per_episode.append(agent.get_alpha_loss())
        q_values_per_episode.append(agent.get_q_values())
        target_q_values_per_episode.append(agent.get_target_q_values())
        next_state_q_values_per_episode.append(agent.get_next_state_q_values())
        log_prob_per_episode.append(agent.get_log_prob())
        
        print(f"Total Reward: {total_reward}")
        print(f"Alpha: {agent.get_alpha()}")
        print(f"Actor Loss: {agent.get_actor_loss()}")
        print(f"Critic Loss: {agent.get_critic_loss()}")
        print(f"Alpha Loss: {agent.get_alpha_loss()}")
        print(f"Q-values (Current State-Action): {agent.get_q_values()}")
        print(f"Target Q-Values: {agent.get_target_q_values()}")
        print(f"Next-State Q-Values: {agent.get_next_state_q_values()}")
        print(f"Log Probability: {agent.get_log_prob()}")
        print()
        print("-" * 57)

    agent.save(save_path)
        
    save_plot(rewards_per_episode, hyperparameters, "rewards_per_episode", run_number)
    save_plot(alpha_per_episode, hyperparameters, "alpha_per_episode", run_number)
    save_plot(actor_loss_per_episode, hyperparameters, "actor_loss_per_episode", run_number)
    save_plot(critic_loss_per_episode, hyperparameters, "critic_loss_per_episode", run_number)
    save_plot(alpha_loss_per_episode, hyperparameters, "alpha_loss_per_episode", run_number)
    save_plot(q_values_per_episode, hyperparameters, "q_values_per_episode", run_number)
    save_plot(target_q_values_per_episode, hyperparameters, "target_q_values_per_episode", run_number)
    save_plot(next_state_q_values_per_episode, hyperparameters, "next_state_q_values_per_episode", run_number)
    save_plot(log_prob_per_episode, hyperparameters, "log_prob_per_episode", run_number)

    
def simulate(episodes, hyperparameters, env_name, sleep_time):
    checkpoint_path = "agent_checkpoint.pth"

    env = gym.make(env_name, render=True)

    agent = Agent(**hyperparameters)
    
    agent.load(checkpoint_path)

    state = env.reset()

    start_episode = max(0, episodes - 10)
    end_episode = episodes
    
    for i in range(start_episode, end_episode):
        print("\n" * 2)
        print("=" * 57)
        print(f"Simulation Episode: {i+1}")
        print("=" * 57)

        done = False


        while not done:
            with torch.no_grad():
                action = agent.choose_action(state, env.action_space.low[0], env.action_space.high[0])

            next_state, reward, done, info = env.step(action)

            time.sleep(sleep_time)

            state = next_state if not done else env.reset()

    env.close()


def save_plot(data, hyperparameters, data_name, number):
    directory = os.path.join("plots", data_name)
    os.makedirs(directory, exist_ok=True)
    
    number += 1
    filename = f"{data_name}_{number}"
    
    xlabel = "Episodes"
    ylabel = "Values"

    label = f"hs={hyperparameters['hidden_size']}, actor_lr={hyperparameters['actor_lr']}, critic_lr={hyperparameters['critic_lr']}, alpha_lr={hyperparameters['alpha_lr']}, tau={hyperparameters['tau']}, γ={hyperparameters['gamma']}"

    plt.figure(figsize=(7, 5))

    episodes = range(len(data))
    plt.plot(episodes, data, label=label) 

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(data_name)
    plt.legend(loc="best", fontsize=8)
    plt.grid(True)

    filepath = os.path.join(directory, f"{filename}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

