import torch
import numpy as np
import gym
import time
import sys
import pybullet as p
import pybullet_envs
import matplotlib.pyplot as plt
import os

from agent import Agent
from models import Actor, Critic

def output_to_terminal():
    sys.stdout = open('CON', 'w')


def output_to_notebook():
    sys.stdout = sys.__stdout__


def train(episodes, hyperparameters, env_name):
    env = gym.make(env_name)

    agent = Agent(**hyperparameters)
                  
    rewards_per_episode = []

    save_path = "agent_checkpoint.pth"
    
    for i in range(episodes):
        print("EPISODE:", i)
        states = env.reset()
        
        done = False
        
        total_reward = 0
        while not done:
            action = agent.choose_action(states, env.action_space.low[0], env.action_space.high[0])
            #print("Chosen action:", action)
                #assert not torch.isnan(action).any(), "Action contains NaN!"
            next_states, reward, done, info = env.step(action)
            done = done
            
            total_reward += reward

            agent.learn(states, action, reward, next_states, done)
            
            states = next_states
            
        rewards_per_episode.append(total_reward)

        agent.save(save_path)
        
    print("TRAINING COMPLETE")
    return rewards_per_episode


def simulate(episodes, hyperparameters, env_name, sleep_time, num_symbols=120):
    checkpoint_path = "agent_checkpoint.pth"
    
    env = gym.make(env_name, render=True)
    
    agent = Agent(**hyperparameters)
    agent.load(checkpoint_path)
    
    state = env.reset()

    start_time = time.time()
    
    print("\n" * num_symbols)
    
    for i in range(episodes):
        action = agent.choose_action(state, env.action_space.high[0], env.action_space.low[0])
        
        next_state, reward, done, info = env.step(action)
        #env.render()

        time.sleep(0.05)
        
        if done:
            state = env.reset()
        else:
            state = next_state
    
    env.close()


def initialise(terminal_output=False):
    if terminal_output:
        output_to_terminal()
    else:
        output_to_notebook()  

def save_plot(rewards_over_time, hyperparameters, filename):
    os.makedirs("plots", exist_ok=True)
    
    label = f"H={hyperparameters['hidden_size']}, ALR={hyperparameters['actor_learning_rate']}, CLR={hyperparameters['critic_learning_rate']}, γ={hyperparameters['gamma']}"

    plt.figure(figsize=(7, 5))
    plt.plot(rewards_over_time, label=label)

    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Rewards Over Time")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True)

    filepath = f"plots/{filename}.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved: {filepath}")


def plot_rewards(rewards_over_episodes):
    plt.plot(rewards_over_episodes)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards Over Episodes')
    plt.show()