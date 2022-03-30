import numpy as np

import gym

import tensorflow.compat.v1 as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import random
from collections import deque
import time

from agent import Agent

tf.disable_v2_behavior()

if __name__ == '__main__':
    env = gym.make("LunarLander-v2")
    optimizer = Adam(learning_rate = 0.0001)

    agent = Agent(env, optimizer, batch_size = 64)
    state_size = env.observation_space.shape[0]

    timestep=0
    rewards = []
    aver_reward = []
    aver = deque(maxlen=100)


    for episode in range(1000):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            
            env.render()

            total_reward += reward
            
            agent.memorize_exp(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            timestep += 1
            
            
        aver.append(total_reward)     
        aver_reward.append(np.mean(aver))
        
        rewards.append(total_reward)
            
        agent.update_brain_target()

        agent.epsilon = max(0.1, 0.995 * agent.epsilon)
        print("Episode ", episode, total_reward)
        
    plt.title("Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(rewards)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(aver_reward, 'r')

    agent.brain_policy.save('./model.h5')