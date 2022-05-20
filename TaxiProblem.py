#!/usr/bin/env python
# coding: utf-8

# # Sub-Task 2 (Taxi Problem):

# In[1]:


#Imports the necessary libraries 
import gym
import numpy as np
import sys
import random
import math
from time import sleep
from IPython.display import clear_output
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import itertools
import networkx


# In[2]:


"""
This code is adapted from the taxi problem tutorial at:

https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

Added a graph to visualize reward data

All credit goes to the authors.

"""


# ### Environment Setup

# In[3]:


#Creates the Taxi environment via gym
env = gym.make('Taxi-v3').env


# In[5]:


#Resets environment to a new, random state
env.reset()


# In[6]:


#Renders one frame of the environment for visualization 
env.render()


# In[7]:


# reset environment to a new, random state
env.reset()
#Renders one frame of the environment for visualization
env.render()

#Prints the size of the action space
print("Action Space {}".format(env.action_space))
#Prints the size of the state space
print("State Space {}".format(env.observation_space))


# In[8]:


#(taxi row, taxi column, passenger index, destination index)
state = env.encode(3, 1, 2, 0)
print("State:", state)

env.s = state
env.render()


# In[9]:


#Creates a reward table with # of states as rows # of actions as columns
#states x actions matrix
#0-5 represent taxi actions (south, north, east, west, pickup, dropoff)
#1.0 is environment probability at state 328
#The nextstate is the state we would be in if we take the action at this index of the dict
#Movement actions have -1 reward at state 328
#Pickup/dropoff actions have -10 reward at state 328
#Successful drop off is represented by T or F
env.P[328]


# ### Solving the Environment Without Reinforcement Learning

# In[13]:


#Solving the Environment without RL
#Set environment to illustration's state
env.s = 328  

epochs = 0
penalties, reward = 0, 0
episode_reward = 0

frames = [] # for animation
rewards_array = [] #for comparison w/qlearning 
penalties_array = []

done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    episode_reward += reward

    if reward == -10:
        penalties += 1
    rewards_array.append(episode_reward)
    penalties_array.append(penalties)
    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1
    
    
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))


# In[14]:


#Prints the reward table of the current state, states x actions matrix 
#States = rows
#Number of actions = columns 
env.P[state]


# In[17]:


#Creates a Graph of Cumulative Penalties Data
plt.title("Taxi Penalties Graph Random Search")
plt.plot(penalties_array, linewidth=1)
plt.ylabel('Cumulative Penalties')
plt.xlabel('Algorithm Iterations')
plt.show()


# In[18]:


#Creates a Graph of Cumulative Penalties Data
plt.title("Taxi Rewards Graph Random Search")
plt.plot(rewards_array, linewidth=1)
plt.ylabel('Cumulative Rewards')
plt.xlabel('Algorithm Iterations')
plt.show()


# In[130]:


#Visualisation in real time of the Taxi environment & actions
def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        
print_frames(frames)


# ### Q-Learning

# In[20]:


#Initializes the q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])


# In[21]:


get_ipython().run_cell_magic('time', '', '"""Training the agent"""\n\n# Hyperparameters\nalpha = 0.1\ngamma = 0.6\nepsilon = 0.1\n\n# For plotting metrics\nall_epochs = []\nall_penalties = []\n\nfor i in range(1, 100001):\n    state = env.reset()\n\n    epochs, penalties, reward, = 0, 0, 0\n    done = False\n    \n    while not done:\n        if random.uniform(0, 1) < epsilon:\n            action = env.action_space.sample() # Explore action space\n        else:\n            action = np.argmax(q_table[state]) # Exploit learned values\n\n        next_state, reward, done, info = env.step(action) \n        \n        old_value = q_table[state, action]\n        next_max = np.max(q_table[next_state])\n        \n        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)\n        q_table[state, action] = new_value\n\n        if reward == -10:\n            penalties += 1\n        state = next_state\n        epochs += 1\n        \n    if i % 100 == 0:\n        clear_output(wait=True)\n        print(f"Episode: {i}")\n\nprint("Training finished.\\n")')


# In[22]:


#Displays the q-table for state 328
q_table[328]


# ### Evaluate agent's performance after Q-learning

# In[23]:


total_epochs, total_penalties = 0, 0
episodes = 300

penalties_array = []
rewards_array = []

for _ in range(episodes):
    state = env.reset()
    episode_reward = 0
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)
        episode_reward += reward

        if reward == -10:
            penalties += 1

        epochs += 1
        penalties_array.append(penalties)
        rewards_array.append(episode_reward)

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")


# In[24]:


env.P[state]


# In[25]:


#Creates a Graph of Cumulative Penalties Data
plt.title("Taxi Penalties Graph with Q-Learning")
plt.plot(penalties_array, linewidth=1)
plt.ylabel('Cumulative Penalties')
plt.xlabel('Algorithm Iterations')
plt.show()


# In[26]:


#Creates a Graph of Rewards Data 
plt.title("Taxi Rewards Graph with Q-Learning")
plt.plot(rewards_array, linewidth=1)
plt.ylabel('Cumulative Rewards')
plt.xlabel('Algorithm Iterations')
plt.show()


# In[ ]:




