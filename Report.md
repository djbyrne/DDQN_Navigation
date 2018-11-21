# Intro

This project looks at implementing the Deep Q Network (DQN) algorithm in order to solve a relatively simple navigation environment. 
This implementation used the Lunar Lander exercises as a foundation and was adapted to suite the Banana environment. As well as this, 
the following additional features were added:

* Double Learning
* Duelling Network
* Prioritized Experience Replay

This report will outline the experiments, techniques and results of implementing these various methods of DQN to solve the Banana environment.

# The environment

![Environment](https://github.com/djbyrne/DDQN_Navigation/blob/master/images/Screen%20Shot%202018-11-20%20at%2006.50.19.png)

The environment used for this project was built using the Unity [ml-agents](https://github.com/Unity-Technologies/ml-agents) framework.
The environment itself is quite simple. The goal is to collect as many good objects as possible (yellow bananas) while avoiding dangerous objects(blue bananas).
The agent recieves a positive reward of +1 for each good banana collected and a negative reward -1 for each blue banana it hits.

The agent is given 37 state features to identify what is going on in the world around it. These are made up of 37 features that contain the velocity of the agent and 
sensor data recieved from the raycasts shooting from the agent and returning the information about what the ray is hit such as distance and color of the object hit. 
This works in a simular way to Lidar on a self driving car.

The agent can decide to move in 1 of 4 directions for each time steps:
* North 
* South
* East
* West

# Deep Q Learning Bot
DQN introduces a lot of changes to the traditional Q learning algorithm and isn't just replacing the Q table with a neural network.
As described in Deep Minds groundbreaking paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf) there are 
several advancements that make this algorithm work. Below I will go through the core techniques used to successfully implement the DQN algorithm.

## Models

Like I mentioned previously, one of the big changes in how we approach Q learning is the introduction of using a neural network to replace the Q table. Instead of storing every Q value in a look up table corresponding to its state and action, we can use the neural network to approximate the Q value of any state/action pair. This allows us to tackle infinitely more complex tasks that were out of reach for the simple tabular approach.

This base implementation of this project uses a simple 2 layer multi layer perceptron(MLP). The model takes in the 37 state features as input and returns the best Q values for each of our 4 possible actions given the current state as output. We then take the max of these values as our best action. For the base model of this project I kept the same architecture and parameters as the lunar lander excerise in order to get a solid working agent to begin. The base model looks like the following:

| Layers           |Parameters           |
|:-------------:| :-------------:| 
| Dense Layer| 16| 
| ReLU activation| NA|   
| Dense Layer| 32| 
| ReLU activation| NA| 
|Dense Layer|4| 

## Memory

One of the main improvements provided by the DQN paper is the use of Experience Replay. This acts like episodic memory and allows our agent to continually learn from its previous experiences as opposed to throughing them away at the end of each step. The use of experience replay has a few steps:

1) We need to observe our environment for several iterations in order to fill up our memory buffer with enough experiences to learn from. At each time step we will add our experience to our memory buffer. This experience stores the state, action, reward, next state and done variables for that step.

2) When we have enough memories stored (at least equal to our batch size) we can start learning. At each time step, after we add our experience to our buffer we check and see if we want to perform a learning step (every 4 steps) if so we sample from our memory buffer

3)Once we have this sample of previous experiences we can train on them. This will allow our agent to learn how to correctly approxminate the Q values. The details of this method will be discussed in the next section

## Agent

Next we have our agent. This of course the main portion of the project and ties everything together. The details of the agent are in the notebook itself but I will give a breif overview of the key parts

The agent can be intialised with several augmentations. The base model is set by default. This used only the standard DQN moel. Double Learning, Duelling Networks and Prioritized Experience Replay can all be added and removed during initialization.

The step function in the agent is taken after the environment step function and shouldn't be confused. Here we take in the state, action, reward, next_state and done variables from the last step and adds it to our experience replay buffer memory. Then the agent carries out the learn function.

# Training



# Double Learning

# Duelling Network

# Prioritized Experience Replay

# Results

# Future Work
