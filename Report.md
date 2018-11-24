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

The step function in the agent is taken after the environment step function and shouldn't be confused. Here we take in the state, action, reward, next_state and done variables from the last step and add it to our experience replay buffer. Then the agent carries out the learn function.

Inside the learn function we take a sample from our experience replay buffer and iterate through that sample of experience. For each of these we update the Q value corresponding to the state action pair. This is done by computing the loss between the target and the expected prediction. In the base model the target is the predicted Q value of the next states and choosing the best best action using our target model. We then calculate the discounted rewards of that target model prediction to form our new target. This is then compared to the local models prediction of the Q value given the initial state. After this we carry out a soft update of the target model.

The last function worth discussing is the act method. Here we take in the current state of the environment and get the best prediction from our local model. Next we use the epsilon greedy strategy to determine wether we use our models action or if we use a random action. 

# Training

The training portion of the notebook contains the main game loop iterating through the environment and utilising our agent. As with most machine learning problems, a lot of the improvements come from hyper parameter tuning. This can often taken longer than building the actual algorithm itself. Unfortunately I couldn't dedicate too much time to hyper parameter tuning and was only able to test a few changes for each type of model. Below are the parameters that I experimented with. Each parameter change was added in and tested individually in order to identify which parameter changes gave the best results.

| Parameter | Params 1 |  Params 2  |
|:-------------:| :-------------:| :-------------:| 
|Layer 1    | 16     |   32     |
|Layer 2    | 32     |   64     |
|Learning Rate    | 0.0005     |   0.0001     |
|Batch Size    | 64     |   32     |
|Buffer Size    | 100000     |  200000     |


The experiments showed that the hyperparameters didnt make a huge improvement, with most models reaching a stable score of 13 within 800 episodes and a score ~15-17 after 2000. The variation on the agents final score was mostly influenced by the type of model used. The hyperparameters mainly effected how quickly the agent could reach a score of 13+. For example, the base DQN with the initial set of parameters learned to get a score of 13+ within ~450 episodes where as the same model with the second set of parameters took ~700 episodes. Both of these models achieved a final average score of 16 after 2000 episodes and could achieve the max score of 25. I believe that the model is capable achieving a better highscore with correct hyperparameter tuning and more training episodes. Below is the training graph of the base DQN agent running with the 2nd set of hyperparameters.

![DQN_2000_Episodes](https://github.com/djbyrne/DDQN_Navigation/blob/master/images/dqn_2000.png)


# Double Learning

The first addition to the base dqn model was the implementation of double learning. This technique was introduce by Hado Van Hasselt in [Deep Reinforcement Learning with Double Q Learning](https://arxiv.org/abs/1509.06461) in order to fix the over estimation problem found in the base DQN model.

Overestimation is caused by the max operator inside our update function. This operator is what lets us pick the best Q value when updating our agent and propogates that Q value to other states. Heres the problem, our Q values are not always accurate and can be inherently noisy for periods of time. For example when the agent has just started training, most of its Q values will be wrong. By picking the max of a noisy sample of Q values and then subsequently building upon those values causes our agent to overestimate.

Double learning fixes this by using two sets of parameters. One set is to pick the action and the other set is to evaluate it. The parameters are our local and target networks. Both of these networks must agree on the action being taken, otherwise the Q value is not that high. This prevents overestimated Q values from being propogated further.

Inside the learn method of the agent I have included an option to use double learning for finding the Q values and the base method. Using double learn we can break the process down into these steps

*  1) Predict the Q values using the local model given the current state, this is our expected output

*  2) Predict the best action given the next state using the local model

*  3) Predict the Q values of the next state using the best action calculated by the local model in step 2

*  4) Calculate the updated rewards based on these predictions 

      <code>
      rewards+(gamma * next_state_values.detach() * (1-dones))
      </code>
        
*  5) The loss is then calculated between the expected values and our target values that we just calculated.

The results of adding double learning to the base model didnt seem to have much of an effect. Both models were able to achieve a score of ~16 after 2000 episodes. The double Q learning did take a few more episodes to make initial progress. The base model could reacha score of 13+ after just 430 episodes, where as the double learning model was closer to the 460. This small increase in learning time can be attributed to the model reducing the overestimations of the Q values during the early stages of training. This means that the agent wont make as many big jumps in training, but it will also be more robust and consistent. I think the reason I didn't see massive differences in the two models is because the environment is quite simple and the real benefit of double learning isnt seen here as opposed to other more complex environments.

![DQN_2000_Episodes](https://github.com/djbyrne/DDQN_Navigation/blob/master/images/ddqn_2000.png)

# Duelling Network

Another addition to the DQN model is the use of duelling networks. The methodology and results of this addition can be found in the paper [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581). Duelling alter our existing neural network from having a single head layer, to having two. One head calculates the value of the given state where as the second head calculates the advantage of the state/action pair. Finally we calculate the Q values by combining the output of our value and advantage layers.

![Duelling network](https://cdn-images-1.medium.com/max/1600/0*280wCeKlu11zvztQ.jpg)


The idea behind seperating the value and advantage functions is that throughout training the value of most states doesn't vary across actions. By directly estimating these values we can improve the generalisation of our learning agent.

In my implementation I made a second model for the duelling network. This model used the same 2 dense layers to begin with. The output of the second dense layer is then fed into each of my head layers for the value and advantage. In the forward function of my model I then combine the output of the 2 head layers to return the Q value

<code>
    q = v.expand_as(a) + (a - a.mean(1, keepdim=True).expand_as(a))  
</code>

As seen from the results below, there is a small improvement by using the dueling network for this environment. This agent was able to reached a score of 13+ in 485 episodes and achieve a 100 episodes average score of 17.30.This is roughly +1 more than the previous model.

![Duelling Network](https://github.com/djbyrne/DDQN_Navigation/blob/master/images/correct_duelling17.png)

# Prioritized Experience Replay

For the final addition to the base DQN agent I implemented Prioritized Experience Replay(PER) introduced in the paper [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) in 2016. The idea behind PER is to not only use past experiences but to identify which experiences the agent can learn the most from. 

Some experiences are simply more valuable to learn from than others. This is similar to how we learn as humans. We will learn much more from a mistake or loss than we will from an average experience. Unfortunately these highly valuable experiences are much rarer, meaning that by just random sampling we will be less likely to use these experiences for training.

PER assigns a priority to each experience in order to use more valuable experiences for learning. One method of assigning priority to an experience is by using the TD error delta along with a small constant "e" to ensure that priorities are never 0 and subsequently never picked. As well as this we add a second hyperparameter "a" used to some unifromed samply. We multiply our priority to the power of "a". 

The higher the error, the more we can learn. When sampling we use the TD error to determine a sampling probability. This is done by selecting an experience that is equal to the priority value being used. This is then normalised by all priority values inside the replay buffer all raised to the power of "a". When an experience is picked we then update that experiences priority with the new TD error of the latest Q values.

When updating our sampling priorities there are a few things we need to consider. Our sampling must match the underlying distribution of our data set. With the normal experience replay method this isn't a problem as we are always sampling randomly. However with PER we are not and can run into the problem of over fitting to a small subsection of our data that we have deemed as "prioritized". To fix this we introduce a sampling weight which is 1 over the buffer size multiplied by 1 over the sampling probabilities raised to the power of another hyperparameter beta. 

[Update Sampling Weights](https://www.google.ie/url?sa=i&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwi5yprBpu3eAhVNzqQKHe41DmgQjRx6BAgBEAU&url=https%3A%2F%2Fmedium.com%2Farxiv-bytes%2Fsummary-prioritized-experience-replay-e5f9257cef2d&psig=AOvVaw36_M7dajVjGfG97vXbJGi_&ust=1543158193505676)

Beta is used to determine how much these weights effect learning. As we get to the end of training we want these weights to be more important when updating. As such we steadily increase the value of beta over time. The update function for the sampling probability weights can be seen below.

<code>
      weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
      
 </code>
 
 
The results from using PER were not amazing. It seemed that the agent performed slightly worse when using PER as opposed to the standard replay buffer. I suspect that is due more to my implementation than the technique itself. 



This was definitely the most complicated section of the assignment for me and I still require some more time with PER. I should point out that a lot of my understanding came from the implementation of PER found [here](https://github.com/rlcode/per/blob/master/cartpole_per.py). Ideally I would have liked to spend more time working on PER, however outside life has forced me to cut this assignment a little short. As such any feedback on this portion of the topic would be greatly appreciated.



# Results

DDDQN 509, score of 17.4, high score frequently
Graph of DQN using base weights

Graph of Double

Graph of Duelling

Graph of Rainbow

# Future Work

More tuning

AutoML


