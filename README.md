# DDQN_Navigation
This is my submission for the Udacity navigation project in the Deep Reinforcement Learning Nano Degree. This project uses Deep Q Networks to navigate a room collecting objects (bananas) while learning to avoid obstacles (blue bananas). 

![Environment](/images/Screen Shot 2018-11-20 at 06.50.19.png)

## Improvements Added:
* Double Learning
* Duelling Network
* Prioritized Experience Replay

## Environment
The environment uses Unity's ML Agents platform to create a simple navigation challenge. The agent navigate through a large square room learning to collect as many good bananas as possible while avoiding the bad bananas. Although this problem appears simpple, it is easy to imagine greater use cases for this type of agent, such as autonomous driving.

### Rewards:
Collecting a banana: +1
Hitting a bad banana: -1

### State Space:
37 features including the velocity of the agent and data collected from raycast sensors about the closeness of its surrounding objects (much like lidar on a car)

### Controls
* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

### Completion
The agent is considered solved when it can achieve an average score of 13+ over 100 episodes


### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Replace the current banana.app file with the correct version for your OS

3. Finally, run the setup.py file to install all dependencies for this project

## Running the project

All of the code is contained in the Navigation_Submission notebook. The dqn method has tags for double learning, duelling network and prioritised experience replay. These can all be set at runtime.


