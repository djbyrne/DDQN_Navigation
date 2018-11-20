# DDQN_Navigation
This is my submission for the Udacity navigation project in the Deep Reinforcement Learning Nano Degree. This project uses Deep Q Networks to navigate a room collecting objects (bananas) while learning to avoid obstacles (blue bananas). 

## Improvements Added:
* Double Learning
* Duelling Network
* Prioritized Experience Replay

[GIF]

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

## Getting Setup

