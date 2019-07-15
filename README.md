# Navigation-RL
This is implementation of a RL agent that learns to play the Banana Collector game.

# Environment

The environment is a plane with yellow and blue bananas in the plane. The bananas are kept randomly in the plane.

![alt text](https://github.com/mijo2/Navigation-RL/blob/master/report/banana.gif)

### Goal

The goal is to collect as many yellow bananas as possible while avoiding blue bananas.

### State Space

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. 

### Action space

There are 4 possible actions:
* `0`: Forward
* `1`: Backward
* `2`: Left
* `3`: Right

### Rewards

The reward function looks something like below -
  1. `+1` for interaction with yellow banana
  2. `-1` for interaction with blue banana

# Getting Started

## Creating an environment(conda) that fulfills the requirements 

After creating the conda environment using the commands mentioned below and activating it, you will be able to run the code in this repository.

### Creating the conda environment

`conda create --name bananas python=3.6`

`conda activate bananas`

### Installing the requirements

Note - If you are having a problem with loading unityagents library in python or problem with loading the Banana environment, the following these steps of creating the environment, activating the created environment and installing the requirements mentioned in requirements.txt may help. Sometimes, the environment may not load from jupyter notebook, so run it from python script instead.

`pip install -r requirements.txt`

## Downloading the environment(task)

Download the environment from one of the links below. You need only select the environment that matches your operating system:

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
* (For Windows users) Check out this [link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

* (For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

# Instructions

## Training the agent

Simply run the train.py script using the following command

`python train.py`

The above script runs the training for 1800 episodes and saves the data in the form of a dictionary as a file named as "checkpoint.pth"

## Testing the agent

Running the test.py runs loads the model saved by the train.py script and runs 10 episodes with the trained Q Network which takes a state and returns best action that maximises the action value function.

`python test.py`
 
