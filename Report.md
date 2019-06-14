# Task

The task of this project is to create, train and test an agent that learns to play the banana collector game using Deep Q learning. 

# About the Environment

The environment is a plane with yellow and blue bananas in the plane. 

![alt text](https://github.com/mijo2/Navigation-RL/blob/master/report/banana.gif)

### Goal

The goal is to collect as many yellow bananas as possible while avoiding blue bananas.

### Rewards

The reward function looks something like below -
  1. +1 for interaction with yellow banana
  2. -1 for interaction with blue banana
  
### Introduction to brains

In unity environment, when there are multiple agents that are playing with or against one another, it is feasible to train a single Q value approximation for both of them. Both the agents will obviously have different states and may take different actions accordingly, but they will effectively use the same action-value function to approximate the values and imporve on the function accordingly together

![alt text](https://github.com/mijo2/Navigation-RL/blob/master/report/brain_image.png)

For example - Suppose there are multiple agents 

### State representation

State of the agent is a vector of size 37.

### Action representation

There are 4 possible actions:
1. Forward
2. Backward
3. Left
4. Right

# Deep Q Learning

For this task, deep Q learning is used to train the agent.

## Introduction to Q Learning 

### Policy Iteration

For each time step, we will select an action based on the epsilon-greedy policy, and then estimate the action value functions according to the action taken and the state the agent is currently in and the next state.

![alt_text](https://github.com/mijo2/Navigation-RL/blob/master/report/MC_control.png)

### Action value function estimation

To estimate the action value function, we will use Q-learning formula - 

Q(s,a) = Q(s,a) + alpha * ( R(t) + gamma * max(Q(s',a)) - Q(s,a) )

Here the Q(s,a) is a function approximation namely, a deep neural network architecture that approximates the action value function based on the state and action taken.

### Epsilon greedy policy

With probability epsilon, we will select a random action to be taken and with probability 1-epsilon, we will take an action that maximises action value function for a particular given current state. 

![alt_text](https://github.com/mijo2/Navigation-RL/blob/master/report/epsilon-greedy.png)

## Deep Q learning basics

### Memory samples

We are not going to estimate the action value function based on the current state, action taken and next state but rather, we will append the transition(also known as experience) (state, action, reward, next_state, done) into a memory buffer. And at each time step, if the memory buffer has enough experiences stored, we will take a sample of random batch of experiences use these experiences to train our agent(namely estimate our action value function).

### Local Network and Target Network

Since, here we are bootstrapping when we are estimating the action value functions, there maybe problems because we are updating the Q values and as well as using them to update. This is like a dog chasing its own tail and the convergence of the action value functions slows down due to this point. Namely, the targets aren't fixed. Here, the target is (R(t) + gamma * max(Q(s',a)).

So, to deal with this problem, we are going to use two networks. Firstly, the local network which is always updated according to the formula. But the target value used (namely the max term), we use another network to get that value. 

Q(s,a) = Q(s,a) + alpha * ( R(t) + gamma * max(Q'(s',a)) - Q(s,a) )

where Q is the local network while Q' is the target network. 

And at every time step, the Q' network is updated but ever so slightly using the formula:

Q' = beta * Q' + (1-beta) * Q

where beta is in order of 1e-3.

## Architecture

The architecture of the Q Network contains 4 layers, two hidden layers, one input layer and one output layer.

Input layer - 37 
Hidden layer 1 - 64
Hidden layer 2 - 64
Output layer - 4

![alt_text](https://github.com/mijo2/Navigation-RL/blob/master/report/nerual_architecture.png)

## Hyperparameters

1. Batch size: 64
2. Soft update parameter TAU: 1e-3
3. Learning rate: 5e-4
4. GAMMA: 0.99
5. Memory buffer size: 1e5

# Results

## Training results 

The plot scores vs episodes

![alt_text](https://github.com/mijo2/Navigation-RL/blob/master/report/train_results.png)

## Test results

Average score for 10 episodes = 15.9

