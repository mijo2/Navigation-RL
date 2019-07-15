# Task

The task of this project is to create, train and test an agent that learns to play the banana collector game using Deep Q learning. 

# About the Environment

The environment is a plane with yellow and blue bananas in the plane. 

![alt text](https://github.com/mijo2/Navigation-RL/blob/master/report/banana.gif)

### Goal

The goal is to collect as many yellow bananas as possible while avoiding blue bananas. The more the yellow bananas and the fewer the blue bananas, the more the score of the agent in an episode

### Rewards

The reward function looks something like below -
  1. +1 for interaction with yellow banana
  2. -1 for interaction with blue banana
  
### Introduction to brains

In unity environment, when there are multiple agents that are playing with or against one another, it is feasible to train a single Q value approximation for both of them. Both the agents will obviously have different states and may take different actions accordingly, but they will effectively use the same action-value function to approximate the values and improve on the function accordingly together.

![alt text](https://github.com/mijo2/Navigation-RL/blob/master/report/brain_image.png) 

### State representation

State of the agent is a vector of size 37. The state space contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. 

### Action representation

There are 4 possible actions:
* `0`: Forward
* `1`: Backward
* `2`: Left
* `3`: Right

## End Goal

The task is episodic, and in order to solve the environment, an agent must get an average score of +13 over 100 consecutive episodes.

# Learning Algorithm: Deep Q Learning

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

### DQN in a nutshell

What DQN does is simply:

  1. Perform an action based on epsilon-greedy policy
  2. Store a transition in the memory buffer
  3. Samples some transitions from the memory buffer and use the target network to estimate the target estimates for each transition
  4. Calculate the loss function based on policy values and target values
  5. Backpropagate the loss in the policy network
  6. Update the target network softly(compared to the policy network)

Here, policy network is synonymous to Local Network and target value = R(t) + gamma * max(Q'(s',a)) where Q'(s',a) is the target network value

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

These hyper-parameter values proved to be more effective and batch size 64 was the fastest one to train. Memory buffer size of 5th order proved to be efficient as larger memory buffers proved to be costly and slow. This learning rate was better compared to 1e-4 and 1e-5 based on speed(efficiency increase per unit time) of the training.

# Plot of Rewards

The plot scores vs episodes

![alt_text](https://github.com/mijo2/Navigation-RL/blob/master/report/train_results.png)

## Test results

Average score for 10 episodes = 15.9

# Ideas for Future work

A bunch of things can be done to improve the agent's performance:-

1. Incoporating prioritized experience replay in the agent has the potential to improve the agent's performance significantly.
2. Using newer approaches like dueling dqn, double dqn and Rainbow may yield better results.
3. Better exploration strategies can be used to improve the training efficiency

