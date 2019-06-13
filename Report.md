# Task

The task of this project is to create, train and test an agent that learns to play the banana collector game using Deep Q learning. 

# About the Environment

The environment is a plane with yellow and blue bananas in the plane. 

### Goal

The goal is to collect as many yellow bananas as possible while avoiding blue bananas.

### Rewards

The reward function looks something like below -
  1. +1 for interaction with yellow banana
  2. -1 for interaction with blue banana
  
### Introduction to brains

In unity environment, when there are multiple agents that are playing with or against one another, it is feasible to train a single Q value approximation for both of them. Both the agents will obviously have different states and may take different actions accordingly, but they will effectively use the same action-value function to approximate the values and imporve on the function accordingly together
