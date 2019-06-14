from unityagents import UnityEnvironment
import numpy as np
from dqn_agent import Agent
from model import QNetwork
import torch

env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64", no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

data = torch.load("data_checkpoint.chkpt")

trained_agent = QNetwork(state_size, action_size).load_state_dict(data['parameters'])

VAL_TESTING  = 10
scores = []
env_info = env.reset(train_mode=True)[brain_name]
state = env_info.vector_observations[0]

for i in range(VAL_TESTING):
    state = env.reset()
    score = 0
    while True:
        action = trained_agent(state).max(1)[1]
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0] 
        state = next_state
        score += reward
        if done:
            scores.append(score)
            break 

print(f"For {VAL_TESTING} episodes, the average score is {np.mean(scores)}")
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig("test_results.png")
