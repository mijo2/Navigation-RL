from unityagents import UnityEnvironment
import numpy as np
from dqn_agent import Agent
from collections import deque
import torch

env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64", no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

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

SAVE_EVERY = 15

def dqn(agent, env, n_episodes=1800, max_t=10000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start
    global state_size
    global action_size
    base_epoch = 1
    
    checkpoint_exists = os.path.exists("data_checkpoint.chkpt")
    if checkpoint_exists:
        data = torch.load("data_checkpoint.chkpt")
        base_epoch = data["n_epochs"]
        agent.qnetwork_local = QNetwork(state_size, action_size).load_state_dict(data["parameters"])
        agent.qnetwork_target = agent.qnetwork_local
        scores = data["scores"]
        scores_window = data["scores_window"]

    for i_episode in range(base_epoch, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0] 
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}, Current score: {:2f}'.format(i_episode, np.mean(scores_window), score), end="")
    
        if i_episode % SAVE_EVERY == 0:
            data = {
                "parameters": agent.qnetwork_target.state_dict(),
                "n_epochs": n_episodes + base_epoch, 
                "scores": scores,
                "scores_window": scores_window
            }
            torch.save(data, "data_checkpoint.chkpt")
        
    return scores

agent = Agent(state_size, action_size)
scores = dqn(agent, env)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
plt.savefig("train_results.png")