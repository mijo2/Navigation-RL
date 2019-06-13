# Navigation-RL
This is implementation of a RL agent that learns to play the Banana Collector game.

## Creating an environment that fulfills the requirements 

After creating the conda environment using the commands mentioned below and activating it, you will be able to run the code in this repository.

### Creating the conda environment

`conda create --name bananas python=3.6`\\
`conda activate bananas`

### Installing the requirements

Note - If you are having a problem with loading unityagents library in python or problem with loading the Banana environment, the following these steps of creating the environment, activating the created environment and installing the requirements mentioned in requirements.txt may help. Sometimes, the environment may not load from jupyter notebook, so run it from python script instead.

`pip install -r requirements.txt`

## Training the agent

Simply run the train.py script using the following command

`python train.py`

The above script runs the training for 1800 episodes and saves the data in the form of a dictionary as a file named as "checkpoint.pth"

## Testing the agent

Running the test.py runs loads the model saved by the train.py script and runs 10 episodes with the trained Q Network which takes a state and returns best action that maximises the action value function.

`python test.py`
 
