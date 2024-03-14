# RL_Othello

A reinforcement learning project on the game Othello

## Environment Overview

Our project includes 2 environments, one using OpenAI's Gymnasium, and the other we created from scratch which we dubbed "pygame" as we originally planned to use the pygame library to add graphical rendering (which was left out for time constraints). The Gymnasium environment resides in the Othello class in Othello.py and the "pygame" environment resides in the OthelloGame class in othelloGame.py. They both extend a common abstract Environment class (environment.py) so that they have common methods that we can call on both for training and testing. Additionally, othelloGame should be playable on a human against human setup albeit with separate methods, and does allow tile placement selection both by moving a cursor around (similar to the Atari game) and directly selecting valid coordinates.

## Algorithms Overview

We designed and implemented 6 reinforcement learning agents: DQN, Double DQN, Dueling DQN, Deep Sarsa, Double Deep Sarsa, and Dueling Deep Sarsa. These can be found in dqn.py and sarsa.py for their respective learning types. Each of these classes extend from an abstract agent class for the same reasons we used one for environment. The "normal", "double" and "dueling" categories implement the neural networks used by their respective papers that we referenced (see paper), and "dqn" (q learning) and "sarsa" implement the training and evaluation formulas for their respective algorithms learned in class. In order to accomodate both q learning and sarsa in the same training and testing code, we find the next state and next action regardless, yet the non sarsa agents simply ignore this argument.

## Deep Learning overview

We built the neural networks for these models in the neuralNet.py file using either a PixelNeuralNet or StateNeuralNet subclass of BaseNeuralNet depending on which environment, since the gymnasium can only return a grid of pixel values and the pygame environment can only return an array of actual tile values. The ReplayMemory class serves as the buffer of game state and action tensors using pytorch Tensor objects and stores individual experiences and retrieves batches of random past experiences for the agent to learn from.

## Other Features

We also created a MetricLogger class (log.py) to keep track of the cumulative rewards, q values, losses, and lengths of episodes, as well as to create charts from this data to illustrate the learning process. The graphs are saved to PNGs and the metrics to a CSV in a folder structure based on the date/time within a common folder logs/.

The models of the agents themselves, including all network weights and hyperparameters, are also saved to files, both post training and periodically to the trained_models/ folder with a similar inner folder structure. Both the logs and the model files contain the agent name so as not to overwrite each other when executing from the same timestamp. Recordings, taken from the gymnasium environment, are saved in the recordings/ folder. There is also a LogFileAnalyzer class in plots.py that can take the logs of hand selected models and combine them, per metric, into graphs which are included with this submission.

We train all the models on both environments in train.py, called from main.py, using a run_agents method that accomodates both q learning and sarsa algorithms as well as optionally training or simply testing the agents. This allows us to use the same code for every run, no matter the environment, learning paradigm, network type, or training/testing so that we always use consistent code, only adapted as needed so that most aspects of the programming environment stays neutral.

## Getting Started

1. Install [Python 3.11](https://www.python.org/downloads/release/python-3118/)
2. Optional: Create a virtual environment

   1. Install venv: `python -m pip install venv`
   2. Create venv: `python -m venv venv`
   3. Activate venv: Run `venv\Scripts\activate`
3. Install the requirements: `python -m pip install -r requirements.txt`
4. Modify the parameters in the main.py file
5. Run the script to begin: `python -m main`

## Running the code explanation

1. To try this project, first ensure that you are using Python 3.11. We use certain syntax, such as unpacking tuples into subscripts, and libraries, namely pytorch, that require this specific version.
2. Next, install the necessary requirements in requirements.txt.
3. It may be helpful to build a virtual environment for python.
4. Modify the parameters near the top of the main.py file as seen here:

   - ```
     # AGENT PARAMS
     EPSILON = 1
     EPSILON_DECAY_RATE = 0.999975
     EPSILON_MIN = 0.1
     ALPHA = 0.00025
     GAMMA = 0.9
     SKIP_TRAINING = 100
     SAVE_INTERVAL = 100
     SYNC_INTERVAL = 500

     # TRAINING PARAMS
     EPISODES = 3_000
     MAX_STEPS_PYGAME = 30
     MAX_STEPS_GYM = 1000

     MEMORY_CAPACITY = 100_000
     BATCH_SIZE = 64
     ```
   - Description of parameters:

     - EPSILON: The starting epsilon for training
     - EPSILON_DECAY: The rate at which the epsilon will decay at each time step and continues through multiple games
     - EPSILON_MIN: the minimum epsilon the decay shoud stop at even if training episodes continues
     - ALPHA: the learning rate of the agent
     - GAMMA: the discount factor between immediate and future rewards
     - SKIP_TRAINING: The amount of steps/experiences to store in memory before starting to train. Minimum amount should be BATCH_SIZE.
     - SAVE_INTERVAL: The episodic frequency at which to save the models weights for re-loading purposes.
     - SYNC_INTERVAL: The step frequecy at which to sync the main network parameters to the target network parameters.
     - EPISODES: The amount of training/evaluating episodes
     - MAX_STEPS_PYGAME: The numper of max steps for an episode in the "Pygame" Custom Othello environment
     - MAX_STEPS_GYM: The number of max steps for an episode in the Gymnasium Othello environment
     - MEMORY_CAPACITY: The capacity of the replay buffer. How many steps can it store.
     - BATCH_SIZE: The batch size of the recall method in the replay buffer. How many steps should go through the network as once.
5. Finally, execute the program using `python main.py` which will set up both environments, all 6 agents, and run all agents on both environments followed by immediately testing them.

   - This may take differing lengths of time, however we found that training one agent on the pygame environment for 3000 episodes, or games, takes about 1-2 hours and the same agent, for the same number of games, on the gymnasium environment 20-30 hours.
   - This is because while our pygame environment allows the agent to directly select coordinate positions on the board by an index, which is filtered to valid moves, the gymnasium environment requires that the agent move a cursor around before placing a tile, adding another layer of complexity to the challenge as it has to learn both how to place a tile and how to play the game.
   - We all believe it would benefit from much longer testing, and further refinement of the learning hyperparameters, however we unfortunately did not have time to test this to the extent that we would like.
