from othello import Othello, RENDER_MODE, OBS_SPACE

from dqn import DDQN, DQN, DuelDQN
from sarsa import SARSA, SARSA_DDQN, SARSA_DuelDQN

from train import train_QLearning, train_SARSA
from test import test_agent

from pathlib import Path
import time
from log import MetricLogger

date = time.strftime('%m-%d-%Y')
t = time.strftime('%H_%M_%S')

# Model saving path
save_model_path = Path("trained_models") / date / t
save_model_path.mkdir(parents=True, exist_ok=True)

# Logs saving path
save_logs_path = Path("logs") / date / t
save_logs_path.mkdir(parents=True, exist_ok=True)
logger = MetricLogger(save_logs_path)

save_recordings_path = Path("recordings") / date / t
save_recordings_path.mkdir(parents=True, exist_ok=True)

# AGENT PARAMS
EPSILON = 1
EPSILON_DECAY_RATE = 0.9
EPSILON_MIN = 0.01
ALPHA = 0.01 #0.00025
GAMMA = 0.9
SKIP_TRAINING = 1_000 
SAVE_INTERVAL = 50_000
SYNC_INTERVAL = 10_000

# TRAINING PARAMS
EPISODES = 100
MAX_STEPS = 100

MEMORY_CAPACITY = 100_000


if __name__ == '__main__':
    """
    RENDER_MODE = ["human" or "rgb_array"] Human mode makes you see the board where as RGB will just do it in the background
    OBSERVATION_TYPE = ["RGB", "GRAY", "RAM"] # RGB for color image, and GRAY for grayscale. RAM needs a diff env so that will cause an error for rn
    VIDEO = Bool  Note: Only works with render_mode: rgb_array. render_mode is hardcoded alread if this is True
    """
    othello = Othello(render_mode=RENDER_MODE.RGB, observation_type=OBS_SPACE.RGB, record_video=False, save_recordings_path=save_recordings_path)

    # Check the state of gymnasium nvironment after n_steps
    # othello.run(n_steps=1000)

    environment = othello

    # Define agent parameters once so it's not quite so verbose
    params = {'state_shape' : environment.state_space,
              'num_actions' : environment.num_actions,
              'epsilon' : EPSILON,
              'epsilon_decay_rate' : EPSILON_DECAY_RATE,
              'epsilon_min' : EPSILON_MIN,
              'alpha' : ALPHA,
              'gamma' : GAMMA,
              'sync_interval' : SYNC_INTERVAL,
              'skip_training' : SKIP_TRAINING,
              'save_interval' : SAVE_INTERVAL,
              'max_memory' : MEMORY_CAPACITY,
              'save_path' : save_model_path
              }

    # Q-Learning Agents
    dqn = DQN(**params)
    ddqn = DDQN(**params)
    dueldqn = DuelDQN(**params)

    # SARSA Agents
    sarsa = SARSA(**params)
    dsarsa = SARSA_DDQN(**params)
    duelsarsa = SARSA_DuelDQN(**params)


    # 
    """
    Training Agents:
    Set the hyperparams 
    uncomment one of the train functions and specify one of the agents
    Uncomment the logger.record_hyperparams to save the hyperparams to the log file
    """
    AGENT = dqn

    # make sure Agent is Q-Learning Agent
    train_QLearning(environment=othello, agent=AGENT, n_episodes=EPISODES, max_steps=MAX_STEPS, logger=logger)

    # make sure Agent is SARSA Agent
    # train_SARSA(save_path=save_model_path, agent=AGENT, n_episodes=EPISODES, max_steps=MAX_STEPS, logger=logger)

    # At the end of the log file after training save hyerparams for reference
    # logger.record_hyperparams(params)

    """
    Evaluate Agents:
        Comment out the above train function(s) and logger.record_hyperparams lines

        Put the full path from this scripts location to the saved models location
        Set the global hyperparams to what is at the end of the log files for the model you trained
        If you want to record video: Set the record_video param of the environment to True
    """

    # AGENT.load_model('./trained_models/03-08-2024/18_08_54/DQN_model_7')
    # test_agent(environment=othello, agent=AGENT)
