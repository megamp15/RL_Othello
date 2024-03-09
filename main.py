from othello import Othello, RENDER_MODE, OBS_SPACE

from othello2 import Othello2

from othelloGame import Othello as OthelloGame
from othelloPlayer import AgentPlayer
from neuralNet import StateNeuralNet
from othelloUtil import MoveMode

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
EPSILON_DECAY_RATE = 0.99999975
EPSILON_MIN = 0.1
ALPHA = 0.01 #0.00025
GAMMA = 0.9
SKIP_TRAINING = 20_000 
SAVE_INTERVAL = 50_000
SYNC_INTERVAL = 10_000

# TRAINING PARAMS
EPISODES = 1_000
MAX_STEPS = 8_000

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
    params = {
              #'state_shape' : environment.state_space,
              'state_shape' : (1,1,8,8),
              'num_actions' : 64,
              'epsilon' : EPSILON,
              #'epsilon_decay_rate' : EPSILON_DECAY_RATE,
              'epsilon_decay_rate' : 0.9,
              'epsilon_min' : EPSILON_MIN,
              'alpha' : ALPHA,
              'gamma' : GAMMA,
              'sync_interval' : SYNC_INTERVAL,
              'skip_training' : SKIP_TRAINING,
              'save_interval' : SAVE_INTERVAL,
              'max_memory' : MEMORY_CAPACITY,
              'save_path' : save_model_path
              }

    """
    RENDER_MODE = ["human" or "rgb_array"] Human mode makes you see the board where as RGB will just do it in the background
    OBSERVATION_TYPE = ["RGB", "GRAY", "RAM"] # RGB for color image, and GRAY for grayscale. RAM needs a diff env so that will cause an error for rn
    VIDEO = Bool  Note: Only works with render_mode: rgb_array. render_mode is hardcoded alread if this is True
    """
    #othello = Othello(RENDER_MODE.RGB, OBS_SPACE.RGB, False)
    #othello2 = Othello2(RENDER_MODE.RGB, OBS_SPACE.RGB, False)
    
    mode = MoveMode.FullBoardSelect
    p1_dqn = DQN(net_type=StateNeuralNet,**params)
    p2_dqn = DQN(net_type=StateNeuralNet,**params)
    player1 = AgentPlayer(mode,agent=p1_dqn)
    player2 = AgentPlayer(mode,agent=p2_dqn)
    #player1.setAgent(p1_dqn)
    #player2.setAgent(p2_dqn)

    
    game = OthelloGame(player1,player2,(8,8))
    environment = game
    

    '''
    # Q-Learning Agents
    dqn = DQN(**params)
    ddqn = DDQN(**params)
    dueldqn = DuelDQN(**params)

    # SARSA Agents
    sarsa = SARSA(**params)
    dsarsa = SARSA_DDQN(**params)
    duelsarsa = SARSA_DuelDQN(**params)
    '''
    #game.startGame()
    game.train_agent(agent=p1_dqn)

    #Agent = ddqn

    """
    Training Agents:
    Set the hyperparams 
    uncomment one of the train functions and specify one of the agents
    Uncomment the logger.record_hyperparams to save the hyperparams to the log file
    Uncomment Agent.load_model if you want to continue training from a certain saved model. 
        The model will load the networks weights and the epsilon it left off at. Modify the epsilon if needed
    """
    AGENT = dqn

    # AGENT.load_model('./trained_models/03-08-2024/19_15_54/DQN_model_2') 
    # AGENT.epsilon = 1 # Changing the epsilon

    # make sure Agent is Q-Learning Agent
    train_QLearning(environment=othello, agent=AGENT, n_episodes=EPISODES, max_steps=MAX_STEPS, logger=logger)

    # make sure Agent is SARSA Agent
    # train_SARSA(save_path=save_model_path, agent=AGENT, n_episodes=EPISODES, max_steps=MAX_STEPS, logger=logger)

    # At the end of the log file after training save hyerparams for reference
    params['episodes'] = EPISODES
    params['max_steps'] = MAX_STEPS
    logger.record_hyperparams(params)

    """
    Evaluate Agents:
        Comment out the above train function(s) and logger.record_hyperparams lines

        Put the full path from this scripts location to the saved models location
        Set the global hyperparams to what is at the end of the log files for the model you trained
        If you want to record video: Set the record_video param of the environment to True
    """

    # AGENT.load_model('./trained_models/03-08-2024/18_08_54/DQN_model_7')
    # test_agent(environment=othello, agent=AGENT)
