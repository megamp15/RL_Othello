from othello import Othello, RENDER_MODE, OBS_SPACE

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

date = time.strftime('%m-%d-%Y')

# Model saving path
save_model_path = Path("trained_models") / date
save_model_path.mkdir(parents=True, exist_ok=True)


saveDir_recordings = Path("recordings") / date
saveDir_recordings.mkdir(parents=True, exist_ok=True)


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

    mode = MoveMode.FullBoardSelect
    p1_dqn = DQN(net_type=StateNeuralNet,**params)
    p2_dqn = DQN(net_type=StateNeuralNet,**params)
    player1 = AgentPlayer(mode,agent=p1_dqn)
    player2 = AgentPlayer(mode,agent=p2_dqn)

    
    game = OthelloGame(player1,player2,(8,8))
    environment = game
    

    game.startGame()
    #game.train_agent(agent=p1_dqn)


    # Training Agents
    #train_QLearning(environment=othello, agent=Agent, n_episodes=EPISODES, max_steps=MAX_STEPS)
    # train_SARSA(save_path=save_model_path, agent=sarsa, n_episodes=EPISODES, max_steps=MAX_STEPS)

    # Evaluate Agents
    # Put the full path from this scripts location to the saved models location
    # We are not saving the hyperparams so might want to make a func for that but for now evaluate right after train
    # Agent.load()
    # test_agent(environment=othello, agent=Agent)
