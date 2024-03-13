from typing_extensions import Unpack

from othello import Othello, RENDER_MODE, OBS_SPACE

from othelloGame import Othello as OthelloGame
from othelloPlayer import AgentPlayer
from othelloUtil import MoveMode

from dqn import DDQN, DQN, DuelDQN
from sarsa import SARSA, SARSA_DDQN, SARSA_DuelDQN

from train import train_QLearning, train_SARSA
from test import test_agent
from environment import Environment
from agent import DeepAgent, AgentParams
from neuralNet import StateNeuralNet, PixelNeuralNet

import time
from log import MetricLogger

date = time.strftime('%m-%d-%Y')
t = time.strftime('%H_%M_%S')

# Model saving path
save_model_path = f'trained_models/{date}/{t}'

# Logs saving path
save_logs_path = f'logs/{date}/{t}'
logger = MetricLogger(save_logs_path, 10)

save_recordings_path = f'recordings/{date}/{t}'

# AGENT PARAMS
EPSILON = 1
EPSILON_DECAY_RATE = 0.999975
EPSILON_MIN = 0.1
ALPHA = 0.00025
GAMMA = 0.9
SKIP_TRAINING = 1
SAVE_INTERVAL = 10_000
SYNC_INTERVAL = 500

# TRAINING PARAMS
EPISODES = 3_000
MAX_STEPS_PYGAME = 30
MAX_STEPS_GYM = 1000

MEMORY_CAPACITY = 100_000
BATCH_SIZE = 64

def setup_gym_env() -> Environment:
    othello = Othello(render_mode=RENDER_MODE.RGB, observation_type=OBS_SPACE.GRAY, record_video=False,
                      save_recordings_path=save_recordings_path)
    return othello

def setup_pygame_env() -> Environment:
    player = AgentPlayer(MoveMode.FullBoardSelect,agent=None)
    game = OthelloGame(player,player,(8,8))
    return game

def setup_agents(**kwargs:Unpack[AgentParams]) -> list[DeepAgent]:
    dqn = DQN(**kwargs)
    ddqn = DDQN(**kwargs)
    dueldqn = DuelDQN(**kwargs)

    sarsa = SARSA(**kwargs)
    dsarsa = SARSA_DDQN(**kwargs)
    duelsarsa = SARSA_DuelDQN(**kwargs)
    return [dqn,ddqn,dueldqn,sarsa,dsarsa,duelsarsa]


if __name__ == '__main__':
    for env in [setup_gym_env(),setup_pygame_env()]:
        params = {
            'net_type' : StateNeuralNet if isinstance(env,OthelloGame) else PixelNeuralNet,
            'state_shape' : env.state_space,
            'num_actions' : env.num_actions,
            'epsilon' : EPSILON,
            'epsilon_decay_rate' : EPSILON_DECAY_RATE,
            'epsilon_min' : EPSILON_MIN,
            'alpha' : ALPHA,
            'gamma' : GAMMA,
            'sync_interval' : SYNC_INTERVAL,
            'skip_training' : SKIP_TRAINING,
            'save_interval' : SAVE_INTERVAL,
            'max_memory' : MEMORY_CAPACITY,
            'save_path' : save_model_path,
            'batch_size' : BATCH_SIZE
            }
        dummy_params = {
            'net_type' : StateNeuralNet if isinstance(env,OthelloGame) else PixelNeuralNet,
            'state_shape' : env.state_space,
            'num_actions' : env.num_actions,
            'epsilon' : 1,
            'epsilon_decay_rate' : 1,
            'epsilon_min' : 1,
            'alpha' : ALPHA,
            'gamma' : GAMMA,
            'sync_interval' : int(1e24),
            'skip_training' : int(1e24),
            'save_interval' : int(1e24),
            'max_memory' : MEMORY_CAPACITY,
            'save_path' : save_model_path,
            'batch_size' : BATCH_SIZE
            }
        agents = setup_agents(**params)
        dummy_agent = DQN(**dummy_params)
        max_steps = MAX_STEPS_PYGAME if isinstance(env,OthelloGame) else MAX_STEPS_GYM
        print(f'Environment : {env.name}')
        for agent in agents:
            print(f'Agent: {agent.name}')
            logger.save_dir = f'{save_logs_path}_{agent.name}'
            train_QLearning(env, agent, dummy_agent, 1, max_steps, logger)
            # test_agent(env, agent)
    