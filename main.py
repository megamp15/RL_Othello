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
from neuralNet import StateNeuralNet

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
EPSILON_DECAY_RATE = 0.9999975
EPSILON_MIN = 0.1
ALPHA = 0.00025
GAMMA = 0.9
SKIP_TRAINING = 100
SAVE_INTERVAL = 10_000
SYNC_INTERVAL = 500

# TRAINING PARAMS
EPISODES = 3_000
MAX_STEPS = 60

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
    for env in [setup_pygame_env()]:
        params = {
            'net_type' : StateNeuralNet,
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
            'net_type' : StateNeuralNet,
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
        dummy_agents = setup_agents(**dummy_params)
        train_QLearning(env, agents[0], dummy_agents[0], EPISODES, MAX_STEPS, logger)
        # for agent in agents:
        #     train_QLearning(env, agent, EPISODES, MAX_STEPS, logger)
            # test_agent(env, agent)
    