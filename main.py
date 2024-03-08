from othello import Othello, RENDER_MODE, OBS_SPACE

from othello2 import Othello2

from dqn import DDQN, DQN, DuelDQN
from sarsa import SARSA, SARSA_DDQN, SARSA_DuelDQN

from pathlib import Path
import time

# Model saving path
save_model_path = Path("trained_models") / time.strftime('%Y%m%d-%H%M%S')
save_model_path.mkdir(parents=True)

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
    othello = Othello(RENDER_MODE.RGB, OBS_SPACE.RGB, False)
    othello2 = Othello2(RENDER_MODE.RGB, OBS_SPACE.RGB, False)

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

    environment.train_agent(dqn, 1, 10)

    # Check the state of othello after n_steps
    # othello2.run(n_steps=1000)

    # othello.train_agent(agent=dqn, n_episodes=EPISODES, max_steps=MAX_STEPS)

    othello2.train_QLearning(agent=dqn, n_episodes=EPISODES, max_steps=MAX_STEPS)
    # othello2.train_SARSA(agent=sarsa, n_episodes=EPISODES, max_steps=MAX_STEPS)