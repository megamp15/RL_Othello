from othello import Othello, RENDER_MODE, OBS_SPACE
from dqn import DDQN,DQN,DuelDQN
from sarsa import SARSA, SARSA_DDQN, SARSA_DuelDQN
# TODO: Might want to setup command line arguments

# EPSILON = .75
# EPSILON_DECAY_RATE = 0.99
# EPSILON_MIN = 0.01
# ALPHA = 0.01
# GAMMA = 0.9
# SKIP_TRAINING = 1_000 # This is memory size. Prime the memory with some inital experiences.
# SAVE_INTERVAL = 500
# SYNC_INTERVAL = 250

# AGENT PARAMS
EPSILON = 1
EPSILON_DECAY_RATE = 0.99999975
EPSILON_MIN = 0.1
ALPHA = 0.01 #0.00025
GAMMA = 0.9
SKIP_TRAINING = 1e4 # This is memory size. Prime the memory with some inital experiences.
SAVE_INTERVAL = 1e1
SYNC_INTERVAL = 1e3

# TRAINING PARAMS
EPISODES = 15 # Low to test for now
MAX_STEPS = 10_000


if __name__ == '__main__':
    """
    RENDER_MODE = ["human" or "rgb_array"] Human mode makes you see the board where as RGB will just do it in the background
    OBSERVATION_TYPE = ["RGB", "GRAY", "RAM"] # RGB for color image, and GRAY for grayscale. RAM needs a diff env so that will cause an error for rn
    VIDEO = Bool  Note: Only works with render_mode: rgb_array. render_mode is hardcoded alread if this is True
    """
    othello = Othello(RENDER_MODE.RGB, OBS_SPACE.RGB, False)

    dqn = DQN(state_shape=othello.state_space, num_actions=othello.num_actions, epsilon=EPSILON, epsilon_decay_rate=EPSILON_DECAY_RATE,
              epsilon_min=EPSILON_MIN, alpha=ALPHA, gamma=GAMMA, sync_interval=SYNC_INTERVAL, skip_training=SKIP_TRAINING,
              save_interval=SAVE_INTERVAL, max_memory=10_000)

    othello.train_agent(agent=dqn, n_episodes=EPISODES, max_steps=MAX_STEPS)