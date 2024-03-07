from othello import Othello, RENDER_MODE, OBS_SPACE
from dqn import DDQN,DQN,DuelDQN
from sarsa import SARSA, SARSA_DDQN, SARSA_DuelDQN
# TODO: Might want to setup command line arguments

EPSILON = .75
EPSILON_DECAY_RATE = 0.99
EPSILON_MIN = 0.01
ALPHA = 0.01
GAMMA = 0.9
SKIP_TRAINING = 1_000
SAVE_INTERVAL = 500
SYNC_INTERVAL = 250

if __name__ == '__main__':
    """
    RENDER_MODE = ["human" or "rgb_array"] Human mode makes you see the board where as RGB will just do it in the background
    OBSERVATION_TYPE = ["RGB", "GRAY", "RAM"] # RGB for color image, and GRAY for grayscale. RAM needs a diff env so that will cause an error for rn
    VIDEO = Bool  Note: Only works with render_mode: rgb_array. render_mode is hardcoded alread if this is True
    """
    othello = Othello(RENDER_MODE.RGB, OBS_SPACE.RGB, False)

    # Uncomment the following to run just the base othello
    dqn = DQN(state_shape=othello.state_space, num_actions=othello.num_actions, epsilon=EPSILON, epsilon_decay_rate=EPSILON_DECAY_RATE,
              epsilon_min=EPSILON_MIN, alpha=ALPHA, gamma=GAMMA, sync_interval=SYNC_INTERVAL, skip_training=SKIP_TRAINING,
              save_interval=SAVE_INTERVAL, max_memory=10_000)

    othello.train_agent(dqn, 10, 1000)
    # othello.run()

    # To run a test of the DQN algorithm based on the evaluate method: https://www.kaggle.com/code/pedrobarrios/proyecto2-yandexdataschool-week4-rlataribreakout
    # othello.run_DQN()

    # othello.evaluate_DQN_Agent()