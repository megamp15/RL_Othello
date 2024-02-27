from othello import Othello, RENDER_MODE, OBS_SPACE
# TODO: Might want to setup command line arguments

if __name__ == '__main__':
    """
    RENDER_MODE = ["human" or "rgb_array"] Human mode makes you see the board where as RGB will just do it in the background
    OBSERVATION_TYPE = ["RGB", "GRAY", "RAM"] # RGB for color image, and GRAY for grayscale. RAM needs a diff env so that will cause an error for rn
    VIDEO = Bool  Note: Only works with render_mode: rgb_array. render_mode is hardcoded alread if this is True
    """
    othello = Othello(RENDER_MODE.HUMAN, OBS_SPACE.RGB, False)

    # Uncomment the following to run just the base othello 
    # othello.run()

    # To run a test of the DQN algorithm based on the evaluate method: https://www.kaggle.com/code/pedrobarrios/proyecto2-yandexdataschool-week4-rlataribreakout
    # othello.run_DQN()

    othello.evaluate_DQN_Agent()