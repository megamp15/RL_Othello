from enum import Enum

import othello
# TODO: Might want to setup command line arguments

if __name__ == '__main__':
    """
    RENDER_MODE = ["human" or "rgb_array"]
    OBSERVATION_TYPE = ["RGB", "GRAY", "RAM"] # RAM needs a diff env so that will cause an error
    FRAME_STACK = int : the number of frames to stack together when retrieving the observation from the env
    VIDEO = Bool  Note: Only works with render_mode: rgb_array
    """
    othello = othello.Othello(render_mode=othello.render_mode.HUMAN.value, observation_type=othello.obs_space.RGB.value, frame_stack=1, record_video=False)
    othello.run()