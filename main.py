from enum import Enum

import othello
# TODO: Might want to setup command line arguments

if __name__ == '__main__':
    """
    RENDER_MODE = ["human" or "rgb_array"]
    VIDEO = Bool  Note: Only works with render_mode: rgb_array
    """
    othello = othello.Othello(render_mode=othello.render_mode.HUMAN.value, video=False)
    othello.run()