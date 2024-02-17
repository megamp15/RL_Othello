import gymnasium as gym
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt


class obs_space(Enum):
    """
    The observation spaces available through Gymnasium
    """
    RGB = "rgb"
    GRAY = "grayscale"
    RAM = "ram"

class render_mode(Enum):
    HUMAN = "human"
    RGB = "rgb_array"

class actions(Enum):
    """
    The action space of Gymnasium's Othello environment
    """
    NOOP = 0
    PLACE = 1 # Fire in documentation
    UP = 2
    RIGHT = 3
    LEFT = 4
    DOWN = 5
    UPRIGHT = 6
    UPLEFT = 7
    DOWNRIGHT = 8
    DOWNLEFT = 9

class Othello():
    """
    The Gymnasium Atari Game Othello Environment
    """
    def __init__(self, render_mode, video):
        self.RENDER_MODE = render_mode
        self.VIDEO = video 
        self.env = self.setup_env()


    def setup_env(self):
        """
        Construct and return the Gymnasium Atari Environment for Othello
        """
        if self.VIDEO:
            tmp_env = gym.make("ALE/Othello-v5", render_mode=render_mode.RGB.value)

            # wrap the env in the record video
            env = gym.wrappers.RecordVideo(env=tmp_env, video_folder="./recordings", name_prefix="test-video", episode_trigger=lambda x: x % 2 == 0)
        else:
            env = gym.make("ALE/Othello-v5", render_mode=self.RENDER_MODE)
        return env


    def run(self):
        # env reset for a fresh start
        observation, info = self.env.reset()

        if self.VIDEO:
            self.env.start_video_recorder()


        for _ in range(1000):
            action = self.env.action_space.sample() 
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.env.render()

            # plt.imshow(observation)
            # plt.show()
            if terminated or truncated:
                observation, info = self.env.reset()

        if self.VIDEO:
            self.env.close_video_recorder()

        self.env.close()