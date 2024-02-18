import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, FrameStack, RecordVideo

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import DQN


class Othello():
    """
    The Gymnasium Atari Game Othello Environment
    """
    def __init__(self, render_mode, observation_type, record_video, frame_stack):
        self.RENDER_MODE = render_mode
        self.OBS_TYPE = observation_type
        self.RECORD_VIDEO = record_video 
        self.FRAME_STACK = frame_stack
        self.ENV = self.setup_env()


    def setup_env(self):
        """
        Construct and return the Gymnasium Atari Environment for Othello
        """
        if self.RECORD_VIDEO:
            self.RENDER_MODE = render_mode.RGB.value
        env = gym.make("ALE/Othello-v5", render_mode=self.RENDER_MODE, obs_type=self.OBS_TYPE)
        # Resize obs_space from 210 x 160 -> 105 x 80 to conserve data space but same resolution
        env = ResizeObservation(env, (105, 80))
        env = FrameStack(env, self.FRAME_STACK) # Obs shape becomes (FRAME_STACK, OBS_SPACE[0], OBS_SPACE[1], OBS_SPACE[2])
        if self.RECORD_VIDEO:
            env = RecordVideo(env=env, video_folder="./recordings", name_prefix="test-video", episode_trigger=lambda x: x % 2 == 0)
        return env


    def run(self):
        observation, info = self.ENV.reset()
        # obs = self.preprocess_obs(observation)
        # DQN.DQN(obs.shape, 10, 0.5)
        if self.RECORD_VIDEO:
            self.ENV.start_video_recorder()

        for _ in range(1000):
            action = self.ENV.action_space.sample() 
            observation, reward, terminated, truncated, info = self.ENV.step(action)
            # obs = self.preprocess_obs(observation)
            self.ENV.render()

            if terminated or truncated:
                observation, info = self.ENV.reset()

        if self.RECORD_VIDEO:
            self.ENV.close_video_recorder()

        self.ENV.close()

    def preprocess_obs(self, observation):
        """
        Crop the observation image to only look at the board. 
        """
        # OBS_TYPE = RGB
        if len(observation.shape) > 2: 
            obs_cropped = observation[8:-7, 6:-8, :]
        # OBS_TYPE = GRAY
        else: 
            obs_cropped = observation[8:-7, 6:-8]
        # print(obs_cropped.shape) # 90 x 66 (x 3 if RGB)
        # plt.imshow(obs_cropped)
        # plt.show()
        return obs_cropped

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