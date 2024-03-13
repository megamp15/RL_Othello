import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, FrameStack, RecordVideo

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from environment import Environment

from agent import DeepAgent

class OBS_SPACE(Enum):
    """
    The observation spaces available through Gymnasium
    """
    RGB = "rgb"
    GRAY = "grayscale"
    RAM = "ram"

class RENDER_MODE(Enum):
    HUMAN = "human"
    RGB = "rgb_array"

class OBS_SPACE(Enum):
    """
    The observation spaces available through Gymnasium
    """
    RGB = "rgb"
    GRAY = "grayscale"
    RAM = "ram"

class RENDER_MODE(Enum):
    HUMAN = "human"
    RGB = "rgb_array"

class Othello(Environment):
    """
    The environment for Othello that wraps the Gymnasium Atari Environment
    """
    name = 'Gymnasium Othello'

    def __init__(self, render_mode:RENDER_MODE, observation_type:OBS_SPACE, record_video:bool, save_recordings_path:str=None):
        self.render_mode = render_mode
        self.obs_type = observation_type
        self.record_video = record_video 
        self.save_recordings_path = save_recordings_path
        self.env = self.setup_env()

        obs, _ = self.env.reset()
        obs = self.preprocess_obs(obs)
        self.last_state = None
        self.current_state = obs
        self.state_space = obs.shape
        self.num_actions = self.env.action_space.n
    
    def getState(self) -> np.ndarray:
        return self.current_state
    
    def reset(self) -> None:
        self.last_state = self.current_state
        obs, _ = self.env.reset()
        self.current_state = self.preprocess_obs(obs)
    
    def step(self, action:int, dummy_agent:DeepAgent=None) -> bool:
        obs, reward, terminated, truncated, _ = self.env.step(action)
        self.last_state = self.current_state
        self.current_state = self.preprocess_obs(obs)
        self.reward = reward
        return terminated | truncated
    
    def getReward(self) -> float:
        return self.reward
    
    def getAvailableMoves(self) -> list[int]:
        return list(range(self.num_actions))

    def setup_env(self) -> gym.Env:
        """
        Construct and return the Gymnasium Atari Environment for Othello
        """
        if self.record_video:
            self.render_mode = RENDER_MODE.RGB
        env = gym.make("ALE/Othello-v5", render_mode=self.render_mode.value, obs_type=self.obs_type.value)
        # Resize obs_space from 210 x 160 -> 53 x 40 to conserve data space but same resolution
        env = ResizeObservation(env, (53, 40))
        if self.record_video:
            env = RecordVideo(env, video_folder=self.save_recordings_path, name_prefix="video", episode_trigger=lambda x: x)
        return env

    def run(self, n_steps:int) -> None:
        """
        Run Othello Environment for Testing and see the environment after n_steps
        """
        observation, _ = self.env.reset()
        obs = self.preprocess_obs(observation)
        if self.record_video:
            self.env.start_video_recorder()

        for _ in range(n_steps):
            action = self.env.action_space.sample() 
            observation, reward, terminated, truncated, _ = self.env.step(action)
            # obs = self.preprocess_obs(observation)
            self.env.render()

            if terminated or truncated:
                observation, _ = self.env.reset()

        if self.record_video:
            self.env.close_video_recorder()

        obs = self.preprocess_obs(observation, show_state=True)
        self.env.close()

        
    def preprocess_obs(self, obs:np.ndarray, show_state:bool=False) -> np.ndarray:
        """
        Crop the observation image to only look at the board.
        This should come in as [105, 80] for GRAY or [105, 80, 3] for RGB
        Add color channel to first (or rotate if RGB), and add another in first for batch size
        Output size should be [batch_size, n_channel, height, width] = [1, 1 (or 3), 90, 66]
        Uncomment the imshow to see the images before the axes are changed
        """
        # OBS_TYPE = RGB - Don't think we need this. We should train on GRAY scale images anyways
        if obs.ndim > 2:
            # Crop
            obs_cropped = obs[4:-4, 3:-4, :] # 53 x 40
            # obs_cropped = obs[8:-7, 6:-8, :] # 105 x 80
            if show_state:
                plt.imshow(obs_cropped)
                plt.show()
            # Move color channel in front
            obs_cropped = np.moveaxis(obs_cropped, -1, 0)
        # OBS_TYPE = GRAY
        else:
            # Crop 
            obs_cropped = obs[4:-4, 3:-4] # 53 x 40
            # obs_cropped = obs[8:-7, 6:-8] # 105 x 80
            if show_state:
                plt.imshow(obs_cropped)
                plt.show()
            # Add a dimension in front for gray color channel
            obs_cropped = np.expand_dims(obs_cropped, axis=0)
        # Add another dimension in front for batch size
        retval = np.expand_dims(obs_cropped, axis=0)

        
        if self.obs_type == OBS_SPACE.GRAY:
            retval = retval[0]
        else:
            retval = retval.squeeze()

        return retval
    
    def close(self) -> None:
        if self.record_video:
            self.env.close_video_recorder()
        self.env.close()
