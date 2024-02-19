import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, FrameStack, RecordVideo

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import Agents


class Othello():
    """
    The Gymnasium Atari Game Othello Environment
    """
    def __init__(self, render_mode, observation_type, record_video):
        self.RENDER_MODE = render_mode
        self.OBS_TYPE = observation_type
        self.RECORD_VIDEO = record_video 
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
        if self.RECORD_VIDEO:
            env = RecordVideo(env=env, video_folder="./recordings", name_prefix="test-video", episode_trigger=lambda x: x % 2 == 0)
        return env


    def run(self):
        """
        Run Othello Environment for Testing
        """
        observation, info = self.ENV.reset()
        obs = self.preprocess_obs(observation)
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

    def run_DQN(self, episodes=1, T=10000):
        # Get initial observation to preprocess and setup Agent
        observation, info = self.ENV.reset()
        obs = self.preprocess_obs(observation)
        DQN_Agent = Agents.DQN(env=self.ENV, state_shape=obs.shape, num_actions=self.ENV.action_space.n, epsilon=0.5)

        rewards = []
        for e in range(episodes):
            observation, info = self.ENV.reset()
            state = self.preprocess_obs(observation)
            r = 0
            for t in range(T):
                # print(f"t: {t}")
                action = DQN_Agent.action(state)
                observation, reward, terminated, truncated, info = self.ENV.step(action)
                next_state = self.preprocess_obs(observation)
                state = next_state
                r += reward
                if terminated or truncated:
                    break
            rewards.append(r)
        print(f"Len rewards: {len(rewards)}")
        print(f"Rewards: {np.mean(rewards)}")



    def preprocess_obs(self, obs):
        """
        Crop the observation image to only look at the board. 
        Switch Channel from last index to first index for PyTorch: [c, h, w]
        Uncomment the imshow to see the images before the axes are changed
        """
        # OBS_TYPE = RGB - Don't think we need this. We should train on GRAY scale images anyways
        if len(obs.shape) > 2: 
            obs_cropped = obs[8:-7, 6:-8, :]
            # plt.imshow(obs_cropped)
            # plt.show()
            obs_cropped = np.moveaxis(obs_cropped, -1, 0)
        # OBS_TYPE = GRAY
        else: 
            obs_cropped = obs[8:-7, 6:-8]
            # plt.imshow(obs_cropped)
            # plt.show()
            obs_cropped = [obs_cropped]
        obs = np.array(obs_cropped)
        # print(obs.shape) # channel (1 if GRAY and 3 if RGB) x 90 x 66 
        return obs

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