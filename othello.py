import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, FrameStack, RecordVideo

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import Agents

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


class Othello():
    """
    The environment for Othello that wraps the Gymnasium Atari Environment
    """
    def __init__(self, render_mode:RENDER_MODE, observation_type:OBS_SPACE, record_video:bool):
        self.render_mode = render_mode
        self.obs_type = observation_type
        self.record_video = record_video 
        self.env = self.setup_env()

    def setup_env(self) -> gym.Env:
        """
        Construct and return the Gymnasium Atari Environment for Othello
        """
        if self.record_video:
            self.render_mode = RENDER_MODE.RGB
        env = gym.make("ALE/Othello-v5", render_mode=self.render_mode.value, obs_type=self.obs_type.value)
        # Resize obs_space from 210 x 160 -> 105 x 80 to conserve data space but same resolution
        env = ResizeObservation(env, (105, 80))
        if self.record_video:
            env = RecordVideo(env, video_folder="./recordings", name_prefix="test-video",\
                              episode_trigger=lambda x: x % 2 == 0)
        return env

    def run(self) -> None:
        """
        Run Othello Environment for Testing
        """
        observation, _ = self.env.reset()
        obs = self.preprocess_obs(observation)
        if self.record_video:
            self.env.start_video_recorder()

        for _ in range(1000):
            action = self.env.action_space.sample() 
            observation, reward, terminated, truncated, _ = self.env.step(action)
            # obs = self.preprocess_obs(observation)
            self.env.render()

            if terminated or truncated:
                observation, _ = self.env.reset()

        if self.record_video:
            self.env.close_video_recorder()

        self.env.close()

    def run_DQN(self, episodes=1, max_turns=10000) -> None:
        # Get initial observation to preprocess and setup Agent
        observation, _ = self.env.reset()
        obs = self.preprocess_obs(observation)
        DQN_Agent = Agents.DQN(self.env, obs.shape, num_actions=self.env.action_space.n, epsilon=0.5)

        rewards = []
        for _ in range(episodes):
            observation, _ = self.env.reset()
            state = self.preprocess_obs(observation)
            r = 0
            for _ in range(max_turns):
                # print(f"t: {t}")
                action = DQN_Agent.action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.preprocess_obs(observation)
                state = next_state
                r += reward
                if terminated or truncated:
                    break
            rewards.append(r)
        print(f"Len rewards: {len(rewards)}")
        print(f"Rewards: {np.mean(rewards)}")

    def preprocess_obs(self, obs:np.ndarray) -> np.ndarray:
        """
        Crop the observation image to only look at the board.
        This should come in as [105, 80, 3] for RGB or [105, 80] for GRAY
        Rotate color channel to first (or add if GRAY), and add another in first for batch size
        Output size should be [batch_size, n_channel, height, width] = [1, 1 (or 3), 90, 66]
        Uncomment the imshow to see the images before the axes are changed
        """
        # OBS_TYPE = RGB - Don't think we need this. We should train on GRAY scale images anyways
        if obs.ndim > 2:
            # Crop
            obs_cropped = obs[8:-7, 6:-8, :]
            # Move color channel in front
            obs_cropped = np.moveaxis(obs_cropped, -1, 0)
        # OBS_TYPE = GRAY
        else:
            # Crop 
            obs_cropped = obs[8:-7, 6:-8]
            # Add a dimension in front for color channel
            obs_cropped = np.expand_dims(obs_cropped, axis=0)
        # Add another dimension in front for batch size
        return np.expand_dims(obs_cropped, axis=0)