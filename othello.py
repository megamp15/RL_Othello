import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, FrameStack, RecordVideo

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from dqn import DDQN,DQN,DuelDQN
from sarsa import SARSA, SARSA_DDQN, SARSA_DuelDQN

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

    def train_QLearning(self, episodes=5, max_turns=50_000) -> None:
        # Get initial observation to preprocess and setup Agent
        observation, _ = self.env.reset()
        obs = self.preprocess_obs(observation)

        # DQN Agents
        # Agent = DQN(env=self.env, state_shape=obs.shape, num_actions=self.env.action_space.n, epsilon=0.5, alpha=0.01, gamma=0.9, skip_training=10_000, save_interval=10_1000, sync_interval=10_000)
        # Agent = DDQN(env=self.env, state_shape=obs.shape, num_actions=self.env.action_space.n, epsilon=0.5, alpha=0.01, gamma=0.9, skip_training=10_000, save_interval=10_1000, sync_interval=10_000)
        Agent = DuelDQN(env=self.env, state_shape=obs.shape, num_actions=self.env.action_space.n, epsilon=0.5, alpha=0.01, gamma=0.9, skip_training=10_000, save_interval=10_1000, sync_interval=10_000)

        rewards = []
        loss_record = []
        q_record = []
        # Repeat (for each episode)
        for e in trange(episodes):
            # Initalize S
            observation, _ = self.env.reset()
            state = self.preprocess_obs(observation)
            r = 0
            for t in trange(max_turns):
                if len(Agent.memory) > 10**4:
                    # Cant store any more experience
                    break

                # Choose A from S using policy
                action = Agent.get_action(state)

                # Take action A, observe R, S'
                observation, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.preprocess_obs(observation)

                if terminated or truncated: 
                    terminate = 1
                else: 
                    terminate  = 0
                
                 # Store step in memory 
                Agent.memory.cache(state.squeeze(), action, reward, next_state.squeeze(), terminate)

                # Update Q-Vals
                # Q(S,A) <- Q(S,A) + alpha[R + gamma * max_a Q(S',a) - Q(S,A)]
                q_vals, loss = Agent.train()

                # S <- S'
                state = next_state
                r += reward
                if terminated or truncated:
                    break
            if len(Agent.memory) > 10**4:
                # Cant store any more experience
                break
            rewards.append(r)
            loss_record.append(loss)
            q_record.append(q_vals)
        # print(f"Len rewards: {len(rewards)}")
        print(f"Rewards: {rewards}")
        print(f"Loss: {loss_record}")
        print(f"Q_record: {q_record}")

    def train_SARSA(self, episodes=5, max_turns=50_000) -> None:
        # Get initial observation to preprocess and setup Agent
        observation, _ = self.env.reset()
        obs = self.preprocess_obs(observation)

        # SARSA Agents
        Agent = SARSA(env=self.env, state_shape=obs.shape, num_actions=self.env.action_space.n, epsilon=0.5, alpha=0.01, gamma=0.9, skip_training=10_000, save_interval=10_1000, sync_interval=10_000)
        # Agent = SARSA_DDQN(env=self.env, state_shape=obs.shape, num_actions=self.env.action_space.n, epsilon=0.5, alpha=0.01, gamma=0.9, skip_training=10_000, save_interval=10_1000, sync_interval=10_000)
        # Agent = SARSA_DuelDQN(env=self.env, state_shape=obs.shape, num_actions=self.env.action_space.n, epsilon=0.5, alpha=0.01, gamma=0.9, skip_training=10_000, save_interval=10_1000, sync_interval=10_000)

        rewards = []
        loss_record = []
        q_record = []
        # Repeat (for each episode)
        for e in trange(episodes):
            # Initalize S
            observation, _ = self.env.reset()
            state = self.preprocess_obs(observation)
            r = 0

            # Choose A from S using policy
            action = Agent.get_action(state)

            # Repeat (for each step of episode)
            for t in trange(max_turns):
                if len(Agent.memory) > 10**4:
                    # Cant store any more experience
                    break
                # Take action A, observe R, S'
                observation, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.preprocess_obs(observation)

                if terminated or truncated: 
                    terminate = 1
                else: 
                    terminate  = 0

                # Choose A' from S' using policy
                next_action = Agent.get_action(next_state)  # Get next action for SARSA

                # Store step in memory 
                Agent.memory.cache(state.squeeze(), action, reward, next_state.squeeze(), terminate, next_action)

                # Update Q-Vals
                # Q(S,A) <- Q(S,A) + alpha[R + gamma * Q(S',A') - Q(S,A)]
                q_vals, loss = Agent.train()

                # S <- S', A<- A'
                state = next_state
                action = next_action

                r += reward
                if terminated or truncated:
                    break
            if len(Agent.memory) > 10**4:
                # Cant store any more experience
                break
            rewards.append(r)
            loss_record.append(loss)
            q_record.append(q_vals)
        # print(f"Len rewards: {len(rewards)}")
        print(f"Rewards: {rewards}")
        print(f"Loss: {loss_record}")
        print(f"Q_record: {q_record}")        

    def evaluate_QLearning_Agent(self, modelPath:str):
        observation, _ = self.env.reset()
        state = self.preprocess_obs(observation)
        # Make epislon 0 so it always chooses actions learned by the agent
        Test_DDQN_Agent = DDQN(env=self.env, state_shape=state.shape, num_actions=self.env.action_space.n, epsilon=0, alpha=0.01, gamma=0.9, sync_interval=10000)
        Test_DDQN_Agent.load_model(modelPath)

        if self.record_video:
            self.env.start_video_recorder()

        total_reward = 0
        for t in trange(100000):
            action = Test_DDQN_Agent.get_action(state)
            # print(f"action: {action}")
            next_state, reward, terminated, truncated, info = self.env.step(action)
            next_state = self.preprocess_obs(next_state)

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break
        print(f"Reward: {total_reward}")

        if self.record_video:
            self.env.close_video_recorder()

        self.env.close()

    def evaluate_SARSA_Agent(self, modelPath:str):
        observation, _ = self.env.reset()
        state = self.preprocess_obs(observation)
        # Make epislon 0 so it always chooses actions learned by the agent
        Test_DDQN_Agent = DDQN(env=self.env, state_shape=state.shape, num_actions=self.env.action_space.n, epsilon=0, alpha=0.01, gamma=0.9, sync_interval=10000)
        Test_DDQN_Agent.load_model(modelPath)

        if self.record_video:
            self.env.start_video_recorder()

        total_reward = 0
        for t in trange(100000):
            action = Test_DDQN_Agent.get_action(state)
            # print(f"action: {action}")
            next_state, reward, terminated, truncated, info = self.env.step(action)
            next_state = self.preprocess_obs(next_state)

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break
        print(f"Reward: {total_reward}")

        if self.record_video:
            self.env.close_video_recorder()

        self.env.close()
        
    def preprocess_obs(self, obs:np.ndarray) -> np.ndarray:
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

if __name__ == '__main__':
    othello = Othello(RENDER_MODE.RGB, OBS_SPACE.RGB, True)
    othello.train_QLearning()
    # othello.train_SARSA()

