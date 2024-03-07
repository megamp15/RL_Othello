import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, FrameStack, RecordVideo

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from agent import DeepAgent
from dqn import DDQN,DQN,DuelDQN
from sarsa import SARSA, SARSA_DDQN, SARSA_DuelDQN

import time
from pathlib import Path
from log import MetricLogger


saveDir_logs = Path("logs") / time.strftime('%Y%m%d-%H%M%S')
saveDir_logs.mkdir(parents=True, exist_ok=True)
logger = MetricLogger(saveDir_logs)

saveDir_recordings = Path("recordings", exist_ok=True) /  time.strftime('%Y%m%d-%H%M%S')
# saveDir_recordings.mkdir(parents=True)

saveDir_models = Path("models") / time.strftime('%Y%m%d-%H%M%S')
saveDir_recordings.mkdir(parents=True, exist_ok=True)


# AGENT PARAMS
EPSILON = 1
EPSILON_DECAY_RATE = 0.99999975
EPSILON_MIN = 0.1
ALPHA = 0.01 #0.00025
GAMMA = 0.9
SKIP_TRAINING = 1e5 # The amount of experience to gather before starting to train
SAVE_INTERVAL = 2e5
SYNC_INTERVAL = 1e4

# TRAINING PARAMS
EPISODES = 15 # Low to test for now
MAX_STEPS = 10_000


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

        obs, _ = self.env.reset()
        obs = self.preprocess_obs(obs)
        self.state_space = obs.shape
        self.num_actions = self.env.action_space.n

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
            env = RecordVideo(env, video_folder=saveDir_recordings, name_prefix="video", episode_trigger=lambda x: x)
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
    
    def train_agent(self, agent:DeepAgent, n_episodes:int=1, max_steps:int=100) -> None:
        rewards = []
        loss_record = []
        q_record = []
        for e in trange(n_episodes):
            raw_obs, _ = self.env.reset()
            exit = False
            cumulative_reward = 0
            step = 0
            while not exit:
                step += 1
                state = self.preprocess_obs(raw_obs)

                action = agent.get_action(state)
                raw_obs, reward, terminated, truncated, _ = self.env.step(action)

                if terminated or truncated or step >= max_steps:
                    exit = True

                next_state = self.preprocess_obs(raw_obs)

                q, loss, a_exit = agent.update(state, action, reward, next_state, exit)

                logger.log_step(reward, loss, q)

                exit |= a_exit

                cumulative_reward += reward
            rewards.append(cumulative_reward)
            loss_record.append(loss)
            q_record.append(q)
            logger.log_episode()

            # if (e % 1 == 0 or e == EPISODES - 1):
            logger.record(episode=e, epsilon=agent.epsilon, step=agent.step)
            
        print(f"Rewards: {rewards}")
        print(f"Loss: {loss_record}")
        print(f"Q_record: {q_record}")

    def train_QLearning(self, episodes=EPISODES, max_steps=MAX_STEPS) -> None:
        # Get initial observation to preprocess and setup Agent
        observation, _ = self.env.reset()
        obs = self.preprocess_obs(observation)

        # Q-Learning Agents
        Agent = DQN(state_shape=obs.shape, num_actions=self.env.action_space.n, epsilon=EPSILON, epsilon_decay_rate=EPSILON_DECAY_RATE, epsilon_min=EPSILON_MIN, alpha=ALPHA, gamma=GAMMA, skip_training=SKIP_TRAINING, save_interval=SAVE_INTERVAL, sync_interval=SYNC_INTERVAL, max_memory=10_000)
        # Agent = DDQN(agent_type="DQN", env=self.env, state_shape=obs.shape, num_actions=self.env.action_space.n, epsilon=EPSILON, epsilon_decay_rate=EPSILON_DECAY_RATE, epsilon_min=EPSILON_MIN, alpha=ALPHA, gamma=GAMMA, skip_training=SKIP_TRAINING, save_interval=SAVE_INTERVAL, sync_interval=SYNC_INTERVAL)
        # Agent = DuelDQN(agent_type="DQN", env=self.env, state_shape=obs.shape, num_actions=self.env.action_space.n, epsilon=EPSILON, epsilon_decay_rate=EPSILON_DECAY_RATE, epsilon_min=EPSILON_MIN, alpha=ALPHA, gamma=GAMMA, skip_training=SKIP_TRAINING, save_interval=SAVE_INTERVAL, sync_interval=SYNC_INTERVAL)

        rewards = []
        # Repeat (for each episode)
        for e in trange(episodes):
            # Initalize S
            observation, _ = self.env.reset()
            state = self.preprocess_obs(observation)
            r = 0
            for t in trange(max_steps):
                # Choose A from S using policy
                action = Agent.get_action(state)

                # Take action A, observe R, S'
                observation, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.preprocess_obs(observation)

                if terminated or truncated: 
                    terminate = 1
                else: 
                    terminate  = 0
                
                if self.obs_type == OBS_SPACE.GRAY:
                    state_mem = state[0]
                    next_state_mem = next_state[0]
                else:
                    state_mem = state.squeeze()
                    next_state_mem = next_state.squeeze()
                
                 # Store step in memory 
                Agent.memory.cache(state_mem, action, reward, next_state_mem, terminate)

                # Update Q-Vals
                # Q(S,A) <- Q(S,A) + alpha[R + gamma * max_a Q(S',a) - Q(S,A)]
                q, loss = Agent.train()

                logger.log_step(reward, loss, q)

                # S <- S'
                state = next_state
                r += reward
                if terminated or truncated:
                    break
            rewards.append(r)
            logger.log_episode()

            if (e % 5 == 0 or e == EPISODES - 1):
                logger.record(episode=e, epsilon=Agent.epsilon, step=Agent.step)
            print(f"Memory Len: {len(Agent.memory)}")
        print(f"Rewards: {rewards}")


    def train_SARSA(self, episodes=EPISODES, max_steps=MAX_STEPS) -> None:
        # Get initial observation to preprocess and setup Agent
        observation, _ = self.env.reset()
        obs = self.preprocess_obs(observation)

        # SARSA Agents
        Agent = SARSA(agent_type="SARSA", env=self.env, state_shape=obs.shape, num_actions=self.env.action_space.n, epsilon=EPSILON, epsilon_decay_rate=EPSILON_DECAY_RATE, epsilon_min=EPSILON_MIN, alpha=ALPHA, gamma=GAMMA, skip_training=SKIP_TRAINING, save_interval=SAVE_INTERVAL, sync_interval=SYNC_INTERVAL)
        # Agent = SARSA_DDQN(agent_type="SARSA", env=self.env, state_shape=obs.shape, num_actions=self.env.action_space.n, epsilon=EPSILON, epsilon_decay_rate=EPSILON_DECAY_RATE, epsilon_min=EPSILON_MIN,alpha=ALPHA, gamma=GAMMA, skip_training=SKIP_TRAINING, save_interval=SAVE_INTERVAL, sync_interval=SYNC_INTERVAL)
        # Agent = SARSA_DuelDQN(agent_type="SARSA", env=self.env, state_shape=obs.shape, num_actions=self.env.action_space.n, epsilon=EPSILON, epsilon_decay_rate=EPSILON_DECAY_RATE, epsilon_min=EPSILON_MIN,alpha=ALPHA, gamma=GAMMA, skip_training=SKIP_TRAINING, save_interval=SAVE_INTERVAL, sync_interval=SYNC_INTERVAL)

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
            for t in trange(max_steps):
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
                
                if self.obs_type == OBS_SPACE.GRAY:
                    state_mem = state[0]
                    next_state_mem = next_state[0]
                else:
                    state_mem = state.squeeze()
                    next_state_mem = next_state.squeeze()

                # Store step in memory 
                Agent.memory.cache(state_mem, action, reward, next_state_mem, terminate, next_action)

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

        # Q-Learning Agents
        # Agent = DQN(agent_type="DQN", env=self.env, state_shape=state.shape, num_actions=self.env.action_space.n, epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA, skip_training=SKIP_TRAINING, save_interval=SAVE_INTERVAL, sync_interval=SYNC_INTERVAL)
        # Agent = DDQN(agent_type="DQN", env=self.env, state_shape=obs.shape, num_actions=self.env.action_space.n, epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA, skip_training=SKIP_TRAINING, save_interval=SAVE_INTERVAL, sync_interval=SYNC_INTERVAL)
        Agent = DuelDQN(agent_type="DQN", env=self.env, state_shape=observation.shape, num_actions=self.env.action_space.n, epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA, skip_training=SKIP_TRAINING, save_interval=SAVE_INTERVAL, sync_interval=SYNC_INTERVAL)

        Agent.load_model(modelPath)

        if self.record_video:
            self.env.start_video_recorder()

        total_reward = 0
        for t in trange(100000):
            action = Agent.get_action(state)
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

        # SARSA Agents
        Agent = SARSA(agent_type="SARSA", env=self.env, state_shape=state.shape, num_actions=self.env.action_space.n, epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA, skip_training=SKIP_TRAINING, save_interval=SAVE_INTERVAL, sync_interval=SYNC_INTERVAL)
        # Agent = SARSA_DDQN(agent_type="SARSA", env=self.env, state_shape=obs.shape, num_actions=self.env.action_space.n, epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA, skip_training=SKIP_TRAINING, save_interval=SAVE_INTERVAL, sync_interval=SYNC_INTERVAL)
        # Agent = SARSA_DuelDQN(agent_type="SARSA", env=self.env, state_shape=obs.shape, num_actions=self.env.action_space.n, epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA, skip_training=SKIP_TRAINING, save_interval=SAVE_INTERVAL, sync_interval=SYNC_INTERVAL)
        Agent.load_model(modelPath)

        if self.record_video:
            self.env.start_video_recorder()

        total_reward = 0
        for t in trange(100000):
            action = Agent.get_action(state)
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
            # plt.imshow(obs_cropped)
            # plt.show()
            # Move color channel in front
            obs_cropped = np.moveaxis(obs_cropped, -1, 0)
        # OBS_TYPE = GRAY
        else:
            # Crop 
            obs_cropped = obs[8:-7, 6:-8]
            # Add a dimension in front for gray color channel
            obs_cropped = np.expand_dims(obs_cropped, axis=0)
        # Add another dimension in front for batch size
        retval = np.expand_dims(obs_cropped, axis=0)
        return retval

if __name__ == '__main__':
    othello = Othello(RENDER_MODE.RGB, OBS_SPACE.GRAY, True)

    # othello.run()
    othello.train_QLearning()
    # othello.train_SARSA()
    
    # othello.evaluate_SARSA_Agent("./SARSA/20240305-201726_model_17")
