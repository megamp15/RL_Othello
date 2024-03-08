from agent import DeepAgent
from log import MetricLogger

from pathlib import Path
import time
from tqdm import trange

from othello import OBS_SPACE

# Logs saving path
save_logs_path = Path("logs") / time.strftime('%m-%d-%Y')  / time.strftime('%H_%M_%S')
save_logs_path.mkdir(parents=True, exist_ok=True)
logger = MetricLogger(save_logs_path)

def train_QLearning(environment, agent:DeepAgent, n_episodes:int=1, max_steps:int=100) -> None:
    rewards = []
    loss_record = []
    q_record = []
    # Repeat (for each episode)
    for e in trange(n_episodes):
        # Initalize S
        raw_obs, _ = environment.env.reset()
        state = environment.preprocess_obs(raw_obs)
        cumulative_reward = 0
        for t in trange(max_steps):
            # Choose A from S using policy
            action = agent.get_action(state)

            # Take action A, observe R, S'
            raw_obs, reward, terminated, truncated, _ = environment.env.step(action)
            next_state = environment.preprocess_obs(raw_obs)

            terminate = 1 if terminated or truncated else 0
            
            # Remove batch size dimension
            if environment.obs_type == OBS_SPACE.GRAY:
                state_mem = state[0]
                next_state_mem = next_state[0]
            else:
                state_mem = state.squeeze()
                next_state_mem = next_state.squeeze()
            
                # Store step in memory 
            agent.memory.cache(state_mem, action, reward, next_state_mem, terminate)

            # Update Q-Vals
            # Q(S,A) <- Q(S,A) + alpha[R + gamma * max_a Q(S',a) - Q(S,A)]
            q, loss = agent.train()

            logger.log_step(reward, loss, q)

            # S <- S'
            state = next_state
            cumulative_reward += reward

            if terminated or truncated:
                break
        rewards.append(cumulative_reward)
        loss_record.append(loss)
        q_record.append(q)
        logger.log_episode()

        logger.record(episode=e, epsilon=agent.epsilon, step=agent.step)

    print(f"Rewards: {rewards}")
    print(f"Loss: {loss_record}")
    print(f"Q_record: {q_record}")


def train_SARSA(environment, agent:DeepAgent, n_episodes:int=1, max_steps:int=100) -> None:
    rewards = []
    loss_record = []
    q_record = []
    # Repeat (for each episode)
    for e in trange(n_episodes):
        # Initalize S
        raw_obs, _ = environment.env.reset()
        state = environment.preprocess_obs(raw_obs)
        cumulative_reward = 0

        # Choose A from S using policy
        action = agent.get_action(state)

        # Repeat (for each step of episode)
        for t in trange(max_steps):
            # Take action A, observe R, S'
            raw_obs, reward, terminated, truncated, _ = environment.env.step(action)
            next_state = environment.preprocess_obs(raw_obs)

            terminate = 1 if terminated or truncated else 0

            # Choose A' from S' using policy
            next_action = agent.get_action(next_state)  # Get next action for SARSA
            
            # Remove batch size dimension
            if environment.obs_type == OBS_SPACE.GRAY:
                state_mem = state[0]
                next_state_mem = next_state[0]
            else:
                state_mem = state.squeeze()
                next_state_mem = next_state.squeeze()

            # Store step in memory 
            agent.memory.cache(state_mem, action, reward, next_state_mem, terminate, next_action)

            # Update Q-Vals
            # Q(S,A) <- Q(S,A) + alpha[R + gamma * Q(S',A') - Q(S,A)]
            q, loss = agent.train()

            logger.log_step(reward, loss, q)

            # S <- S', A<- A'
            state = next_state
            action = next_action

            cumulative_reward += reward
            if terminated or truncated:
                break
        rewards.append(cumulative_reward)
        loss_record.append(loss)
        q_record.append(q)
        logger.log_episode()

        logger.record(episode=e, epsilon=agent.epsilon, step=agent.step)

    print(f"Rewards: {rewards}")
    print(f"Loss: {loss_record}")
    print(f"Q_record: {q_record}")  
