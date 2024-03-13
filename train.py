from tqdm import trange

from agent import DeepAgent, AgentType
from log import MetricLogger
from environment import Environment
from othelloUtil import *

def train_QLearning(environment:Environment, agent:DeepAgent, dummy_agent:DeepAgent, n_episodes:int, max_steps:int,
                    logger:MetricLogger, train:bool=True) -> None:
    rewards = []
    loss_record = []
    q_record = []
    # Repeat (for each episode)
    for e in trange(n_episodes, unit='episode'):
        # Initalize S
        environment.reset()
        state = environment.getState()
        cumulative_reward = 0
        action = None
        
        if agent.agent_type == AgentType.SARSA:
            # Choose A from S using policy
            available_moves = environment.getAvailableMoves()
            action = agent.get_action(state, available_moves)

        for t in trange(max_steps, unit='turn', leave=False):
            if agent.agent_type == AgentType.Q_LEARNING:
                # Choose A from S using policy
                available_moves = environment.getAvailableMoves()
                action = agent.get_action(state, available_moves)

            # Take action A, observe R, S'
            terminate = environment.step(action, dummy_agent)
            if terminate:
                break

            if train:
                next_state = environment.getState()
                reward = environment.getReward()

                available_moves = environment.getAvailableMoves()
                next_action = agent.get_action(next_state, available_moves)

                # Update Q-Vals
                q, loss = agent.train(state, action, reward, next_state, next_action, terminate)

                logger.log_step(reward, loss, q)

            # S <- S'
            state = environment.getState()
            cumulative_reward += reward
        agent.reset()

        if train:
            rewards.append(cumulative_reward)
            loss_record.append(loss)
            q_record.append(q)
        logger.log_episode()

        logger.record(episode=e, epsilon=agent.epsilon, step=t)

    agent.save_model()
    if train:
        print(f"Rewards: {rewards}")
        print(f"Loss: {loss_record}")
        print(f"Q_record: {q_record}")


def train_SARSA(environment:Environment, agent:DeepAgent, dummy_agent:DeepAgent, n_episodes:int, max_steps:int, logger:MetricLogger) -> None:
    rewards = []
    loss_record = []
    q_record = []
    # Repeat (for each episode)
    for e in trange(n_episodes, unit='episode'):
        # Initalize S
        environment.reset()
        state = environment.getState()
        cumulative_reward = 0

        # Choose A from S using policy
        available_moves = environment.getAvailableMoves()
        action = agent.get_action(state,available_moves)

        # Repeat (for each step of episode)
        for t in trange(max_steps, unit='turn', leave=False):
            # Take action A, observe R, S'
            terminate = environment.step(action, dummy_agent)
            if terminate:
                break

            next_state = environment.getState()
            reward = environment.getReward()

            # Choose A' from S' using policy
            available_moves = environment.getAvailableMoves()
            next_action = agent.get_action(next_state,available_moves)

            # Update Q-Vals
            # Q(S,A) <- Q(S,A) + alpha[R + gamma * Q(S',A') - Q(S,A)]
            q, loss = agent.train(state, action, reward, next_state, next_action, terminate)

            logger.log_step(reward, loss, q)

            # S <- S', A<- A'
            state = environment.getState()
            action = next_action

            cumulative_reward += reward
        agent.reset()

        rewards.append(cumulative_reward)
        loss_record.append(loss)
        q_record.append(q)
        logger.log_episode()

        logger.record(episode=e, epsilon=agent.epsilon, step=agent.step)

    agent.save_model()
    print(f"Rewards: {rewards}")
    print(f"Loss: {loss_record}")
    print(f"Q_record: {q_record}")  
