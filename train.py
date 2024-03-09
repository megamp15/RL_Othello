from agent import DeepAgent
from log import MetricLogger

from pathlib import Path
import time
from tqdm import trange

from othello import OBS_SPACE
from othelloPlayer import AgentPlayer
from othelloUtil import *

# Logs saving path
save_logs_path = Path("logs") / time.strftime('%m-%d-%Y')
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
    
    
def train_QLearning_pygame(environment, agent:DeepAgent, n_episodes:int=1, max_steps:int=100) -> None:
    '''
    Train function for the othelloGame environment

    Parameters
    ----------
    environment : othelloGame
        DESCRIPTION.
    agent : DeepAgent
        DESCRIPTION.
    n_episodes : int, optional
        DESCRIPTION. The default is 1.
    max_steps : int, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    None
    '''
    #Set both players to be the same agent
    player1 = AgentPlayer(MoveMode.FullBoardSelect,agent=agent)
    player2 = AgentPlayer(MoveMode.FullBoardSelect,agent=agent)
    environment.player1 = player1
    environment.player2 = player2
    
    rewards_p1 = []
    rewards_p2 = []
    loss_record = []
    q_record = []
    for e in trange(n_episodes):
        #start episode
        environment.resetBoard()
        episode_over = False
        cumulative_reward_p1 = 0
        cumulative_reward_p2 = 0
        step = 0
        while not episode_over:
            step += 1
            state = environment.board
            #print("Train state.shape start",state.shape)

            availableMoves = environment.getAvailableMoves()
            action = agent.get_action(state,availableMoves)
            #print('trainQLearningPygame action',action)
            #print('step,',step,'legal^',availableMoves)
            next_state, reward = environment.step(action)
            action_index = getIndexFromCoords(action)

            environment.activePlayer = environment.flipTurn(environment.activePlayer)
            terminated = environment.checkGameOver()
            
            truncated = False
            if terminated or truncated or step >= max_steps:
                episode_over = True

            #TODO work this into a better version which accurately handles a pair of turns
            #as a single reward with the temp rewards added together, and updates
            #the agent about to go.
            #print("Train state.shape into buffer",state.shape)
            q, loss, a_exit = agent.update(state.reshape(1,-1),action_index, reward, next_state.reshape(1,-1), episode_over)

            logger.log_step(reward, loss, q)

            episode_over |= a_exit

            cumulative_reward_p1 += reward
            cumulative_reward_p2 += reward
            
        #if step < 60:
        #    print("episode over: current step:",step)
        #    print("ending board:",environment.board)
        #    print("legal moves left:",environment.getAvailableMoves())
        rewards_p1.append(cumulative_reward_p1)
        rewards_p2.append(cumulative_reward_p2)
        loss_record.append(loss)
        q_record.append(q)
        logger.log_episode()

        # if (e % 1 == 0 or e == EPISODES - 1):
        logger.record(episode=e, epsilon=agent.epsilon, step=agent.step)

    print(f"Rewards p1: {rewards_p1}")
    print(f"Rewards p2: {rewards_p2}")
    print(f"Loss: {loss_record}")
    print(f"Q_record: {q_record}")
