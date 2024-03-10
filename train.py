from agent import DeepAgent
from log import MetricLogger

from tqdm import trange

from othello import OBS_SPACE
from othelloPlayer import AgentPlayer
from othelloUtil import *

from environment import Environment


def train_QLearning(environment:Environment, agent:DeepAgent, n_episodes:int, max_steps:int, logger:MetricLogger) -> None:
    rewards = []
    loss_record = []
    q_record = []
    # Repeat (for each episode)
    for e in trange(n_episodes):
        # Initalize S
        environment.reset()
        state = environment.getState()
        cumulative_reward = 0
        for t in trange(max_steps):
            # Choose A from S using policy
            available_moves = environment.getAvailableMoves()
            action = agent.get_action(state, available_moves)

            # Take action A, observe R, S'
            terminate = environment.step(action)
            next_state = environment.getState()
            reward = environment.getReward()

            available_moves = environment.getAvailableMoves()
            next_action = agent.get_action(next_state, available_moves)

            # Update Q-Vals
            # Q(S,A) <- Q(S,A) + alpha[R + gamma * max_a Q(S',a) - Q(S,A)]
            q, loss = agent.train(state, action, reward, next_state, next_action, terminate)

            logger.log_step(reward, loss, q)

            # S <- S'
            state = environment.getState()
            cumulative_reward += reward

            if terminate:
                break
        rewards.append(cumulative_reward)
        loss_record.append(loss)
        q_record.append(q)
        logger.log_episode()

        logger.record(episode=e, epsilon=agent.epsilon, step=t)

    print(f"Rewards: {rewards}")
    print(f"Loss: {loss_record}")
    print(f"Q_record: {q_record}")


def train_SARSA(environment, agent:DeepAgent, n_episodes:int, max_steps:int, logger:MetricLogger) -> None:
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
    
    
def train_QLearning_pygame(environment:Environment, agent:DeepAgent, n_episodes:int=1, max_steps:int=100) -> None:
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
    
    rewards = [[],[]]
    loss_record = []
    q_record = []
    for e in trange(n_episodes):
        #start episode
        environment.resetBoard()
        episode_over = False
        cumulative_reward = [0,0]
        step = 0
        state = None
        action = None
        reward = None
        while not episode_over:
            step += 1
            prev_state = state
            prev_action = action
            prev_reward = reward
            state = environment.board
            if step%2 == 0:
                state = -state

            availableMoves = environment.getAvailableMoves()
            action = agent.get_action(state,availableMoves)
            next_state, reward = environment.step(action)
            if step%2 == 1:
                next_state = -next_state
            if prev_action != None:
                prev_action_index = getIndexFromCoords(prev_action)

            environment.activePlayer = environment.flipTurn(environment.activePlayer)
            terminated = environment.checkGameOver()
            
            truncated = False
            if terminated or truncated or step >= max_steps:
                episode_over = True

            #We calculate reward for the passive player, to account for reward
            #from each players turn to be associated with their action.
            #player 1 on step 1/odd steps, player 2 on step 2/even steps.
            #Put in last turn's state/action with the reward over previous 2 turns
            #Because agent should receive reward from previous action to current action.
            if step > 1:
                passive_player = (step + 1) % 2
                full_turn_reward = reward[passive_player] + prev_reward[passive_player]
                availableMoves = environment.getAvailableMoves()
                next_action = agent.get_action(next_state,availableMoves)
                next_action_index = getIndexFromCoords(next_action)
                q, loss = agent.update(prev_state.reshape(1,-1),prev_action_index, full_turn_reward, next_state.reshape(1,-1), episode_over,next_action_index)
                logger.log_step(reward, loss, q)
                
            cumulative_reward += reward
                

        rewards[0].append(cumulative_reward[0])
        rewards[1].append(cumulative_reward[1])
        loss_record.append(loss)
        q_record.append(q)
        logger.log_episode()

        # if (e % 1 == 0 or e == EPISODES - 1):
        logger.record(episode=e, epsilon=agent.epsilon, step=agent.step)

    print(f"Rewards p1: {rewards[0]}")
    print(f"Rewards p2: {rewards[1]}")
    print(f"Loss: {loss_record}")
    print(f"Q_record: {q_record}")
    