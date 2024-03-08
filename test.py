from agent import DeepAgent
from tqdm import trange


def test_agent(environment,  agent:DeepAgent):
    # Make epislon 0 so it always chooses actions learned by the agent
    agent.epsilon = 0
    
    if environment.record_video:
        environment.env.start_video_recorder()

    total_reward = 0
    for t in trange(100000):
        action = agent.get_action(state)

        next_state, reward, terminated, truncated, _ = environment.env.step(action)
        next_state = environment.preprocess_obs(next_state)

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break
    print(f"Reward: {total_reward}")

    if environment.record_video:
        environment.env.close_video_recorder()

    environment.env.close()

