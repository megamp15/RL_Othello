from agent import DeepAgent

def test_agent(environment,  agent:DeepAgent):
    # Make epislon 0 so it always chooses actions learned by the agent
    agent.epsilon = 0
        
    if environment.record_video:
        environment.env.start_video_recorder()

    raw_obs, _ = environment.env.reset()
    state = environment.preprocess_obs(raw_obs)

    cumulative_reward = 0
    while True:
        action = agent.get_action(state)

        raw_obs, reward, terminated, truncated, _ = environment.env.step(action)
        next_state = environment.preprocess_obs(raw_obs)

        state = next_state
        cumulative_reward += reward

        if terminated or truncated:
            break

    print(f"Reward: {cumulative_reward}")

    if environment.record_video:
        environment.env.close_video_recorder()

    environment.env.close()

