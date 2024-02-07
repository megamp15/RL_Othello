import gymnasium as gym

# GLOBAL VARS
RENDER_MODE = "human" # or rgb_array
VIDEO = True # Only works with render_mode: rgb_array

if VIDEO:
    tmp_env = gym.make("ALE/Othello-v5", render_mode="rgb_array")

    # wrap the env in the record video
    env = gym.wrappers.RecordVideo(env=tmp_env, video_folder="./recordings", name_prefix="test-video", episode_trigger=lambda x: x % 2 == 0)
else:
    env = gym.make("ALE/Othello-v5", render_mode=RENDER_MODE)

# env reset for a fresh start
observation, info = env.reset()

if VIDEO:
    env.start_video_recorder()


for _ in range(1000):
    action = env.action_space.sample() 
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        observation, info = env.reset()

if VIDEO:
    env.close_video_recorder()

env.close()
