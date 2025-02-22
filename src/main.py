from environment import Environment

env = Environment(env_name="ALE/Othello-v5", render_mode="human")

obs_space, act_space = env.get_space()
print(f"Observation space: {obs_space}")
print(f"Action space: {act_space}")

env.close()
