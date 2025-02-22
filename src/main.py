import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make('ALE/Othello-v5')

print("Observation space:", env.observation_space)

print("action space: ", env.action_space)

env.close()
