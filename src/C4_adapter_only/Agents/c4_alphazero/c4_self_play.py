import torch
import numpy as np
from C4_adapter_only.Env.c4_env import ConnectFourEnv
from C4_adapter_only.Agents.c4_alphazero.c4_az_agent import ConnectFourAlphaZeroAgent

def self_play_game(agent: ConnectFourAlphaZeroAgent, env: ConnectFourEnv):
    states = []
    policies = []
    players = []
    rewards = []

    obs, _ = env.reset()
    done = False
    while not done:
        board = obs["board"]
        state_input = torch.tensor(board, dtype=torch.float32).unsqueeze(0).unsqueeze(0) 
        action, policy = agent.choose_action_and_policy(env, temperature=1.0)

        # Ensure policy is a valid NumPy array
        policy = np.array(policy, dtype=np.float32)

        if policy.shape[0] != env.action_space.n:  # Check for correct action space size
            print(f"Warning: Policy shape mismatch! Expected {env.action_space.n}, got {policy.shape[0]}")
            continue  # Skip invalid data
        
        states.append(state_input)
        policies.append(policy)
        players.append(obs["current_player"])
        
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated

    final_reward = rewards[-1] if rewards else 0.0
    training_examples = []
    for state, policy, player in zip(states, policies, players):
        value_target = final_reward if player == 0 else -final_reward
        training_examples.append((state, policy, value_target))
    
    return training_examples, final_reward
