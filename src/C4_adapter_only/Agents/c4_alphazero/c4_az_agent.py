import torch
import numpy as np
import copy
from C4_adapter_only.Env.c4_env import ConnectFourEnv
from C4_adapter_only.Agents.c4_alphazero.c4_az_mcts import ConnectFourMCTSNode

class ConnectFourAlphaZeroAgent:
    def __init__(self, network, num_simulations: int = 800, exploration_weight: float = 1.4):
        self.network = network
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.name = f"AlphaZero_{num_simulations}"
    
    def choose_action(self, env: ConnectFourEnv, temperature: float = 1.0):
        root = ConnectFourMCTSNode(copy.deepcopy(env))
        
        if root.is_terminal():
            valid_moves = [i for i, is_valid in enumerate(root.env._get_observation()["valid_moves"]) if is_valid == 1]
            return np.random.choice(valid_moves) if valid_moves else 0
        
        for _ in range(self.num_simulations):
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.exploration_weight)
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand(self.network)
            result = node.simulate(self.network)
            node.backpropagate(result)
        
        visit_counts = np.array([child.visits for child in root.children])
        actions = np.array([child.action for child in root.children])
        
        if temperature == 0:
            best_action = actions[np.argmax(visit_counts)]
            return int(best_action)
        
        visit_counts = visit_counts ** (1.0 / temperature)
        policy = visit_counts / np.sum(visit_counts) if np.sum(visit_counts) > 0 else np.ones(len(actions)) / len(actions)
        
        valid_indices = actions[actions < env.action_space.n]
        if len(valid_indices) == 0:
            print("Warning: No valid moves found. Defaulting to random action.")
            return np.random.choice(env.action_space.n)
        
        chosen_action = np.random.choice(valid_indices, p=policy[: len(valid_indices)])
        return int(chosen_action), policy
    
    def choose_action_and_policy(self, env: ConnectFourEnv, temperature: float = 1.0):
        root = ConnectFourMCTSNode(copy.deepcopy(env))
        
        if root.is_terminal():
            valid_moves = [i for i, is_valid in enumerate(root.env._get_observation()["valid_moves"]) if is_valid == 1]
            policy = np.zeros(env.action_space.n, dtype=np.float32)
            return (np.random.choice(valid_moves) if valid_moves else 0, policy)

        for _ in range(self.num_simulations):
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.exploration_weight)
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand(self.network)
            result = node.simulate(self.network)
            node.backpropagate(result)

        visit_counts = np.array([child.visits for child in root.children])
        actions = np.array([child.action for child in root.children])
        
        if len(actions) == 0:
            policy = np.zeros(env.action_space.n, dtype=np.float32)
            return 0, policy
        
        visit_counts = visit_counts ** (1.0 / temperature)
        policy = np.zeros(env.action_space.n, dtype=np.float32)
        policy[actions] = visit_counts / np.sum(visit_counts) if np.sum(visit_counts) > 0 else np.ones(len(actions)) / len(actions)
        
        valid_indices = actions[actions < env.action_space.n]
        if len(valid_indices) == 0:
            print("Warning: No valid moves found. Defaulting to random action.")
            return np.random.choice(env.action_space.n), policy
        
        chosen_action = np.random.choice(valid_indices, p=policy[: len(valid_indices)])
        return int(chosen_action), policy
    
    def train(self, env: ConnectFourEnv, num_episodes: int = 100):
        rewards = []
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = self.choose_action(env)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            rewards.append(episode_reward)
        return rewards
    
    def save(self, filename: str):
        torch.save(self.network.state_dict(), filename)
    
    def load(self, filename: str):
        self.network.load_state_dict(torch.load(filename))
        self.network.eval()
