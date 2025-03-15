import numpy as np
import random
import math
import copy
import torch
from tqdm import tqdm
from typing import List
from Env.env import OthelloEnv, BLACK, WHITE, EMPTY, BOARD_SIZE

class MCTSNode:
    def __init__(self, env: OthelloEnv, parent=None, action=None, prior=0.0):
        self.env = copy.deepcopy(env)
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = []
        self.visits = 0
        self.reward = 0.0
        self.untried_actions = self._get_untried_actions()
        self.current_player = 1 if self.env.current_player == BLACK else -1

    def _get_untried_actions(self) -> List[int]:
        valid_moves_array = self.env._get_observation()["valid_moves"]
        return [i for i, is_valid in enumerate(valid_moves_array) if is_valid == 1]

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        obs = self.env._get_observation()
        return np.sum(obs["valid_moves"]) == 0

    def get_reward(self) -> float:
        black_count, white_count = self.env._get_score()
        if black_count > white_count:
            return 1.0 if self.current_player == 1 else -1.0
        elif white_count > black_count:
            return -1.0 if self.current_player == 1 else 1.0
        else:
            return 0.0

    def best_child(self, c_param: float = 1.4) -> 'MCTSNode':
        choices_weights = []
        for child in self.children:
            Q = child.reward / child.visits if child.visits > 0 else 0
            exploration = c_param * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            choices_weights.append(Q + exploration)
        return self.children[np.argmax(choices_weights)]

    def expand(self, network) -> 'MCTSNode':
        action = self.untried_actions.pop()
        obs_board = self.env._get_observation()["board"]
        obs_tensor = torch.tensor(obs_board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        network_output = network.initial_inference(obs_tensor)
        probs = torch.softmax(network_output.policy_logits, dim=1).detach().cpu().numpy().flatten()
        prior = probs[action]
        new_env = copy.deepcopy(self.env)
        _, _, _, _, _ = new_env.step(action)
        child_node = MCTSNode(new_env, parent=self, action=action, prior=prior)
        self.children.append(child_node)
        return child_node

    def simulate(self, network) -> float:
        if self.is_terminal():
            return self.get_reward()
        obs_board = self.env._get_observation()["board"]
        obs_tensor = torch.tensor(obs_board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        network_output = network.initial_inference(obs_tensor)
        value_est = network_output.value.item()
        return value_est

    def backpropagate(self, result: float) -> None:
        self.visits += 1
        self.reward += result
        if self.parent:
            self.parent.backpropagate(result)

class MCTSAgent:
    def __init__(self, network, num_simulations: int = 500, exploration_weight: float = 1.4):
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.network = network
        self.name = f"MCTS_NN_{num_simulations}"

    def choose_action(self, env: OthelloEnv) -> int:
        root = MCTSNode(env)
        if root.is_terminal():
            valid_moves = [i for i, is_valid in enumerate(root.env._get_observation()["valid_moves"]) if is_valid == 1]
            return random.choice(valid_moves) if valid_moves else 0
        for _ in range(self.num_simulations):
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.exploration_weight)
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand(self.network)
            result = node.simulate(self.network)
            node.backpropagate(result)
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action
    
    def choose_action_and_policy(self, env, temperature=1.0):
        # identical to choose_action, but we also track child visit counts to form a policy distribution
        root = MCTSNode(env)
        if root.is_terminal():
            valid_moves = [i for i, is_valid in enumerate(root.env._get_observation()["valid_moves"]) if is_valid == 1]
            if valid_moves:
                return np.random.choice(valid_moves), np.zeros(env.action_space.n)
            else:
                return 0, np.zeros(env.action_space.n)

        for _ in range(self.num_simulations):
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.exploration_weight)
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand(self.network)
            result = node.simulate(self.network)
            node.backpropagate(result)

        # Build a policy distribution from visit counts
        visit_counts = np.zeros(env.action_space.n)
        for child in root.children:
            visit_counts[child.action] = child.visits

        # Apply a temperature parameter to control exploration
        if np.sum(visit_counts) > 0:
            visit_counts = visit_counts ** (1.0 / temperature)
            policy = visit_counts / np.sum(visit_counts)
        else:
            policy = np.zeros(env.action_space.n)

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action, policy

    def train(self, env: OthelloEnv, num_episodes: int = 100) -> List[float]:
        rewards = []
        for episode in tqdm(range(num_episodes), desc=f"Training {self.name}"):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.choose_action(env)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            rewards.append(episode_reward)
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{num_episodes}, Average Reward: {np.mean(rewards[-10:]):.2f}")
        return rewards

    def save(self, filename: str) -> None:
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({
                'num_simulations': self.num_simulations,
                'exploration_weight': self.exploration_weight
            }, f)

    def load(self, filename: str) -> None:
        import pickle
        with open(filename, 'rb') as f:
            params = pickle.load(f)
            self.num_simulations = params['num_simulations']
            self.exploration_weight = params['exploration_weight']
