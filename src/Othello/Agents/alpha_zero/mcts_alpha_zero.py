import numpy as np
import math
import copy
import torch
from typing import List
from C4_adapter_only.Env.c4_env import RED, YELLOW

class ConnectFourMCTSNode:
    def __init__(self, env, parent=None, action=None, prior=0.0):
        self.env = copy.deepcopy(env)
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = []
        self.visits = 0
        self.reward = 0.0
        self.untried_actions = self._get_untried_actions()
        self.current_player = 1 if self.env.current_player == RED else -1

    def _get_untried_actions(self) -> List[int]:
        valid_moves_array = self.env._get_observation()["valid_moves"]
        return [i for i, is_valid in enumerate(valid_moves_array) if is_valid == 1]

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        obs = self.env._get_observation()
        return np.sum(obs["valid_moves"]) == 0

    def get_reward(self) -> float:
        winner = self.env.get_winner()
        if winner == RED:
            return 1.0 if self.current_player == 1 else -1.0
        elif winner == YELLOW:
            return -1.0 if self.current_player == 1 else 1.0
        return 0.0

    def best_child(self, c_param: float = 1.4) -> 'ConnectFourMCTSNode':
        choices_weights = []
        for child in self.children:
            Q = child.reward / (child.visits + 1e-9)
            exploration = c_param * child.prior * math.sqrt(self.visits + 1e-9) / (1 + child.visits)
            choices_weights.append(Q + exploration)
        return self.children[np.argmax(choices_weights)]

    def expand(self, network) -> 'ConnectFourMCTSNode':
        action = self.untried_actions.pop()
        obs_board = self.env._get_observation()["board"]
        obs_tensor = torch.tensor(obs_board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        network_output = network.initial_inference(obs_tensor)
        probs = torch.softmax(network_output.policy_logits, dim=1).detach().cpu().numpy().flatten()
        prior = probs[action]
        new_env = copy.deepcopy(self.env)
        _, _, _, _, _ = new_env.step(action)
        child_node = ConnectFourMCTSNode(new_env, parent=self, action=action, prior=prior)
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
            self.parent.backpropagate(-result)
