import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple
from C4_adapter_only.Env.c4_env import BOARD_ROWS, BOARD_COLS
from C4_adapter_only.Env.c4_env import RED, YELLOW
import numpy as np
import math
import copy


class NetworkOutput(NamedTuple):
    value: torch.Tensor
    policy_logits: torch.Tensor
    hidden_state: torch.Tensor


class ConnectFourAlphaZeroNet(nn.Module):
    def __init__(self, num_channels: int = 64):
        super(ConnectFourAlphaZeroNet, self).__init__()
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(num_channels * BOARD_ROWS * BOARD_COLS, 128)
        self.fc_policy = nn.Linear(128, BOARD_COLS)  # Policy head outputs move probabilities
        self.fc_value = nn.Linear(128, 1)  # Value head predicts state value
    
    def representation(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        hidden = F.relu(self.fc1(x))
        return hidden
    
    def prediction(self, hidden_state: torch.Tensor):
        policy_logits = self.fc_policy(hidden_state)
        value = torch.tanh(self.fc_value(hidden_state))
        return policy_logits, value
    
    def initial_inference(self, obs: torch.Tensor) -> NetworkOutput:
        hidden_state = self.representation(obs)
        policy_logits, value = self.prediction(hidden_state)
        return NetworkOutput(value, policy_logits, hidden_state)
    
    def forward(self, obs: torch.Tensor) -> NetworkOutput:
        return self.initial_inference(obs)


if __name__ == "__main__":
    net = ConnectFourAlphaZeroNet()
    dummy_input = torch.randn(1, 1, BOARD_ROWS, BOARD_COLS)
    output = net(dummy_input)
    print("Policy logits shape:", output.policy_logits.shape)
    print("Value shape:", output.value.shape)
    print("Hidden state shape:", output.hidden_state.shape)


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

    def _get_untried_actions(self):
        valid_moves_array = self.env._get_observation()["valid_moves"]
        return [i for i, is_valid in enumerate(valid_moves_array) if is_valid == 1]

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        obs = self.env._get_observation()
        return np.sum(obs["valid_moves"]) == 0

    def get_reward(self):
        winner = self.env.get_winner()
        if winner == RED:
            return 1.0 if self.current_player == 1 else -1.0
        elif winner == YELLOW:
            return -1.0 if self.current_player == 1 else 1.0
        return 0.0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.reward / child.visits if child.visits > 0 else 0) +
            c_param * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self, network):
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

    def simulate(self, network):
        if self.is_terminal():
            return self.get_reward()
        obs_board = self.env._get_observation()["board"]
        obs_tensor = torch.tensor(obs_board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        network_output = network.initial_inference(obs_tensor)
        value_est = network_output.value.item()
        return value_est

    def backpropagate(self, result):
        self.visits += 1
        self.reward += result
        if self.parent:
            self.parent.backpropagate(result)
