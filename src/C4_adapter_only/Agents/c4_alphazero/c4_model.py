import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple
from C4_adapter_only.Env.c4_env import BOARD_ROWS, BOARD_COLS


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
