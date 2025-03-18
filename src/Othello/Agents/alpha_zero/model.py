import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple

class NetworkOutput(NamedTuple):
    value: torch.Tensor
    policy_logits: torch.Tensor
    hidden_state: torch.Tensor

class SimpleAlphaZeroNet(nn.Module):
    def __init__(self, board_size: int = 8, num_channels: int = 64):
        super(SimpleAlphaZeroNet, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(num_channels * board_size * board_size, 128)
        self.fc_policy = nn.Linear(128, board_size * board_size)
        self.fc_value = nn.Linear(128, 1)
    
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
    net = SimpleAlphaZeroNet()
    dummy_input = torch.randn(1, 1, 8, 8)
    output = net(dummy_input)
    print("Policy logits shape:", output.policy_logits.shape)
    print("Value shape:", output.value.shape)
    print("Hidden state shape:", output.hidden_state.shape)
