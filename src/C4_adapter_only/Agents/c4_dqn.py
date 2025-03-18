import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from Othello.Agents.dqn import DQNAgent as OthelloDQNAgent
from C4_adapter_only.Env.c4_env import BOARD_ROWS, BOARD_COLS


class ConnectFourDQNModel(nn.Module):
    def __init__(self, state_shape: Tuple[int, int] = (BOARD_ROWS, BOARD_COLS), action_size: int = BOARD_COLS):
        super(ConnectFourDQNModel, self).__init__()
        
        # Convolutional layers to process the board
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Calculate the dimension after convolution
        conv_output_size = 128 * state_shape[0] * state_shape[1]
        
        # Dense layers
        self.fc1 = nn.Linear(conv_output_size + action_size + 1, 256)  # +action_size for valid moves, +1 for player
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, board, valid_moves, player):
        x = F.relu(self.bn1(self.conv1(board)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  
        
        x = torch.cat([x, valid_moves, player], dim=1)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        
        # Apply mask to forbid invalid actions
        masked_q_values = torch.where(
            valid_moves.bool(),
            q_values,
            torch.ones_like(q_values) * (-1e9)  # Very low value for invalid moves
        )
        
        return masked_q_values


class ConnectFourDQNAgent(OthelloDQNAgent):
    def __init__(
        self,
        state_shape: Tuple[int, int] = (BOARD_ROWS, BOARD_COLS),
        action_size: int = BOARD_COLS,
        memory_size: int = 10000,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        update_target_freq: int = 10,
        reward_scale: float = 1.0,
    ):
        super().__init__(
            state_shape=state_shape,
            action_size=action_size,
            memory_size=memory_size,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            learning_rate=learning_rate,
            batch_size=batch_size,
            update_target_freq=update_target_freq,
            reward_scale=reward_scale,
        )
        
        self.model = ConnectFourDQNModel(state_shape, action_size).to(self.device)
        self.target_model = ConnectFourDQNModel(state_shape, action_size).to(self.device)
        self.update_target_model()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.name = "Connect4_DQN"
    
    def preprocess_state(self, obs):
        """Preprocesses the observation for the neural network."""
        board = torch.tensor(obs["board"], dtype=torch.float32).reshape(1, 1, *self.state_shape).to(self.device)
        
        valid_moves = torch.tensor(obs["valid_moves"], dtype=torch.float32).reshape(1, -1).to(self.device)
        
        player = torch.tensor([1.0 if obs["current_player"] == 0 else -1.0], dtype=torch.float32).reshape(1, 1).to(self.device)
        
        return board, valid_moves, player