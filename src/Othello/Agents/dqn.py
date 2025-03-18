import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict, Any
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Othello.Env.env import OthelloEnv, BOARD_SIZE

class DQNModel(nn.Module):
    def __init__(self, state_shape: Tuple[int, int] = (BOARD_SIZE, BOARD_SIZE), action_size: int = BOARD_SIZE * BOARD_SIZE):
        super(DQNModel, self).__init__()
        
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


class DQNAgent:
    def __init__(
        self,
        state_shape: Tuple[int, int] = (BOARD_SIZE, BOARD_SIZE),
        action_size: int = BOARD_SIZE * BOARD_SIZE,
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
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration parameter
        self.epsilon_min = epsilon_min  # Minimum epsilon
        self.epsilon_decay = epsilon_decay  # Epsilon decay rate
        self.learning_rate = learning_rate  # Learning rate
        self.batch_size = batch_size  # Batch size
        self.update_target_freq = update_target_freq  # Target network update frequency
        self.reward_scale = reward_scale  # Reward scaling factor
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.step_counter = 0
        
        # Models
        self.model = DQNModel(state_shape, action_size).to(self.device)
        self.target_model = DQNModel(state_shape, action_size).to(self.device)
        self.update_target_model()  # Synchronize the two models

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.name = "DQN"
    
    def update_target_model(self) -> None:
        """Updates the target model with the weights of the main model."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def preprocess_state(self, obs: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocesses the observation for the neural network."""
        board = torch.tensor(obs["board"], dtype=torch.float32).reshape(1, 1, *self.state_shape).to(self.device)

        valid_moves = torch.tensor(obs["valid_moves"], dtype=torch.float32).reshape(1, -1).to(self.device)

        player = torch.tensor([1.0 if obs["current_player"] == 0 else -1.0], dtype=torch.float32).reshape(1, 1).to(self.device)
        
        return board, valid_moves, player
    
    def remember(self, state: Dict[str, np.ndarray], action: int, reward: float, 
                next_state: Dict[str, np.ndarray], done: bool) -> None:
        """Stores the experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, env: OthelloEnv) -> int:
        """Chooses an action according to the epsilon-greedy policy."""
        obs = env._get_observation()
        
        # Valid moves
        valid_moves = obs["valid_moves"]
        valid_actions = [i for i, is_valid in enumerate(valid_moves) if is_valid == 1]
        
        # If there are no valid moves, return 0 (default action)
        if not valid_actions:
            return 0
        
        # Exploration (epsilon-greedy)
        if np.random.rand() <= self.epsilon:
            return np.random.choice(valid_actions)
        
        # Exploitation
        state = self.preprocess_state(obs)
        
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(*state).cpu().numpy()[0]
        self.model.train()
        
        # Find the valid action with the highest Q-value
        return int(np.argmax(q_values))
    
    def replay(self) -> float:
        """Trains the model on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        minibatch = random.sample(self.memory, self.batch_size)
        total_loss = 0.0
    
        states_boards = []
        states_valid_moves = []
        states_players = []
        actions = []
        rewards = []
        next_states_boards = []
        next_states_valid_moves = []
        next_states_players = []
        dones = []
        
        for state, action, reward, next_state, done in minibatch:
            # Preprocess states
            board, valid_moves, player = self.preprocess_state(state)
            
            states_boards.append(board)
            states_valid_moves.append(valid_moves)
            states_players.append(player)
            actions.append(action)
            rewards.append(reward)
            
            # Preprocess next states
            next_board, next_valid_moves, next_player = self.preprocess_state(next_state)
            
            next_states_boards.append(next_board)
            next_states_valid_moves.append(next_valid_moves)
            next_states_players.append(next_player)
            dones.append(done)
        
        # Convert to batch tensors
        states_boards = torch.cat(states_boards, dim=0)
        states_valid_moves = torch.cat(states_valid_moves, dim=0)
        states_players = torch.cat(states_players, dim=0)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_boards = torch.cat(next_states_boards, dim=0)
        next_states_valid_moves = torch.cat(next_states_valid_moves, dim=0)
        next_states_players = torch.cat(next_states_players, dim=0)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Predict and Select Q-values for current states
        self.model.train()
        current_q_values = self.model(states_boards, states_valid_moves, states_players)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(next_states_boards, next_states_valid_moves, next_states_players)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Reduce epsilon (exploration)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target model periodically
        self.step_counter += 1
        if self.step_counter % self.update_target_freq == 0:
            self.update_target_model()
        
        return loss.item()

    def train(self, env: OthelloEnv, num_episodes: int = 100, 
            opponents: Dict[str, Any] = None, opponent_probs: Dict[str, float] = None) -> Tuple[List[float], List[float]]:
        """Trains the agent against specified opponents for a number of episodes."""
        
        if opponents is None:
            opponents = {'self': self}  # Default to self-play

        if opponent_probs is None:
            opponent_probs = {name: 1.0 / len(opponents) for name in opponents}  # Equal probability

        total_prob = sum(opponent_probs.values())
        opponent_probs = {name: prob / total_prob for name, prob in opponent_probs.items()}

        opponent_names = list(opponent_probs.keys())
        cum_probs = np.cumsum([opponent_probs[name] for name in opponent_names])

        rewards, losses = [], []
        opponent_counts = {name: 0 for name in opponent_names}

        original_lr = self.learning_rate  # Store original learning rate

        for episode in range(num_episodes):
            # Reset environment
            state, _ = env.reset()
            total_reward = 0
            episode_losses = []
            done = False

            # Select opponent
            rand_val = np.random.random()
            opponent_idx = np.searchsorted(cum_probs, rand_val)
            opponent_name = opponent_names[opponent_idx]
            opponent = opponents[opponent_name]
            opponent_counts[opponent_name] += 1
            is_self_play = opponent_name == 'self'

            dqn_player = np.random.choice([0, 1])  # 0 = BLACK, 1 = WHITE

            while not done:
                current_player = state["current_player"]

                if current_player == dqn_player:  # DQN's turn
                    action = self.choose_action(env)
                    if isinstance(action, tuple): 
                        action = action[0]

                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    # Ensure reward scaling is safe
                    scaled_reward = reward * self.reward_scale if hasattr(self, "reward_scale") else reward
                    
                    self.remember(state, action, scaled_reward, next_state, done)
                    if len(self.memory) >= self.batch_size:
                        loss = self.replay()
                        if loss > 0:
                            episode_losses.append(loss)
                    
                    total_reward += scaled_reward

                else:  # Opponent's turn
                    action = self.choose_action(env) if is_self_play else opponent.choose_action(env)
                    if isinstance(action, tuple):
                        action = action[0]

                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    # Ensure reward scaling is safe
                    scaled_reward = reward * self.reward_scale if hasattr(self, "reward_scale") else reward

                state = next_state

            rewards.append(total_reward)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            losses.append(avg_loss)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"   Opponent: {opponent_name}")
                print(f"   Total Reward: {total_reward:.2f}")
                print(f"   Avg Loss: {avg_loss:.4f}")
                print(f"   Epsilon: {self.epsilon:.4f}")

        return rewards, losses

    
    def save(self, filename: str) -> None:
        """Saves the model and agent parameters."""
        torch.save(self.model.state_dict(), filename + ".pt")
        
        # Save additional parameters
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump({
                'epsilon': self.epsilon,
                'gamma': self.gamma,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'update_target_freq': self.update_target_freq
            }, f)
    
    def load(self, filename: str) -> None:
        """Loads the model and agent parameters."""
        # Load the model
        self.model.load_state_dict(torch.load(filename + ".pt"))
        self.target_model.load_state_dict(torch.load(filename + ".pt"))
        
        # Load parameters
        with open(filename + ".pkl", 'rb') as f:
            params = pickle.load(f)
            self.epsilon = params['epsilon']
            self.gamma = params['gamma']
            self.epsilon_min = params['epsilon_min']
            self.epsilon_decay = params['epsilon_decay']
            self.learning_rate = params['learning_rate']
            self.batch_size = params['batch_size']
            self.update_target_freq = params['update_target_freq']