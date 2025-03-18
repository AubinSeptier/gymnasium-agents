import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, List, Dict, Any, Optional


# Constants
RED = 1      # First player
YELLOW = -1  # Second player
EMPTY = 0
BOARD_ROWS = 6
BOARD_COLS = 7


class ConnectFourEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Define the action space
        # 7 possible columns to drop a piece
        self.action_space = spaces.Discrete(BOARD_COLS)
        
        # Define the observation space
        # The game board (6x7) can have 3 values for each cell (-1, 0, 1)
        # Plus an indicator for the current player (1 or -1)
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=-1, high=1, shape=(BOARD_ROWS, BOARD_COLS), dtype=np.int8),
            "current_player": spaces.Discrete(2),  # 0 for RED (1), 1 for YELLOW (-1)
            "valid_moves": spaces.Box(low=0, high=1, shape=(BOARD_COLS,), dtype=np.int8)
        })
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Initialize the state
        self.board = None
        self.current_player = None
        self.valid_moves = None
        self.done = None
        self.selected_col = None
        self.cumulative_rewards = {"RED": 0, "YELLOW": 0}
        
        # Reset to initialize values
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize the board (empty)
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        
        # RED starts
        self.current_player = RED
        
        # Reset the done state
        self.done = False
        
        # Calculate valid moves
        self.valid_moves = self._get_valid_moves()
        
        # Initial position of the selection cursor (centered)
        self.selected_col = BOARD_COLS // 2
        
        # Reset cumulative rewards
        self.cumulative_rewards = {"RED": 0, "YELLOW": 0}
        
        # Return the initial observation and information
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        # Action is the column (0-6) where the piece will be dropped
        col = action
        reward = 0  #  Explicitly initialize reward

        # Check if the action is valid
        if not self._is_valid_move(col):
            # Invalid action, end the episode with negative reward
            reward = -10.0
            return self._get_observation(), float(reward), False, True, {"info": "Invalid move"}

        # Find the row where the piece will land
        row = self._get_next_open_row(col)

        # Place the piece
        self.board[row][col] = self.current_player

        # Check if the current player won
        if self._check_win(row, col):
            self.done = True
            reward = 1.0  # Winning reward
            
            # Update cumulative rewards
            if self.current_player == RED:
                self.cumulative_rewards["RED"] += reward
            else:
                self.cumulative_rewards["YELLOW"] += reward

            return self._get_observation(), float(reward), True, False, {
                "winner": "RED" if self.current_player == RED else "YELLOW"
            }

        # Check if board is full (draw)
        if np.all(self.board != EMPTY):
            self.done = True
            reward = 0.0  #  Explicitly set reward for draws
            return self._get_observation(), float(reward), True, False, {"winner": "DRAW"}

        # Switch to the next player
        self.current_player = -self.current_player

        # Update valid moves
        self.valid_moves = self._get_valid_moves()

        #  Ensure reward is always defined in every return path
        return self._get_observation(), float(reward), False, False, {}

    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Returns the current observation."""
        # Convert valid moves to binary array
        valid_moves_array = np.zeros(BOARD_COLS, dtype=np.int8)
        for col in self.valid_moves:
            valid_moves_array[col] = 1
        
        return {
            "board": self.board.copy(),
            "current_player": 0 if self.current_player == RED else 1,
            "valid_moves": valid_moves_array
        }
    
    def move_selection(self, direction: str) -> None:
        """Moves the selection cursor left or right."""
        if direction == "left" and self.selected_col > 0:
            self.selected_col -= 1
        elif direction == "right" and self.selected_col < BOARD_COLS - 1:
            self.selected_col += 1
    
    def get_selected_position(self) -> int:
        """Returns the selected column as an action (0-6)."""
        return self.selected_col
    
    def is_selected_valid(self) -> bool:
        """Checks if the selected column is a valid move."""
        return self.selected_col in self.valid_moves
    
    def _get_valid_moves(self) -> List[int]:
        """Returns the list of valid moves (columns that aren't full)."""
        valid_moves = []
        
        for col in range(BOARD_COLS):
            if self._is_valid_move(col):
                valid_moves.append(col)
        
        return valid_moves
    
    def _is_valid_move(self, col: int) -> bool:
        """Checks if a move is valid (column isn't full)."""
        # Check if column is in bounds
        if not (0 <= col < BOARD_COLS):
            return False
        
        # Check if the top cell of the column is empty
        return self.board[BOARD_ROWS-1][col] == EMPTY
    
    def _get_next_open_row(self, col: int) -> int:
        """Returns the row where a piece would land if dropped in the given column."""
        for row in range(BOARD_ROWS):
            if self.board[row][col] == EMPTY:
                return row
        
        # This should never happen if _is_valid_move is called first
        return -1
    
    def _check_win(self, row: int, col: int) -> bool:
        """Checks if the last move resulted in a win."""
        player = self.board[row][col]
        
        # Check horizontal
        for c in range(max(0, col-3), min(col+1, BOARD_COLS-3)):
            if (self.board[row][c] == player and 
                self.board[row][c+1] == player and 
                self.board[row][c+2] == player and 
                self.board[row][c+3] == player):
                return True
        
        # Check vertical
        for r in range(max(0, row-3), min(row+1, BOARD_ROWS-3)):
            if (self.board[r][col] == player and 
                self.board[r+1][col] == player and 
                self.board[r+2][col] == player and 
                self.board[r+3][col] == player):
                return True
        
        # Check positive diagonal (/)
        for r, c in zip(range(max(0, row-3), min(row+1, BOARD_ROWS-3)), 
                        range(max(0, col-3), min(col+1, BOARD_COLS-3))):
            if (self.board[r][c] == player and 
                self.board[r+1][c+1] == player and 
                self.board[r+2][c+2] == player and 
                self.board[r+3][c+3] == player):
                return True
        
        # Check negative diagonal (\)
        for r, c in zip(range(min(BOARD_ROWS-1, row+3), max(row, 2), -1), 
                        range(max(0, col-3), min(col+1, BOARD_COLS-3))):
            if (self.board[r][c] == player and 
                self.board[r-1][c+1] == player and 
                self.board[r-2][c+2] == player and 
                self.board[r-3][c+3] == player):
                return True
        
        return False
    
    def _get_score(self) -> Tuple[int, int]:
        """Returns the score (number of red pieces, number of yellow pieces)."""
        red_count = np.sum(self.board == RED)
        yellow_count = np.sum(self.board == YELLOW)
        return red_count, yellow_count
    
    def render(self):
        """Render method (placeholder - would need implementation)."""
        pass
    
    def close(self):
        """Closes the environment."""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()