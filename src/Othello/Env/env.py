import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, List, Dict, Any, Optional


# Constants
BLACK = 1
WHITE = -1
EMPTY = 0
BOARD_SIZE = 8


class OthelloEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Define the action space
        # 64 possible positions on the board (8x8)
        self.action_space = spaces.Discrete(BOARD_SIZE * BOARD_SIZE)
        
        # Define the observation space
        # The game board (8x8) can have 3 values for each cell (-1, 0, 1)
        # Plus an indicator for the current player (1 or -1)
        # Board: 8x8 = 64 positions with 3 possible values per position
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=-1, high=1, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int8),
            "current_player": spaces.Discrete(2),  # 0 for BLACK (1), 1 for WHITE (-1)
            "valid_moves": spaces.Box(low=0, high=1, shape=(BOARD_SIZE * BOARD_SIZE,), dtype=np.int8)
        })
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Initialize the state
        self.board = None
        self.current_player = None
        self.valid_moves = None
        self.done = None
        self.selected_x = None
        self.selected_y = None
        self.cumulative_rewards = {"BLACK": 0, "WHITE": 0}
        
        # Reset to initialize values
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize the board with the 4 starting pieces
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE
        
        # Black starts
        self.current_player = BLACK
        
        # Reset the done state
        self.done = False
        
        # Calculate valid moves
        self.valid_moves = self._get_valid_moves()
        
        # Initial position of the selection cursor (centered)
        self.selected_x = 3
        self.selected_y = 3
        
        # Reset cumulative rewards
        self.cumulative_rewards = {"BLACK": 0, "WHITE": 0}
        
        # Return the initial observation and information
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        # Convert the action (0-63) to coordinates (row, col)
        row, col = action // BOARD_SIZE, action % BOARD_SIZE
        reward = 0
        
        # Check if the action is valid
        if not self._is_valid_move(row, col):
            # Invalid action, end the episode with negative reward
            return self._get_observation(), -10.0, False, True, {"info": "Invalid move"}
        
        # Perform the action (place a piece)
        self._place_piece(row, col)
        
        # Calculate the reward (number of captured pieces)
        black_count, white_count = self._get_score()
        if self.current_player == BLACK:
            reward = black_count - white_count
            self.cumulative_rewards["BLACK"] += reward
        else:
            reward = white_count - black_count
            self.cumulative_rewards["WHITE"] += reward
        
        # Switch to the next player
        self.current_player = -self.current_player
        
        # Update valid moves
        self.valid_moves = self._get_valid_moves()
        
        # If the current player has no valid moves, skip their turn
        if len(self.valid_moves) == 0:
            self.current_player = -self.current_player
            self.valid_moves = self._get_valid_moves()
            
            # If after skipping the turn, there are still no valid moves,
            # the game is over
            if len(self.valid_moves) == 0:
                self.done = True
        
        # Check if the game is over
        terminated = self.done
        truncated = False  # We are not using step limit
        
        # Update the selection cursor position to be on a valid move if possible
        if not terminated and self.valid_moves:
            self.selected_y, self.selected_x = self.valid_moves[0]
        
        # Additional information
        info = {
            "score_black": black_count,
            "score_white": white_count,
            "valid_moves_count": len(self.valid_moves),
            "cumulative_reward_black": self.cumulative_rewards["BLACK"],
            "cumulative_reward_white": self.cumulative_rewards["WHITE"]
        }
        
        if terminated:
            # Determine the winner at the end of the game
            if black_count > white_count:
                info["winner"] = "BLACK"
            elif white_count > black_count:
                info["winner"] = "WHITE"
            else:
                info["winner"] = "DRAW"
        
        return self._get_observation(), float(reward), terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Returns the current observation."""
        # Convert valid moves to binary array
        valid_moves_array = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.int8)
        for row, col in self.valid_moves:
            valid_moves_array[row * BOARD_SIZE + col] = 1
        
        return {
            "board": self.board.copy(),
            "current_player": 0 if self.current_player == BLACK else 1,
            "valid_moves": valid_moves_array
        }
    
    def move_selection(self, direction: str) -> None:
        """Moves the selection cursor in the specified direction."""
        if direction == "up" and self.selected_y > 0:
            self.selected_y -= 1
        elif direction == "down" and self.selected_y < BOARD_SIZE - 1:
            self.selected_y += 1
        elif direction == "left" and self.selected_x > 0:
            self.selected_x -= 1
        elif direction == "right" and self.selected_x < BOARD_SIZE - 1:
            self.selected_x += 1
    
    def get_selected_position(self) -> int:
        """Returns the selected position as an action (0-63)."""
        return self.selected_y * BOARD_SIZE + self.selected_x
    
    def is_selected_valid(self) -> bool:
        """Checks if the selected position corresponds to a valid move."""
        return (self.selected_y, self.selected_x) in self.valid_moves
    
    def _get_valid_moves(self) -> List[Tuple[int, int]]:
        """Returns the list of valid moves for the current player."""
        valid_moves = []
        
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self._is_valid_move(row, col):
                    valid_moves.append((row, col))
        
        return valid_moves
    
    def _is_valid_move(self, row: int, col: int) -> bool:
        """Checks if a move is valid."""
        # If the cell is not empty, the move is not valid
        if self.board[row][col] != EMPTY:
            return False
        
        # Directions: horizontal, vertical, and diagonal
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        # For each direction, check if we can capture pieces
        for dr, dc in directions:
            if self._can_capture(row, col, dr, dc):
                return True
        
        return False
    
    def _can_capture(self, row: int, col: int, dr: int, dc: int) -> bool:
        """Checks if we can capture pieces in a given direction."""
        opponent = -self.current_player
        
        # Check if there is at least one adjacent opponent piece
        r, c = row + dr, col + dc
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == opponent):
            return False
        
        # Continue in this direction to find a piece of the current player
        r, c = r + dr, c + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            if self.board[r][c] == EMPTY:
                return False
            if self.board[r][c] == self.current_player:
                return True
            r, c = r + dr, c + dc
        
        return False
    
    def _place_piece(self, row: int, col: int) -> None:
        """Places a piece on the board and captures opponent pieces."""
        self.board[row][col] = self.current_player
        
        # Directions: horizontal, vertical, and diagonal
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        # For each direction, capture pieces
        for dr, dc in directions:
            self._capture(row, col, dr, dc)
    
    def _capture(self, row: int, col: int, dr: int, dc: int) -> None:
        """Captures pieces in a given direction."""
        if not self._can_capture(row, col, dr, dc):
            return
        
        # Capture opponent pieces
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == -self.current_player:
            self.board[r][c] = self.current_player
            r, c = r + dr, c + dc
    
    def _get_score(self) -> Tuple[int, int]:
        """Returns the score (number of black pieces, number of white pieces)."""
        black_count = np.sum(self.board == BLACK)
        white_count = np.sum(self.board == WHITE)
        return black_count, white_count
    
    def render(self):
        """Render method. Implementation is in play_human.py."""
        pass
    
    def close(self):
        """Closes the environment."""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()