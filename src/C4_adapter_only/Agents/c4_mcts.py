import numpy as np
import copy
from typing import List, Tuple
from Othello.Agents.mcts import MCTSNode, MCTSAgent as OthelloMCTSAgent
from C4_adapter_only.Env.c4_env import RED, YELLOW, EMPTY


class ConnectFourMCTSNode(MCTSNode):
    """Customized MCTS Node for Connect Four."""
    
    def __init__(self, env, parent=None, action=None):
        super().__init__(env, parent, action)
        # Override current player to use Connect Four constants
        self.current_player = 1 if env.current_player == RED else -1
    
    def _get_untried_actions(self) -> List[int]:
        """Returns all legal actions not yet tried from this state."""
        if self.untried_actions is None:
            valid_moves_array = self.env._get_observation()["valid_moves"]
            self.untried_actions = [i for i, is_valid in enumerate(valid_moves_array) if is_valid == 1]
        return self.untried_actions.copy()
    
    def is_terminal(self) -> bool:
        """Checks if this state is terminal (win, loss, or draw)."""
        if self.terminal is None:
            # Check first for a win - look at the last move if available
            if self.action is not None:
                # Find the row where the piece landed
                col = self.action
                # We need to find which row the piece landed in
                for row in range(self.env.board.shape[0]):
                    if self.env.board[row][col] != EMPTY:
                        if self.env._check_win(row, col):
                            self.terminal = True
                            return True
                        break
            
            # If no win, check if board is full
            if np.all(self.env.board != EMPTY):
                self.terminal = True
                return True
            
            # If no valid moves, it's terminal
            obs = self.env._get_observation()
            self.terminal = np.sum(obs["valid_moves"]) == 0
        
        return self.terminal
    

    def get_reward(self) -> float:
        """Returns the reward for this terminal state."""
        # Check if the last move caused a win
        if self.action is not None:
            col = self.action
            for row in range(self.env.board.shape[0]):
                if self.env.board[row][col] != EMPTY:
                    if self.env._check_win(row, col):
                        # The current_player of the node is the one who made the move
                        return 1.0 if self.env.board[row][col] == RED else -1.0
                    break
        # Check for draw
        if np.all(self.env.board != EMPTY):
            return 0.0
        # Fallback to original reward calculation based on scores (if needed)
        red_count, yellow_count = self.env._get_score()
        if red_count > yellow_count:
            return 1.0 if self.current_player == 1 else -1.0
        elif yellow_count > red_count:
            return -1.0 if self.current_player == 1 else 1.0
        else:
            return 0.0
    def get_reward(self) -> float:
        """Returns the reward for this terminal state."""
        # Need to check if this is a win, loss, or draw
        # First check for a win by examining the last move
        if self.action is not None:
            col = self.action
            # Find which row the piece landed in
            for row in range(self.env.board.shape[0]):
                if self.env.board[row][col] != EMPTY:
                    if self.env._check_win(row, col):
                        # If this player won, return positive reward
                        last_player = self.env.board[row][col]
                        player_won = 1 if last_player == RED else -1
                        return 1.0 if player_won == self.current_player else -1.0
                    break
        
        # If no win, it's a draw
        return 0.0
    
    def simulate(self) -> float:
        """Simulates a game from this state to a terminal state by choosing random actions."""
        # Local copy of the environment for simulation
        sim_env = copy.deepcopy(self.env)
        terminated = False
        
        # Limit the number of steps to avoid infinite loops
        max_steps = 42  # Maximum possible moves in Connect Four (7*6)
        step_count = 0
        
        while not terminated and step_count < max_steps:
            # Get valid moves efficiently
            obs = sim_env._get_observation()
            valid_moves_indices = np.where(obs["valid_moves"] == 1)[0]
            
            # If no valid moves, the game is over
            if len(valid_moves_indices) == 0:
                break
            
            # Choose a random action (faster with numpy)
            action = np.random.choice(valid_moves_indices)
            
            # Execute the action
            _, _, terminated, _, info = sim_env.step(action)
            step_count += 1
            
            # Check if this move resulted in a win
            if terminated and "winner" in info:
                if info["winner"] == "RED":
                    return 1.0 if self.current_player == 1 else -1.0
                elif info["winner"] == "YELLOW":
                    return -1.0 if self.current_player == 1 else 1.0
                else:  # Draw
                    return 0.0
        
        # If we reached here without a clear winner, determine based on the board state
        # First, check if anyone has connected four
        for row in range(sim_env.board.shape[0]):
            for col in range(sim_env.board.shape[1]):
                if sim_env.board[row][col] != EMPTY:
                    if sim_env._check_win(row, col):
                        winner = sim_env.board[row][col]
                        return 1.0 if (winner == RED and self.current_player == 1) or \
                                    (winner == YELLOW and self.current_player == -1) else -1.0
        
        # If no one has connected four, it's a draw
        return 0.0


class ConnectFourMCTSAgent(OthelloMCTSAgent):
    """MCTS Agent adapted for Connect Four."""
    
    def __init__(self, num_simulations: int = 800, exploration_weight: float = 1.4, 
                 num_processes: int = None, batch_size: int = None):
        super().__init__(
            num_simulations=num_simulations,
            exploration_weight=exploration_weight,
            num_processes=num_processes,
            batch_size=batch_size
        )
        
        # Update name
        self.name = f"Connect4_MCTS_{self.actual_num_simulations}"
    
    def choose_action(self, env):
        """Chooses the best action according to parallelized MCTS."""
        # Check if valid actions exist
        obs = env._get_observation()
        valid_moves = [i for i, is_valid in enumerate(obs["valid_moves"]) if is_valid == 1]
        
        # If no valid moves, return a default action
        if not valid_moves:
            return 0
        
        # If only one valid move, return it immediately
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Create a Connect Four specific MCTS node as the root
        root = ConnectFourMCTSNode(copy.deepcopy(env))
        
        # Run simulations
        for _ in range(self.actual_num_simulations):
            node = root
            
            # Selection
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.exploration_weight)
            
            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            # Simulation
            result = node.simulate()
            
            # Backpropagation
            node.backpropagate(result)
        
        # Select the action with the most visits
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.action