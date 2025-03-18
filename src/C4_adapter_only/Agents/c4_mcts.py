import numpy as np
import copy
import random
from typing import List
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
            # Check for a win
            for row in range(self.env.board.shape[0]):
                for col in range(self.env.board.shape[1]):
                    if self.env.board[row][col] != EMPTY:
                        if self.env._check_win(row, col):
                            self.terminal = True
                            return True
            
            # Check if board is full
            if np.all(self.env.board != EMPTY):
                self.terminal = True
                return True
                
            # Otherwise, not terminal
            self.terminal = False
        
        return self.terminal
        
    def get_reward(self) -> float:
        """Returns the reward for this terminal state."""
        # Check for a win
        for row in range(self.env.board.shape[0]):
            for col in range(self.env.board.shape[1]):
                if self.env.board[row][col] != EMPTY:
                    if self.env._check_win(row, col):
                        winner = self.env.board[row][col]
                        return 1.0 if (winner == RED and self.current_player == 1) or \
                                      (winner == YELLOW and self.current_player == -1) else -1.0
        
        # If board is full, it's a draw
        if np.all(self.env.board != EMPTY):
            return 0.0
            
        # Otherwise, not terminal
        return 0.0
    
    def expand(self) -> 'ConnectFourMCTSNode':
        """Adds a new child to the node by randomly choosing an unexplored action."""
        untried_actions = self._get_untried_actions()
        if not untried_actions:
            return self  
            
        action = untried_actions.pop()
        self.untried_actions = untried_actions  
    
        new_env = copy.deepcopy(self.env)
        _, _, _, _, _ = new_env.step(action)
        
        child_node = ConnectFourMCTSNode(new_env, parent=self, action=action)
        self.children.append(child_node)
        
        if not self.untried_actions:
            self.fully_expanded = True
            
        return child_node
    
    def simulate(self) -> float:
        """Simulates a game from this state to a terminal state by choosing random actions."""
        # Local copy of the environment for simulation
        sim_env = copy.deepcopy(self.env)
        sim_player = 1 if sim_env.current_player == RED else -1  
        
        # Limit the number of steps to avoid infinite loops
        max_steps = 42  
        step_count = 0
        
        while step_count < max_steps:
            # Get valid moves efficiently
            obs = sim_env._get_observation()
            valid_moves_indices = np.where(obs["valid_moves"] == 1)[0]
            
            if len(valid_moves_indices) == 0:
                break
            
            action = np.random.choice(valid_moves_indices)
            
            next_obs, _, terminated, _, info = sim_env.step(action)
            step_count += 1
            
            # Check if the game is over
            if terminated:
                break
        
        # Evaluate final state
        for row in range(sim_env.board.shape[0]):
            for col in range(sim_env.board.shape[1]):
                if sim_env.board[row][col] != EMPTY:
                    if sim_env._check_win(row, col):
                        winner = sim_env.board[row][col]
                        return 1.0 if (winner == RED and sim_player == 1) or \
                                      (winner == YELLOW and sim_player == -1) else -1.0
        
        # If no win, it's a draw
        return 0.0


class ConnectFourMCTSAgent(OthelloMCTSAgent):
    """MCTS Agent adapted for Connect Four."""
    
    def __init__(self, num_simulations: int = 800, exploration_weight: float = 1.4, 
                 num_processes: int = 1, batch_size: int = None):
        super().__init__(
            num_simulations=num_simulations,
            exploration_weight=exploration_weight,
            num_processes=num_processes,
            batch_size=batch_size
        )
        
        self.name = f"Connect4_MCTS_{self.actual_num_simulations}"
    
    def choose_action(self, env):
        """Chooses the best action according to MCTS."""
        # Check if valid actions exist
        obs = env._get_observation()
        valid_moves = [i for i, is_valid in enumerate(obs["valid_moves"]) if is_valid == 1]
        
        if not valid_moves:
            return 0
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        root = ConnectFourMCTSNode(copy.deepcopy(env))
        
        # Run simulations
        for _ in range(self.actual_num_simulations):
            node = root
            
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.exploration_weight)
            
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            result = node.simulate()
            
            node.backpropagate(result)
        
        if not root.children:
            return random.choice(valid_moves)
            
        # Select the action with the most visits
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.action