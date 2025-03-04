# mcts_node.py
import numpy as np
import math
import random
from collections import defaultdict

class MonteCarloTreeSearchNode:
    def __init__(self, state, parent=None, parent_action=None):
        """
        state: an instance of OthelloState (from state.py)
        parent: the parent MCTS node
        parent_action: the move (tuple) that was applied to get to this state
        """
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []  # List of child nodes
        self._number_of_visits = 0
        self._results = defaultdict(int)  # For storing win/loss counts (keys: 1 for win, -1 for loss)
        self._results[1] = 0
        self._results[-1] = 0
        # Get the list of legal actions for this state
        self._untried_actions = self.untried_actions()
    
    def untried_actions(self):
        # Assumes your OthelloState provides get_legal_actions() returning a list of moves (e.g., [(row,col), ...])
        return self.state.get_legal_actions()
    
    def q(self):
        # Returns the net wins (wins - losses)
        wins = self._results[1]
        losses = self._results[-1]
        return wins - losses
    
    def n(self):
        # Returns the number of visits to this node
        return self._number_of_visits
    
    def expand(self):
        # Remove one untried action and apply it to get a new state, then create a child node.
        action = self._untried_actions.pop()
        next_state = self.state.move(action)  # Apply the move; move() should return a new cloned state.
        child_node = MonteCarloTreeSearchNode(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node
    
    def is_terminal_node(self):
        # Check if the state is terminal (game over)
        return self.state.is_game_over()
    
    def rollout(self):
        # Simulate a random playout (rollout) until the game is over.
        current_rollout_state = self.state.clone()  # Clone the state for simulation.
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            if not possible_moves:
                break
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result()
    
    def backpropagate(self, result):
        # Update the statistics: number of visits and results, and propagate up to the parent.
        self._number_of_visits += 1
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)
    
    def is_fully_expanded(self):
        # The node is fully expanded if there are no untried actions left.
        return len(self._untried_actions) == 0
    
    def best_child(self, c_param=0.1):
        # Use UCB1 to select the best child node.
        choices_weights = []
        for child in self.children:
            if child.n() == 0:
                # If a child hasn't been visited, treat its UCB score as infinity.
                ucb = float('inf')
            else:
                ucb = (child.q() / child.n()) + c_param * math.sqrt((2 * math.log(self.n())) / child.n())
            choices_weights.append(ucb)
        return self.children[np.argmax(choices_weights)]
    
    def rollout_policy(self, possible_moves):
        # Simple random policy for simulation.
        return possible_moves[random.randint(0, len(possible_moves)-1)]
    
    def _tree_policy(self):
        # Traverse the tree using UCB until a node that is not fully expanded or a terminal state is reached.
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
    
    def best_action(self, simulation_no=100):
        # Run MCTS simulations for a fixed number of iterations, then return the best action from the root.
        for _ in range(simulation_no):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        # Return the action of the child with the highest average reward.
        best_child_node = self.best_child(c_param=0.)
        return best_child_node.parent_action  # The action that led to that child.


