
import math
import random
from state import OthelloState

class MonteCarloTreeSearchNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.reward = 0.0
        self.untried_actions = self.state.get_legal_actions()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def expand(self):
        if not self.untried_actions:
            return self  
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child = MonteCarloTreeSearchNode(next_state, parent=self, action=action)
        self.children.append(child)
        return child

    def best_child(self, c_param=1.4):
        # UCB1
        choices = [
            (child.reward / child.visits) +
            c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices.index(max(choices))]

    def rollout(self):
        current_state = self.state.clone()
        while not current_state.is_game_over():
            legal = current_state.get_legal_actions()
            action = random.choice(legal)  # random move or pass
            current_state = current_state.move(action)
        return current_state.game_result()

    def backpropagate(self, result):
        self.visits += 1
        self.reward += result
        if self.parent:
            self.parent.backpropagate(result)

    def best_action(self, simulations_number=100):
        
        if self.state.is_game_over():
            return None
        for _ in range(simulations_number):
            node = self
            #Badically la on fait les quatres steps de MCTS
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
            # Expansion
            node = node.expand()
            # Rollout
            result = node.rollout()
            # Backprop
            node.backpropagate(result)
        if not self.children:
            return None
        best_child = max(self.children, key=lambda c: c.visits)
        return best_child.action
