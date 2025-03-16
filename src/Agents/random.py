import random

class RandomAgent:
    def __init__(self):
        self.name = "Random"
    
    def choose_action(self, env):
        """Randomly chooses an action from valid moves."""
        # Get valid moves
        obs = env._get_observation()
        valid_moves = [i for i, is_valid in enumerate(obs["valid_moves"]) if is_valid == 1]
        
        # If no valid moves, return a default action
        if not valid_moves:
            return 0
        
        # Return a random move
        return random.choice(valid_moves)