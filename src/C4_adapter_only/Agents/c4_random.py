import random

class ConnectFourRandomAgent:
    def __init__(self):
        self.name = "Connect4_Random"
    
    def choose_action(self, env):
        """Randomly chooses an action from valid moves."""
        obs = env._get_observation()
        valid_moves = [i for i, is_valid in enumerate(obs["valid_moves"]) if is_valid == 1]
        
        if not valid_moves:
            return 0
        
        return random.choice(valid_moves)