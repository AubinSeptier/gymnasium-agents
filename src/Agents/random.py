import random

class RandomAgent:
    def __init__(self):
        self.name = "Random"
    
    def choose_action(self, env):
        """Choisit aléatoirement une action parmi les coups valides."""
        # Récupérer les coups valides
        obs = env._get_observation()
        valid_moves = [i for i, is_valid in enumerate(obs["valid_moves"]) if is_valid == 1]
        
        # Si aucun coup valide, retourner une action par défaut
        if not valid_moves:
            return 0
        
        # Retourner un coup aléatoire
        return random.choice(valid_moves)