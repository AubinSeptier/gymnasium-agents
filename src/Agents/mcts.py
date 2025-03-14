import numpy as np
import random
import math
import copy
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Any
from Env.env import OthelloEnv, BLACK, WHITE, EMPTY, BOARD_SIZE

class MCTSNode:
    def __init__(self, env: OthelloEnv, parent=None, action=None):
        self.env = copy.deepcopy(env)  # Copie de l'environnement
        self.parent = parent  # Nœud parent
        self.action = action  # Action qui a mené à ce nœud
        self.children = []  # Nœuds enfants
        self.visits = 0  # Nombre de visites
        self.reward = 0  # Récompense cumulée
        self.untried_actions = self._get_untried_actions()  # Actions non explorées
        self.current_player = 1 if self.env.current_player == BLACK else -1  # Joueur actuel (1 pour BLACK, -1 pour WHITE)
    
    def _get_untried_actions(self) -> List[int]:
        """Retourne toutes les actions légales non essayées à partir de cet état."""
        valid_moves_array = self.env._get_observation()["valid_moves"]
        return [i for i, is_valid in enumerate(valid_moves_array) if is_valid == 1]
    
    def is_fully_expanded(self) -> bool:
        """Vérifie si tous les mouvements possibles depuis ce nœud ont été explorés."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Vérifie si cet état est terminal (fin de partie)."""
        obs = self.env._get_observation()
        return np.sum(obs["valid_moves"]) == 0  # Aucun mouvement valide
    
    def get_reward(self) -> float:
        """Retourne la récompense pour cet état terminal."""
        black_count, white_count = self.env._get_score()
        if black_count > white_count:
            return 1.0 if self.current_player == 1 else -1.0
        elif white_count > black_count:
            return -1.0 if self.current_player == 1 else 1.0
        else:
            return 0.0  # Match nul
    
    def best_child(self, c_param: float = 1.4) -> 'MCTSNode':
        """Sélectionne le meilleur enfant selon la formule UCB."""
        # Formule UCB: exploitation (Q/N) + exploration (c * sqrt(ln(N_parent) / N_child))
        choices_weights = [
            (child.reward / child.visits) + c_param * math.sqrt((2 * math.log(self.visits)) / child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]
    
    def expand(self) -> 'MCTSNode':
        """Ajoute un nouvel enfant au nœud en choisissant une action non explorée aléatoirement."""
        action = self.untried_actions.pop()
        
        # Copie l'environnement et applique l'action
        new_env = copy.deepcopy(self.env)
        _, _, _, _, _ = new_env.step(action)
        
        # Crée un nouveau nœud
        child_node = MCTSNode(new_env, parent=self, action=action)
        self.children.append(child_node)
        return child_node
    
    def simulate(self) -> float:
        """Simule une partie à partir de cet état jusqu'à un état terminal en choisissant des actions aléatoires."""
        sim_env = copy.deepcopy(self.env)
        terminated = False
        
        while not terminated:
            # Récupération des actions valides
            obs = sim_env._get_observation()
            valid_moves = [i for i, is_valid in enumerate(obs["valid_moves"]) if is_valid == 1]
            
            # Si aucun mouvement valide, la partie est terminée
            if not valid_moves:
                break
            
            # Choix d'une action aléatoire
            action = random.choice(valid_moves)
            
            # Exécution de l'action
            _, _, terminated, _, _ = sim_env.step(action)
        
        # Calcul de la récompense finale
        black_count, white_count = sim_env._get_score()
        if black_count > white_count:
            return 1.0 if self.current_player == 1 else -1.0
        elif white_count > black_count:
            return -1.0 if self.current_player == 1 else 1.0
        else:
            return 0.0  # Match nul
    
    def backpropagate(self, result: float) -> None:
        """Met à jour les statistiques de ce nœud et de ses parents avec le résultat de la simulation."""
        self.visits += 1
        self.reward += result
        
        # Propager aux nœuds parents
        if self.parent:
            self.parent.backpropagate(result)


class MCTSAgent:
    def __init__(self, num_simulations: int = 500, exploration_weight: float = 1.4):
        self.num_simulations = num_simulations  # Nombre de simulations par mouvement
        self.exploration_weight = exploration_weight  # Poids d'exploration (c dans UCB)
        self.name = f"MCTS_{num_simulations}"
    
    def choose_action(self, env: OthelloEnv) -> int:
        """Choisit la meilleure action selon MCTS."""
        # Créer le nœud racine avec l'état actuel
        root = MCTSNode(env)
        
        # Si le nœud racine est terminal (fin de partie), retourne une action aléatoire
        if root.is_terminal():
            valid_moves = [i for i, is_valid in enumerate(root.env._get_observation()["valid_moves"]) if is_valid == 1]
            if valid_moves:
                return random.choice(valid_moves)
            else:
                return 0  # Aucun mouvement valide
        
        # Exécuter les simulations MCTS
        for _ in range(self.num_simulations):
            node = root
            
            # Sélection
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.exploration_weight)
            
            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            # Simulation
            result = node.simulate()
            
            # Rétropropagation
            node.backpropagate(result)
        
        # Choisir le meilleur enfant selon les visites uniquement (pas l'exploration)
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action
    
    def train(self, env: OthelloEnv, num_episodes: int = 100) -> List[float]:
        """Entraîne l'agent sur un certain nombre d'épisodes et retourne les récompenses."""
        rewards = []
        
        for episode in tqdm(range(num_episodes), desc=f"Entraînement {self.name}"):
            # Réinitialiser l'environnement
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Choisir une action
                action = self.choose_action(env)
                
                # Exécuter l'action
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            rewards.append(episode_reward)
            
            # Afficher la progression
            if (episode + 1) % 10 == 0:
                print(f"Épisode {episode + 1}/{num_episodes}, Récompense moyenne: {np.mean(rewards[-10:]):.2f}")
        
        return rewards
    
    def save(self, filename: str) -> None:
        """Sauvegarde l'agent."""
        # Pour MCTS, nous sauvegardons simplement les paramètres
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({
                'num_simulations': self.num_simulations,
                'exploration_weight': self.exploration_weight
            }, f)
    
    def load(self, filename: str) -> None:
        """Charge l'agent."""
        import pickle
        with open(filename, 'rb') as f:
            params = pickle.load(f)
            self.num_simulations = params['num_simulations']
            self.exploration_weight = params['exploration_weight']