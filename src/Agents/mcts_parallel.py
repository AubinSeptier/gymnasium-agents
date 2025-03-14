import numpy as np
import random
import math
import copy
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Any, Union
from Env.env import OthelloEnv, BLACK, WHITE, EMPTY, BOARD_SIZE


class MCTSNodeParallel:
    """Classe représentant un nœud dans l'arbre MCTS, optimisée pour la parallélisation."""
    
    __slots__ = ('env', 'parent', 'action', 'children', 'visits', 'reward', 
                'untried_actions', 'current_player', 'terminal', 'fully_expanded')
    
    def __init__(self, env: OthelloEnv, parent=None, action=None):
        self.env = env  # État de l'environnement (sans copie profonde ici pour l'efficacité)
        self.parent = parent  # Nœud parent
        self.action = action  # Action qui a mené à ce nœud
        self.children = []  # Nœuds enfants
        self.visits = 0  # Nombre de visites
        self.reward = 0.0  # Récompense cumulée
        self.untried_actions = None  # Actions non explorées (calculées à la demande)
        self.current_player = 1 if env.current_player == BLACK else -1  # Joueur actuel
        self.terminal = None  # Indicateur d'état terminal (calculé à la demande)
        self.fully_expanded = False  # Indicateur d'expansion complète
    
    def _get_untried_actions(self) -> List[int]:
        """Retourne toutes les actions légales non essayées à partir de cet état."""
        if self.untried_actions is None:
            valid_moves_array = self.env._get_observation()["valid_moves"]
            self.untried_actions = [i for i, is_valid in enumerate(valid_moves_array) if is_valid == 1]
        return self.untried_actions.copy()  # Copie pour éviter des modifications externes
    
    def is_fully_expanded(self) -> bool:
        """Vérifie si tous les mouvements possibles depuis ce nœud ont été explorés."""
        if not self.fully_expanded:
            self.fully_expanded = len(self._get_untried_actions()) == 0
        return self.fully_expanded
    
    def is_terminal(self) -> bool:
        """Vérifie si cet état est terminal (fin de partie)."""
        if self.terminal is None:
            obs = self.env._get_observation()
            self.terminal = np.sum(obs["valid_moves"]) == 0  # Aucun mouvement valide
        return self.terminal
    
    def get_reward(self) -> float:
        """Retourne la récompense pour cet état terminal."""
        black_count, white_count = self.env._get_score()
        if black_count > white_count:
            return 1.0 if self.current_player == 1 else -1.0
        elif white_count > black_count:
            return -1.0 if self.current_player == 1 else 1.0
        else:
            return 0.0  # Match nul
    
    def best_child(self, c_param: float = 1.4) -> 'MCTSNodeParallel':
        """Sélectionne le meilleur enfant selon la formule UCB."""
        # Optimisation : préallouer le tableau pour les poids
        choices_weights = np.zeros(len(self.children))
        
        # Vectoriser le calcul UCB pour plus d'efficacité
        for i, child in enumerate(self.children):
            # Formule UCB: exploitation (Q/N) + exploration (c * sqrt(ln(N_parent) / N_child))
            exploitation = child.reward / child.visits
            exploration = c_param * math.sqrt((2 * math.log(self.visits)) / child.visits)
            choices_weights[i] = exploitation + exploration
        
        return self.children[np.argmax(choices_weights)]
    
    def expand(self) -> 'MCTSNodeParallel':
        """Ajoute un nouvel enfant au nœud en choisissant une action non explorée aléatoirement."""
        untried_actions = self._get_untried_actions()
        if not untried_actions:
            return self  # Aucune action à explorer
            
        action = untried_actions.pop()
        self.untried_actions = untried_actions  # Mettre à jour les actions non essayées
        
        # Copie l'environnement et applique l'action
        new_env = copy.deepcopy(self.env)
        _, _, _, _, _ = new_env.step(action)
        
        # Crée un nouveau nœud
        child_node = MCTSNodeParallel(new_env, parent=self, action=action)
        self.children.append(child_node)
        
        # Vérifier si toutes les actions ont été explorées
        if not self.untried_actions:
            self.fully_expanded = True
            
        return child_node
    
    def simulate(self) -> float:
        """Simule une partie à partir de cet état jusqu'à un état terminal en choisissant des actions aléatoires."""
        # Copie locale de l'environnement pour la simulation
        sim_env = copy.deepcopy(self.env)
        terminated = False
        
        # Limiter le nombre de pas pour éviter les boucles infinies
        max_steps = 60  # Limite arbitraire
        step_count = 0
        
        while not terminated and step_count < max_steps:
            # Récupération des actions valides de manière efficace
            obs = sim_env._get_observation()
            valid_moves_indices = np.where(obs["valid_moves"] == 1)[0]
            
            # Si aucun mouvement valide, la partie est terminée
            if len(valid_moves_indices) == 0:
                break
            
            # Choix d'une action aléatoire (plus rapide avec numpy)
            action = np.random.choice(valid_moves_indices)
            
            # Exécution de l'action
            _, _, terminated, _, _ = sim_env.step(action)
            step_count += 1
        
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
        # Utiliser une boucle itérative au lieu de la récursion pour éviter les erreurs de pile
        node = self
        while node is not None:
            node.visits += 1
            node.reward += result
            node = node.parent


def run_simulation(root_state, exploration_weight=1.4):
    """Fonction pour exécuter une simulation MCTS complète (sélection, expansion, simulation, backpropagation)."""
    # Créer une copie de l'état racine
    env_copy = copy.deepcopy(root_state)
    node = MCTSNodeParallel(env_copy)
    
    # Sélection
    while not node.is_terminal() and node.is_fully_expanded():
        node = node.best_child(exploration_weight)
    
    # Expansion
    if not node.is_terminal() and not node.is_fully_expanded():
        node = node.expand()
    
    # Simulation
    result = node.simulate()
    
    # Backpropagation
    node.backpropagate(result)
    
    # Retourner le résultat et l'action
    return result, node.action


# Classe de résultats agrégés pour le parallélisme
class SimulationResults:
    """Classe pour agréger les résultats des simulations parallèles."""
    
    def __init__(self):
        self.action_visits = {}  # {action: nombre de visites}
        self.action_rewards = {}  # {action: récompense totale}
    
    def add_result(self, action, reward):
        """Ajoute un résultat de simulation."""
        if action in self.action_visits:
            self.action_visits[action] += 1
            self.action_rewards[action] += reward
        else:
            self.action_visits[action] = 1
            self.action_rewards[action] = reward
    
    def merge(self, other):
        """Fusionne avec un autre objet SimulationResults."""
        for action, visits in other.action_visits.items():
            if action in self.action_visits:
                self.action_visits[action] += visits
                self.action_rewards[action] += other.action_rewards[action]
            else:
                self.action_visits[action] = visits
                self.action_rewards[action] = other.action_rewards[action]
    
    def best_action(self):
        """Retourne l'action la plus visitée."""
        if not self.action_visits:
            return None
        return max(self.action_visits.items(), key=lambda x: x[1])[0]


def run_batch_simulations(env, batch_size, exploration_weight=1.4):
    """Exécute un lot de simulations et retourne les résultats agrégés."""
    results = SimulationResults()
    
    for _ in range(batch_size):
        reward, action = run_simulation(env, exploration_weight)
        results.add_result(action, reward)
    
    return results


class MCTSAgentParallel:
    """Agent MCTS avec parallélisation pour améliorer les performances."""
    
    def __init__(self, num_simulations: int = 500, exploration_weight: float = 1.4, 
                 num_processes: int = None, batch_size: int = None):
        self.num_simulations = num_simulations  # Nombre total de simulations
        self.exploration_weight = exploration_weight  # Poids d'exploration
        
        # Déterminer le nombre de processus à utiliser
        if num_processes is None:
            self.num_processes = max(1, multiprocessing.cpu_count() - 1)  # Laisser un cœur libre
        else:
            self.num_processes = num_processes
        
        # Déterminer la taille du lot par processus
        if batch_size is None:
            # Diviser les simulations également entre les processus
            self.batch_size = max(1, self.num_simulations // self.num_processes)
        else:
            self.batch_size = batch_size
        
        # Ajuster le nombre total de simulations pour qu'il soit divisible
        self.actual_num_simulations = self.batch_size * self.num_processes
        
        self.name = f"MCTS_Parallel_{self.actual_num_simulations}"
        self.pool = None  # ProcessPoolExecutor sera initialisé à la demande
    
    def _initialize_pool(self):
        """Initialise le pool de processus si nécessaire."""
        if self.pool is None:
            self.pool = ProcessPoolExecutor(max_workers=self.num_processes)
    
    def choose_action(self, env: OthelloEnv) -> int:
        """Choisit la meilleure action selon MCTS parallélisé."""
        # Vérifier si des actions valides existent
        obs = env._get_observation()
        valid_moves = [i for i, is_valid in enumerate(obs["valid_moves"]) if is_valid == 1]
        
        # Si aucun mouvement valide, retourner une action par défaut
        if not valid_moves:
            return 0
        
        # Si un seul mouvement valide, le retourner immédiatement
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Initialiser le pool de processus si nécessaire
        self._initialize_pool()
        
        # Exécuter les simulations en parallèle
        results = SimulationResults()
        batch_futures = []
        
        try:
            # Soumettre les tâches
            for _ in range(self.num_processes):
                future = self.pool.submit(run_batch_simulations, env, self.batch_size, self.exploration_weight)
                batch_futures.append(future)
            
            # Récupérer les résultats
            for future in as_completed(batch_futures):
                batch_results = future.result()
                results.merge(batch_results)
        
        except Exception as e:
            print(f"Exception pendant le calcul MCTS parallèle: {e}")
            # Fallback à une action valide aléatoire en cas d'erreur
            return random.choice(valid_moves)
        
        # Retourner l'action la plus visitée
        best_action = results.best_action()
        return best_action if best_action is not None else random.choice(valid_moves)
    
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
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({
                'num_simulations': self.num_simulations,
                'exploration_weight': self.exploration_weight,
                'num_processes': self.num_processes,
                'batch_size': self.batch_size
            }, f)
    
    def load(self, filename: str) -> None:
        """Charge l'agent."""
        import pickle
        with open(filename, 'rb') as f:
            params = pickle.load(f)
            self.num_simulations = params['num_simulations']
            self.exploration_weight = params['exploration_weight']
            self.num_processes = params.get('num_processes', multiprocessing.cpu_count() - 1)
            self.batch_size = params.get('batch_size', max(1, self.num_simulations // self.num_processes))
            self.actual_num_simulations = self.batch_size * self.num_processes
    
    def __del__(self):
        """Destructeur pour nettoyer les ressources."""
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None


# Fonction utilitaire pour mesurer les performances
def benchmark_mcts_agent(agent, env, num_actions=10):
    """Mesure le temps moyen pour choisir une action."""
    total_time = 0
    
    for _ in range(num_actions):
        # Réinitialiser l'environnement
        obs, _ = env.reset()
        
        # Mesurer le temps pour choisir une action
        start_time = time.time()
        _ = agent.choose_action(env)
        end_time = time.time()
        
        total_time += (end_time - start_time)
    
    return total_time / num_actions


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer l'environnement
    env = OthelloEnv()
    
    # Créer les agents
    mcts_agent = MCTSAgentParallel(num_simulations=500, exploration_weight=1.4)
    
    # Benchmark
    print(f"Benchmarking {mcts_agent.name}...")
    avg_time = benchmark_mcts_agent(mcts_agent, env, num_actions=5)
    print(f"Temps moyen par action: {avg_time:.4f} secondes")
    
    # Nettoyer
    del mcts_agent