import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Env.env import OthelloEnv, BLACK, WHITE, EMPTY, BOARD_SIZE

class DQNModel(nn.Module):
    def __init__(self, state_shape: Tuple[int, int] = (BOARD_SIZE, BOARD_SIZE), action_size: int = BOARD_SIZE * BOARD_SIZE):
        super(DQNModel, self).__init__()
        
        # Couches convolutives pour traiter le plateau
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Calcul de la dimension après convolution
        conv_output_size = 128 * state_shape[0] * state_shape[1]
        
        # Couches denses
        self.fc1 = nn.Linear(conv_output_size + action_size + 1, 256)  # +action_size pour les mouvements valides, +1 pour le joueur
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, board, valid_moves, player):
        # Traitement du plateau par CNN
        x = F.relu(self.bn1(self.conv1(board)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Aplatir
        
        # Concaténation avec les mouvements valides et le joueur actuel
        x = torch.cat([x, valid_moves, player], dim=1)
        
        # Couches denses
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        
        # Application du masque pour interdire les actions invalides
        masked_q_values = torch.where(
            valid_moves.bool(),
            q_values,
            torch.ones_like(q_values) * (-1e9)  # Valeur très basse pour les mouvements invalides
        )
        
        return masked_q_values


class DQNAgent:
    def __init__(
        self,
        state_shape: Tuple[int, int] = (BOARD_SIZE, BOARD_SIZE),
        action_size: int = BOARD_SIZE * BOARD_SIZE,
        memory_size: int = 10000,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        update_target_freq: int = 10,
    ):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # Facteur de réduction
        self.epsilon = epsilon  # Paramètre d'exploration
        self.epsilon_min = epsilon_min  # Epsilon minimal
        self.epsilon_decay = epsilon_decay  # Taux de décroissance d'epsilon
        self.learning_rate = learning_rate  # Taux d'apprentissage
        self.batch_size = batch_size  # Taille du batch
        self.update_target_freq = update_target_freq  # Fréquence de mise à jour du réseau cible
        
        # Vérifier si CUDA est disponible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Compteur d'étapes
        self.step_counter = 0
        
        # Modèles
        self.model = DQNModel(state_shape, action_size).to(self.device)
        self.target_model = DQNModel(state_shape, action_size).to(self.device)
        self.update_target_model()  # Synchroniser les deux modèles
        
        # Optimiseur
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Fonction de perte
        self.criterion = nn.MSELoss()
        
        # Nom pour l'identification
        self.name = "DQN"
    
    def update_target_model(self) -> None:
        """Met à jour le modèle cible avec les poids du modèle principal."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def preprocess_state(self, obs: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prétraite l'observation pour le réseau de neurones."""
        # Normaliser le plateau (-1, 0, 1) -> (-1, 0, 1)
        board = torch.tensor(obs["board"], dtype=torch.float32).reshape(1, 1, *self.state_shape).to(self.device)
        
        # Mouvements valides
        valid_moves = torch.tensor(obs["valid_moves"], dtype=torch.float32).reshape(1, -1).to(self.device)
        
        # Joueur actuel
        player = torch.tensor([1.0 if obs["current_player"] == 0 else -1.0], dtype=torch.float32).reshape(1, 1).to(self.device)
        
        return board, valid_moves, player
    
    def remember(self, state: Dict[str, np.ndarray], action: int, reward: float, 
                next_state: Dict[str, np.ndarray], done: bool) -> None:
        """Stocke l'expérience en mémoire."""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, env: OthelloEnv) -> int:
        """Choisit une action selon la politique epsilon-greedy."""
        obs = env._get_observation()
        
        # Mouvements valides
        valid_moves = obs["valid_moves"]
        valid_actions = [i for i, is_valid in enumerate(valid_moves) if is_valid == 1]
        
        # S'il n'y a pas de mouvements valides, retourne 0 (action par défaut)
        if not valid_actions:
            return 0
        
        # Exploration (epsilon-greedy)
        if np.random.rand() <= self.epsilon:
            return np.random.choice(valid_actions)
        
        # Exploitation
        state = self.preprocess_state(obs)
        
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(*state).cpu().numpy()[0]
        self.model.train()
        
        # Trouve l'action valide avec la plus grande Q-valeur
        return int(np.argmax(q_values))
    
    def replay(self) -> float:
        """Entraîne le modèle sur un batch d'expériences."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Échantillonner un batch de la mémoire
        minibatch = random.sample(self.memory, self.batch_size)
        
        total_loss = 0.0
        
        # Préparer les tenseurs par lots
        states_boards = []
        states_valid_moves = []
        states_players = []
        actions = []
        rewards = []
        next_states_boards = []
        next_states_valid_moves = []
        next_states_players = []
        dones = []
        
        for state, action, reward, next_state, done in minibatch:
            # Prétraiter les états
            board, valid_moves, player = self.preprocess_state(state)
            
            states_boards.append(board)
            states_valid_moves.append(valid_moves)
            states_players.append(player)
            actions.append(action)
            rewards.append(reward)
            
            # Prétraiter les états suivants
            next_board, next_valid_moves, next_player = self.preprocess_state(next_state)
            
            next_states_boards.append(next_board)
            next_states_valid_moves.append(next_valid_moves)
            next_states_players.append(next_player)
            dones.append(done)
        
        # Convertir en tenseurs batch
        states_boards = torch.cat(states_boards, dim=0)
        states_valid_moves = torch.cat(states_valid_moves, dim=0)
        states_players = torch.cat(states_players, dim=0)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_boards = torch.cat(next_states_boards, dim=0)
        next_states_valid_moves = torch.cat(next_states_valid_moves, dim=0)
        next_states_players = torch.cat(next_states_players, dim=0)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Prédire les Q-valeurs pour les états actuels
        self.model.train()
        current_q_values = self.model(states_boards, states_valid_moves, states_players)
        
        # Sélectionner les Q-valeurs pour les actions prises
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculer les Q-valeurs cibles
        with torch.no_grad():
            next_q_values = self.target_model(next_states_boards, next_states_valid_moves, next_states_players)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Calculer la perte
        loss = self.criterion(current_q_values, target_q_values)
        
        # Mettre à jour les poids
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Réduire epsilon (exploration)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Mettre à jour le modèle cible périodiquement
        self.step_counter += 1
        if self.step_counter % self.update_target_freq == 0:
            self.update_target_model()
        
        return loss.item()
    
    def train(self, env: OthelloEnv, num_episodes: int = 100) -> Tuple[List[float], List[float]]:
        """Entraîne l'agent sur un certain nombre d'épisodes et retourne les récompenses et les pertes."""
        rewards = []
        losses = []
        
        for episode in range(num_episodes):
            # Réinitialiser l'environnement
            state, _ = env.reset()
            total_reward = 0
            episode_losses = []
            done = False
            
            while not done:
                # Choisir une action
                action = self.choose_action(env)
                
                # Exécuter l'action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Stocker l'expérience
                self.remember(state, action, reward, next_state, done)
                
                # Apprentissage
                loss = self.replay()
                if loss > 0:
                    episode_losses.append(loss)
                
                # Mettre à jour l'état
                state = next_state
                total_reward += reward
            
            # Enregistrer les résultats
            rewards.append(total_reward)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            losses.append(avg_loss)
            
            # Afficher la progression
            if (episode + 1) % 10 == 0:
                print(f"Épisode {episode + 1}/{num_episodes}, Récompense: {total_reward:.2f}, "
                      f"Perte: {avg_loss:.4f}, Epsilon: {self.epsilon:.4f}")
        
        return rewards, losses
    
    def save(self, filename: str) -> None:
        """Sauvegarde le modèle et les paramètres de l'agent."""
        torch.save(self.model.state_dict(), filename + ".pt")
        
        # Sauvegarder les paramètres supplémentaires
        import pickle
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump({
                'epsilon': self.epsilon,
                'gamma': self.gamma,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'update_target_freq': self.update_target_freq
            }, f)
    
    def load(self, filename: str) -> None:
        """Charge le modèle et les paramètres de l'agent."""
        # Charger le modèle
        self.model.load_state_dict(torch.load(filename + ".pt"))
        self.target_model.load_state_dict(torch.load(filename + ".pt"))
        
        # Charger les paramètres
        import pickle
        with open(filename + ".pkl", 'rb') as f:
            params = pickle.load(f)
            self.epsilon = params['epsilon']
            self.gamma = params['gamma']
            self.epsilon_min = params['epsilon_min']
            self.epsilon_decay = params['epsilon_decay']
            self.learning_rate = params['learning_rate']
            self.batch_size = params['batch_size']
            self.update_target_freq = params['update_target_freq']