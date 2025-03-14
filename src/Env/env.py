import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, List, Dict, Any, Optional


# Constantes
BLACK = 1
WHITE = -1
EMPTY = 0
BOARD_SIZE = 8


class OthelloEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Définition de l'espace d'action
        # 64 positions possibles sur le plateau (8x8)
        self.action_space = spaces.Discrete(BOARD_SIZE * BOARD_SIZE)
        
        # Définition de l'espace d'observation
        # Le plateau de jeu (8x8) peut avoir 3 valeurs pour chaque case (-1, 0, 1)
        # Plus un indicateur pour le joueur actuel (1 ou -1)
        # Plateau: 8x8 = 64 positions avec 3 valeurs possibles par position
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=-1, high=1, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int8),
            "current_player": spaces.Discrete(2),  # 0 pour BLACK (1), 1 pour WHITE (-1)
            "valid_moves": spaces.Box(low=0, high=1, shape=(BOARD_SIZE * BOARD_SIZE,), dtype=np.int8)
        })
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Initialiser l'état
        self.board = None
        self.current_player = None
        self.valid_moves = None
        self.done = None
        self.selected_x = None
        self.selected_y = None
        self.cumulative_rewards = {"BLACK": 0, "WHITE": 0}
        
        # Reset pour initialiser les valeurs
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialiser le plateau avec les 4 pièces de départ
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE
        
        # Noir commence
        self.current_player = BLACK
        
        # Réinitialiser l'état terminé
        self.done = False
        
        # Calculer les mouvements valides
        self.valid_moves = self._get_valid_moves()
        
        # Position initiale du curseur de sélection (centré)
        self.selected_x = 3
        self.selected_y = 3
        
        # Réinitialiser les récompenses cumulatives
        self.cumulative_rewards = {"BLACK": 0, "WHITE": 0}
        
        # Retourner l'observation initiale et les informations
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        # Convertir l'action (0-63) en coordonnées (row, col)
        row, col = action // BOARD_SIZE, action % BOARD_SIZE
        reward = 0
        
        # Vérifier si l'action est valide
        if not self._is_valid_move(row, col):
            # Action invalide, terminer l'épisode avec récompense négative
            return self._get_observation(), -10.0, False, True, {"info": "Invalid move"}
        
        # Effectuer l'action (placer une pièce)
        self._place_piece(row, col)
        
        # Calculer la récompense (nombre de pièces capturées)
        black_count, white_count = self._get_score()
        if self.current_player == BLACK:
            reward = black_count - white_count
            self.cumulative_rewards["BLACK"] += reward
        else:
            reward = white_count - black_count
            self.cumulative_rewards["WHITE"] += reward
        
        # Passer au joueur suivant
        self.current_player = -self.current_player
        
        # Mettre à jour les mouvements valides
        self.valid_moves = self._get_valid_moves()
        
        # Si le joueur actuel n'a pas de mouvements valides, passer son tour
        if len(self.valid_moves) == 0:
            self.current_player = -self.current_player
            self.valid_moves = self._get_valid_moves()
            
            # Si après avoir passé le tour, il n'y a toujours pas de mouvements valides,
            # la partie est terminée
            if len(self.valid_moves) == 0:
                self.done = True
        
        # Vérifier si la partie est terminée
        terminated = self.done
        truncated = False  # Nous n'utilisons pas de limitation de nombre d'étapes
        
        # Mettre à jour la position du curseur de sélection pour qu'il soit sur un coup valide si possible
        if not terminated and self.valid_moves:
            self.selected_y, self.selected_x = self.valid_moves[0]
        
        # Information supplémentaire
        info = {
            "score_black": black_count,
            "score_white": white_count,
            "valid_moves_count": len(self.valid_moves),
            "cumulative_reward_black": self.cumulative_rewards["BLACK"],
            "cumulative_reward_white": self.cumulative_rewards["WHITE"]
        }
        
        if terminated:
            # Déterminer le gagnant à la fin de la partie
            if black_count > white_count:
                info["winner"] = "BLACK"
            elif white_count > black_count:
                info["winner"] = "WHITE"
            else:
                info["winner"] = "DRAW"
        
        return self._get_observation(), float(reward), terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Retourne l'observation actuelle."""
        # Convertir les mouvements valides en tableau binaire
        valid_moves_array = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.int8)
        for row, col in self.valid_moves:
            valid_moves_array[row * BOARD_SIZE + col] = 1
        
        return {
            "board": self.board.copy(),
            "current_player": 0 if self.current_player == BLACK else 1,
            "valid_moves": valid_moves_array
        }
    
    def move_selection(self, direction: str) -> None:
        """Déplace le curseur de sélection dans la direction spécifiée."""
        if direction == "up" and self.selected_y > 0:
            self.selected_y -= 1
        elif direction == "down" and self.selected_y < BOARD_SIZE - 1:
            self.selected_y += 1
        elif direction == "left" and self.selected_x > 0:
            self.selected_x -= 1
        elif direction == "right" and self.selected_x < BOARD_SIZE - 1:
            self.selected_x += 1
    
    def get_selected_position(self) -> int:
        """Retourne la position sélectionnée sous forme d'action (0-63)."""
        return self.selected_y * BOARD_SIZE + self.selected_x
    
    def is_selected_valid(self) -> bool:
        """Vérifie si la position sélectionnée correspond à un coup valide."""
        return (self.selected_y, self.selected_x) in self.valid_moves
    
    def _get_valid_moves(self) -> List[Tuple[int, int]]:
        """Retourne la liste des mouvements valides pour le joueur actuel."""
        valid_moves = []
        
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self._is_valid_move(row, col):
                    valid_moves.append((row, col))
        
        return valid_moves
    
    def _is_valid_move(self, row: int, col: int) -> bool:
        """Vérifie si un mouvement est valide."""
        # Si la case n'est pas vide, le mouvement n'est pas valide
        if self.board[row][col] != EMPTY:
            return False
        
        # Directions: horizontal, vertical et diagonale
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        # Pour chaque direction, vérifier si on peut capturer des pièces
        for dr, dc in directions:
            if self._can_capture(row, col, dr, dc):
                return True
        
        return False
    
    def _can_capture(self, row: int, col: int, dr: int, dc: int) -> bool:
        """Vérifie si on peut capturer des pièces dans une direction donnée."""
        opponent = -self.current_player
        
        # Vérifier s'il y a au moins une pièce adverse adjacente
        r, c = row + dr, col + dc
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == opponent):
            return False
        
        # Continuer dans cette direction pour trouver une pièce de sa couleur
        r, c = r + dr, c + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            if self.board[r][c] == EMPTY:
                return False
            if self.board[r][c] == self.current_player:
                return True
            r, c = r + dr, c + dc
        
        return False
    
    def _place_piece(self, row: int, col: int) -> None:
        """Place une pièce sur le plateau et capture les pièces adverses."""
        self.board[row][col] = self.current_player
        
        # Directions: horizontal, vertical et diagonale
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        # Pour chaque direction, capturer les pièces
        for dr, dc in directions:
            self._capture(row, col, dr, dc)
    
    def _capture(self, row: int, col: int, dr: int, dc: int) -> None:
        """Capture les pièces dans une direction donnée."""
        if not self._can_capture(row, col, dr, dc):
            return
        
        # Capturer les pièces adverses
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == -self.current_player:
            self.board[r][c] = self.current_player
            r, c = r + dr, c + dc
    
    def _get_score(self) -> Tuple[int, int]:
        """Retourne le score (nombre de pièces noires, nombre de pièces blanches)."""
        black_count = np.sum(self.board == BLACK)
        white_count = np.sum(self.board == WHITE)
        return black_count, white_count
    
    def render(self):
        """Méthode pour le rendu. L'implémentation est dans play_human.py."""
        pass
    
    def close(self):
        """Ferme l'environnement."""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()