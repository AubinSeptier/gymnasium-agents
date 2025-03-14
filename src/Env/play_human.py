import pygame
import sys
import numpy as np
from Env.env import OthelloEnv, BLACK, WHITE, EMPTY, BOARD_SIZE

# Constantes pour l'interface graphique
SQUARE_SIZE = 60
BOARD_WIDTH = BOARD_SIZE * SQUARE_SIZE
INFO_PANEL_WIDTH = 300
WINDOW_WIDTH = BOARD_WIDTH + INFO_PANEL_WIDTH
WINDOW_HEIGHT = BOARD_WIDTH
BACKGROUND_COLOR = (0, 120, 0)  # Vert foncé
LINE_COLOR = (0, 0, 0)  # Noir
BLACK_COLOR = (0, 0, 0)  # Noir
WHITE_COLOR = (255, 255, 255)  # Blanc
HIGHLIGHT_COLOR = (255, 255, 0)  # Jaune
INFO_PANEL_COLOR = (50, 50, 50)  # Gris foncé
TEXT_COLOR = (255, 255, 255)  # Blanc
HIGHLIGHT_ALPHA = 100  # Transparence de la surbrillance
VALID_MOVE_COLOR = (0, 255, 0, 150)  # Vert semi-transparent

class OthelloGame:
    def __init__(self):
        # Initialiser Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Othello - Human Play")
        
        # Créer l'environnement
        self.env = OthelloEnv(render_mode="human")
        
        # Initialiser les polices
        self.title_font = pygame.font.SysFont("Arial", 30, bold=True)
        self.info_font = pygame.font.SysFont("Arial", 20)
        self.score_font = pygame.font.SysFont("Arial", 24, bold=True)
        
        # Gestion du temps
        self.clock = pygame.time.Clock()
        
        # Créer une surface semi-transparente pour la surbrillance
        self.highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        self.highlight_surface.fill((HIGHLIGHT_COLOR[0], HIGHLIGHT_COLOR[1], HIGHLIGHT_COLOR[2], HIGHLIGHT_ALPHA))
        
        # Créer une surface pour les coups valides
        self.valid_move_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        self.valid_move_surface.fill((VALID_MOVE_COLOR[0], VALID_MOVE_COLOR[1], VALID_MOVE_COLOR[2], VALID_MOVE_COLOR[3]))
        
        # Dernière récompense obtenue
        self.last_reward = 0
        self.game_over = False
        self.winner = None
        
        # Réinitialiser l'environnement
        self.obs, _ = self.env.reset()
    
    def draw_board(self):
        """Dessine le plateau de jeu."""
        # Fond du plateau
        self.screen.fill(BACKGROUND_COLOR, (0, 0, BOARD_WIDTH, WINDOW_HEIGHT))
        
        # Lignes du plateau
        for i in range(BOARD_SIZE + 1):
            pygame.draw.line(self.screen, LINE_COLOR, (i * SQUARE_SIZE, 0), 
                             (i * SQUARE_SIZE, BOARD_WIDTH), 2)
            pygame.draw.line(self.screen, LINE_COLOR, (0, i * SQUARE_SIZE), 
                             (BOARD_WIDTH, i * SQUARE_SIZE), 2)
        
        # Récupérer l'état actuel
        board = self.obs["board"]
        valid_moves_array = self.obs["valid_moves"]
        
        # Convertir le tableau valide_moves en liste de tuples
        valid_moves = []
        for i in range(len(valid_moves_array)):
            if valid_moves_array[i] == 1:
                row, col = i // BOARD_SIZE, i % BOARD_SIZE
                valid_moves.append((row, col))
        
        # Dessiner les mouvements valides en surbrillance
        for row, col in valid_moves:
            self.screen.blit(self.valid_move_surface, (col * SQUARE_SIZE, row * SQUARE_SIZE))
        
        # Dessiner les pièces
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
                center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2
                
                if board[row][col] == BLACK:
                    pygame.draw.circle(self.screen, BLACK_COLOR, (center_x, center_y), 
                                      SQUARE_SIZE // 2 - 5)
                elif board[row][col] == WHITE:
                    pygame.draw.circle(self.screen, WHITE_COLOR, (center_x, center_y), 
                                      SQUARE_SIZE // 2 - 5)
        
        # Dessiner la sélection
        selected_x, selected_y = self.env.selected_x, self.env.selected_y
        self.screen.blit(self.highlight_surface, (selected_x * SQUARE_SIZE, selected_y * SQUARE_SIZE))
    
    def draw_info_panel(self):
        """Dessine le panneau d'informations."""
        # Fond du panneau
        pygame.draw.rect(self.screen, INFO_PANEL_COLOR, 
                         (BOARD_WIDTH, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT))
        
        # Titre
        title = self.title_font.render("OTHELLO", True, TEXT_COLOR)
        self.screen.blit(title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - title.get_width() // 2, 20))
        
        # Score
        black_count, white_count = self.env._get_score()
        
        score_title = self.info_font.render("SCORE", True, TEXT_COLOR)
        self.screen.blit(score_title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - score_title.get_width() // 2, 80))
        
        score_black = self.score_font.render(f"Noir: {black_count}", True, TEXT_COLOR)
        self.screen.blit(score_black, (BOARD_WIDTH + 20, 120))
        
        score_white = self.score_font.render(f"Blanc: {white_count}", True, TEXT_COLOR)
        self.screen.blit(score_white, (BOARD_WIDTH + 20, 150))
        
        # Tour du joueur
        current_player = "Noir" if self.obs["current_player"] == 0 else "Blanc"
        player_text = self.info_font.render(f"Tour: {current_player}", True, TEXT_COLOR)
        self.screen.blit(player_text, (BOARD_WIDTH + 20, 200))
        
        # Dernière récompense
        reward_text = self.info_font.render(f"Dernière récompense: {self.last_reward}", True, TEXT_COLOR)
        self.screen.blit(reward_text, (BOARD_WIDTH + 20, 230))
        
        # Récompense cumulative
        cumul_black = self.info_font.render(f"Cumul Noir: {self.env.cumulative_rewards['BLACK']:.1f}", True, TEXT_COLOR)
        self.screen.blit(cumul_black, (BOARD_WIDTH + 20, 260))
        
        cumul_white = self.info_font.render(f"Cumul Blanc: {self.env.cumulative_rewards['WHITE']:.1f}", True, TEXT_COLOR)
        self.screen.blit(cumul_white, (BOARD_WIDTH + 20, 290))
        
        # Commandes
        commands_title = self.info_font.render("COMMANDES", True, TEXT_COLOR)
        self.screen.blit(commands_title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - commands_title.get_width() // 2, 350))
        
        commands = [
            "Flèches: Déplacer sélection",
            "Espace: Placer pièce",
            "R: Réinitialiser",
            "Q: Quitter"
        ]
        
        for i, cmd in enumerate(commands):
            cmd_text = self.info_font.render(cmd, True, TEXT_COLOR)
            self.screen.blit(cmd_text, (BOARD_WIDTH + 20, 390 + i * 30))
        
        # Afficher le gagnant si la partie est terminée
        if self.game_over:
            if self.winner == "BLACK":
                winner_text = "Noir a gagné!"
            elif self.winner == "WHITE":
                winner_text = "Blanc a gagné!"
            else:
                winner_text = "Match nul!"
            
            winner_surface = self.title_font.render(winner_text, True, TEXT_COLOR)
            self.screen.blit(winner_surface, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - winner_surface.get_width() // 2, 500))
    
    def run(self):
        """Boucle principale du jeu."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:  # Quitter
                        running = False
                    
                    elif event.key == pygame.K_r:  # Réinitialiser
                        self.obs, _ = self.env.reset()
                        self.last_reward = 0
                        self.game_over = False
                        self.winner = None
                    
                    elif not self.game_over:  # Actions de jeu uniquement si la partie n'est pas terminée
                        if event.key == pygame.K_UP:
                            self.env.move_selection("up")
                        
                        elif event.key == pygame.K_DOWN:
                            self.env.move_selection("down")
                        
                        elif event.key == pygame.K_LEFT:
                            self.env.move_selection("left")
                        
                        elif event.key == pygame.K_RIGHT:
                            self.env.move_selection("right")
                        
                        elif event.key == pygame.K_SPACE:
                            # Vérifier si la sélection est valide
                            if self.env.is_selected_valid():
                                # Jouer le coup
                                action = self.env.get_selected_position()
                                self.obs, reward, terminated, truncated, info = self.env.step(action)
                                self.last_reward = reward
                                
                                # Vérifier si la partie est terminée
                                if terminated:
                                    self.game_over = True
                                    self.winner = info.get("winner", None)
            
            # Dessiner le jeu
            self.draw_board()
            self.draw_info_panel()
            
            # Mettre à jour l'affichage
            pygame.display.flip()
            
            # Limiter la fréquence d'images
            self.clock.tick(60)
        
        # Fermer Pygame
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = OthelloGame()
    game.run()