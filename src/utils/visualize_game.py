import pygame
import time
from Env.env import OthelloEnv, BOARD_SIZE, BLACK, WHITE


def visualize_game(agent1, agent2, delay=0.5):
    """Visualise une partie entre deux agents."""
    # Configuration de pygame
    pygame.init()
    
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
    INFO_PANEL_COLOR = (50, 50, 50)  # Gris foncé
    TEXT_COLOR = (255, 255, 255)  # Blanc
    VALID_MOVE_COLOR = (0, 255, 0, 150)  # Vert semi-transparent
    
    # Création de la fenêtre
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"{agent1.name} vs {agent2.name}")
    
    # Initialisation des polices
    title_font = pygame.font.SysFont("Arial", 30, bold=True)
    info_font = pygame.font.SysFont("Arial", 20)
    score_font = pygame.font.SysFont("Arial", 24, bold=True)
    
    # Créer une surface pour les coups valides
    valid_move_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    valid_move_surface.fill((0, 255, 0, 100))  # Vert semi-transparent
    
    # Horloge pour contrôler la vitesse
    clock = pygame.time.Clock()
    
    # Initialiser l'environnement
    env = OthelloEnv()
    obs, _ = env.reset()
    done = False
    black_reward = 0
    white_reward = 0
    actions_history = []
    
    def draw_board():
        """Dessine le plateau de jeu."""
        # Fond du plateau
        screen.fill(BACKGROUND_COLOR, (0, 0, BOARD_WIDTH, WINDOW_HEIGHT))
        
        # Lignes du plateau
        for i in range(BOARD_SIZE + 1):
            pygame.draw.line(screen, LINE_COLOR, (i * SQUARE_SIZE, 0), 
                             (i * SQUARE_SIZE, BOARD_WIDTH), 2)
            pygame.draw.line(screen, LINE_COLOR, (0, i * SQUARE_SIZE), 
                             (BOARD_WIDTH, i * SQUARE_SIZE), 2)
        
        # Récupérer l'état actuel
        board = obs["board"]
        valid_moves_array = obs["valid_moves"]
        
        # Convertir le tableau valide_moves en liste de tuples
        valid_moves = []
        for i in range(len(valid_moves_array)):
            if valid_moves_array[i] == 1:
                row, col = i // BOARD_SIZE, i % BOARD_SIZE
                valid_moves.append((row, col))
        
        # Dessiner les mouvements valides en surbrillance
        for row, col in valid_moves:
            screen.blit(valid_move_surface, (col * SQUARE_SIZE, row * SQUARE_SIZE))
        
        # Dessiner les pièces
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
                center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2
                
                if board[row][col] == BLACK:
                    pygame.draw.circle(screen, BLACK_COLOR, (center_x, center_y), 
                                      SQUARE_SIZE // 2 - 5)
                elif board[row][col] == WHITE:
                    pygame.draw.circle(screen, WHITE_COLOR, (center_x, center_y), 
                                      SQUARE_SIZE // 2 - 5)
    
    def draw_info_panel():
        """Dessine le panneau d'informations."""
        # Fond du panneau
        pygame.draw.rect(screen, INFO_PANEL_COLOR, 
                         (BOARD_WIDTH, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT))
        
        # Titre
        title = title_font.render("OTHELLO", True, TEXT_COLOR)
        screen.blit(title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - title.get_width() // 2, 20))
        
        # Agents
        agents_title = info_font.render(f"{agent1.name} (Noir) vs {agent2.name} (Blanc)", True, TEXT_COLOR)
        screen.blit(agents_title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - agents_title.get_width() // 2, 60))
        
        # Score
        black_count, white_count = env._get_score()
        
        score_title = info_font.render("SCORE", True, TEXT_COLOR)
        screen.blit(score_title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - score_title.get_width() // 2, 100))
        
        score_black = score_font.render(f"Noir: {black_count}", True, TEXT_COLOR)
        screen.blit(score_black, (BOARD_WIDTH + 20, 130))
        
        score_white = score_font.render(f"Blanc: {white_count}", True, TEXT_COLOR)
        screen.blit(score_white, (BOARD_WIDTH + 20, 160))
        
        # Tour du joueur
        current_player = "Noir" if obs["current_player"] == 0 else "Blanc"
        current_agent = agent1.name if obs["current_player"] == 0 else agent2.name
        player_text = info_font.render(f"Tour: {current_player} ({current_agent})", True, TEXT_COLOR)
        screen.blit(player_text, (BOARD_WIDTH + 20, 200))
        
        # Récompenses cumulatives
        reward_title = info_font.render("RÉCOMPENSES", True, TEXT_COLOR)
        screen.blit(reward_title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - reward_title.get_width() // 2, 240))
        
        reward_black = info_font.render(f"Noir: {black_reward:.1f}", True, TEXT_COLOR)
        screen.blit(reward_black, (BOARD_WIDTH + 20, 270))
        
        reward_white = info_font.render(f"Blanc: {white_reward:.1f}", True, TEXT_COLOR)
        screen.blit(reward_white, (BOARD_WIDTH + 20, 300))
        
        # Historique des actions
        history_title = info_font.render("DERNIÈRES ACTIONS", True, TEXT_COLOR)
        screen.blit(history_title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - history_title.get_width() // 2, 340))
        
        # Afficher les 5 dernières actions
        for i, (action, player) in enumerate(actions_history[-5:]):
            row, col = action // BOARD_SIZE, action % BOARD_SIZE
            player_name = "Noir" if player == 0 else "Blanc"
            action_text = info_font.render(f"{player_name}: ({row}, {col})", True, TEXT_COLOR)
            screen.blit(action_text, (BOARD_WIDTH + 20, 370 + i * 25))
        
        # Instructions
        instructions = info_font.render("Appuyez sur ESPACE pour faire avancer", True, TEXT_COLOR)
        screen.blit(instructions, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - instructions.get_width() // 2, 490))
        
        quit_instr = info_font.render("ou Q pour quitter", True, TEXT_COLOR)
        screen.blit(quit_instr, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - quit_instr.get_width() // 2, 520))
    
    # Boucle principale de visualisation
    running = True
    auto_play = False
    last_action_time = 0
    
    while running:
        current_time = time.time()
        
        # Gérer les événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  # Quitter
                    running = False
                elif event.key == pygame.K_SPACE:  # Faire avancer manuellement
                    if not done:
                        auto_play = False
                        last_action_time = 0  # Forcer la prochaine action
                elif event.key == pygame.K_a:  # Mode automatique
                    auto_play = not auto_play
        
        # Mode automatique ou action manuelle
        if (auto_play and current_time - last_action_time > delay) or (not auto_play and last_action_time == 0):
            if not done:
                # Déterminer quel agent joue
                current_player = obs["current_player"]
                current_agent = agent1 if current_player == 0 else agent2
                
                # Choisir une action
                action = current_agent.choose_action(env)
                
                # Exécuter l'action
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Enregistrer l'action dans l'historique
                actions_history.append((action, current_player))
                
                # Mettre à jour les récompenses
                if current_player == 0:  # BLACK
                    black_reward += reward
                else:  # WHITE
                    white_reward += reward
                
                # Mettre à jour l'état
                obs = next_obs
                done = terminated or truncated
                
                # Mettre à jour le temps de la dernière action
                last_action_time = current_time
            else:
                # Partie terminée, afficher le résultat
                black_count, white_count = env._get_score()
                if black_count > white_count:
                    print(f"Noir ({agent1.name}) a gagné! {black_count}-{white_count}")
                elif white_count > black_count:
                    print(f"Blanc ({agent2.name}) a gagné! {white_count}-{black_count}")
                else:
                    print(f"Match nul! {black_count}-{white_count}")
                
                # Attendre un peu avant de fermer
                if auto_play:
                    time.sleep(3)
                    running = False
        
        # Dessiner le jeu
        draw_board()
        draw_info_panel()
        
        # Mettre à jour l'affichage
        pygame.display.flip()
        
        # Limiter la fréquence d'images
        clock.tick(60)
    
    # Fermer pygame
    pygame.quit()