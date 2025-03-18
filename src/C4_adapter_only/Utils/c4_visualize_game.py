import pygame
import time
from C4_adapter_only.Env.c4_env import ConnectFourEnv, BOARD_ROWS, BOARD_COLS, RED, YELLOW, EMPTY


def visualize_connect_four_game(agent1, agent2, delay=0.5):
    """Visualize a game of Connect Four between two agents."""
    pygame.init()
    
    # Constants for the graphical interface
    SQUARE_SIZE = 80
    BOARD_WIDTH = BOARD_COLS * SQUARE_SIZE
    BOARD_HEIGHT = BOARD_ROWS * SQUARE_SIZE
    INFO_PANEL_WIDTH = 300
    WINDOW_WIDTH = BOARD_WIDTH + INFO_PANEL_WIDTH
    WINDOW_HEIGHT = BOARD_HEIGHT + SQUARE_SIZE  # Extra space at top for move indicators
    BACKGROUND_COLOR = (0, 0, 139)  # Dark blue
    BOARD_COLOR = (0, 0, 200)       # Blue
    RED_COLOR = (255, 0, 0)         # Red
    YELLOW_COLOR = (255, 255, 0)    # Yellow
    INFO_PANEL_COLOR = (50, 50, 50) # Dark gray
    TEXT_COLOR = (255, 255, 255)    # White
    VALID_MOVE_COLOR = (0, 255, 0, 150)  # Semi-transparent green
    
    # Create the window
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"{agent1.name} vs {agent2.name}")
    
    # Initialize fonts
    title_font = pygame.font.SysFont("Arial", 30, bold=True)
    info_font = pygame.font.SysFont("Arial", 20)
    score_font = pygame.font.SysFont("Arial", 24, bold=True)
    
    # Create a surface for valid moves
    valid_move_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    pygame.draw.circle(valid_move_surface, (0, 255, 0, 100), (SQUARE_SIZE//2, SQUARE_SIZE//2), SQUARE_SIZE//2 - 5)
    
    # Clock to control the speed
    clock = pygame.time.Clock()
    
    # Initialize the environment
    env = ConnectFourEnv()
    obs, _ = env.reset()
    done = False
    red_reward = 0
    yellow_reward = 0
    actions_history = []
    
    def draw_board():
        """Draw the Connect Four game board."""
        # Background
        screen.fill(BACKGROUND_COLOR)
        
        # Draw indicator row above the board
        for col in range(BOARD_COLS):
            center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
            center_y = SQUARE_SIZE // 2
            
            if obs["valid_moves"][col] == 1:
                screen.blit(valid_move_surface, (col * SQUARE_SIZE, 0))
            
            col_num = info_font.render(str(col), True, TEXT_COLOR)
            screen.blit(col_num, (center_x - col_num.get_width()//2, center_y - col_num.get_height()//2))
        
        # Draw the board
        pygame.draw.rect(screen, BOARD_COLOR, (0, SQUARE_SIZE, BOARD_WIDTH, BOARD_HEIGHT))
        
        board = obs["board"]
        
        # Draw the grid and pieces
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
                center_y = (row + 1) * SQUARE_SIZE + SQUARE_SIZE // 2  # +1 for the indicator row
                
                # Draw empty circles (holes in the board)
                if board[row][col] == EMPTY:
                    pygame.draw.circle(screen, BACKGROUND_COLOR, (center_x, center_y), SQUARE_SIZE//2 - 5)
                elif board[row][col] == RED:
                    pygame.draw.circle(screen, RED_COLOR, (center_x, center_y), SQUARE_SIZE//2 - 5)
                elif board[row][col] == YELLOW:
                    pygame.draw.circle(screen, YELLOW_COLOR, (center_x, center_y), SQUARE_SIZE//2 - 5)
    
    def draw_info_panel():
        """Draw the information panel."""
        # Panel background
        pygame.draw.rect(screen, INFO_PANEL_COLOR, 
                         (BOARD_WIDTH, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT))
        
        title = title_font.render("CONNECT FOUR", True, TEXT_COLOR)
        screen.blit(title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - title.get_width() // 2, 20))
        
        agents_title = info_font.render(f"{agent1.name} (Red) vs {agent2.name} (Yellow)", True, TEXT_COLOR)
        screen.blit(agents_title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - agents_title.get_width() // 2, 60))
        
        red_count, yellow_count = env._get_score()
        
        score_title = info_font.render("PIECES", True, TEXT_COLOR)
        screen.blit(score_title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - score_title.get_width() // 2, 100))
        
        score_red = score_font.render(f"Red: {red_count}", True, RED_COLOR)
        screen.blit(score_red, (BOARD_WIDTH + 20, 130))
        
        score_yellow = score_font.render(f"Yellow: {yellow_count}", True, YELLOW_COLOR)
        screen.blit(score_yellow, (BOARD_WIDTH + 20, 160))
        
        current_player = "Red" if obs["current_player"] == 0 else "Yellow"
        current_agent = agent1.name if obs["current_player"] == 0 else agent2.name
        player_text = info_font.render(f"Turn: {current_player} ({current_agent})", True, TEXT_COLOR)
        screen.blit(player_text, (BOARD_WIDTH + 20, 200))
        
        reward_title = info_font.render("REWARDS", True, TEXT_COLOR)
        screen.blit(reward_title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - reward_title.get_width() // 2, 240))
        
        reward_red = info_font.render(f"Red: {red_reward:.1f}", True, TEXT_COLOR)
        screen.blit(reward_red, (BOARD_WIDTH + 20, 270))
        
        reward_yellow = info_font.render(f"Yellow: {yellow_reward:.1f}", True, TEXT_COLOR)
        screen.blit(reward_yellow, (BOARD_WIDTH + 20, 300))
        
        history_title = info_font.render("LAST ACTIONS", True, TEXT_COLOR)
        screen.blit(history_title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - history_title.get_width() // 2, 340))
        
        # Display the last 5 actions
        for i, (action, player) in enumerate(actions_history[-5:]):
            player_name = "Red" if player == 0 else "Yellow"
            action_text = info_font.render(f"{player_name}: Column {action}", True, TEXT_COLOR)
            screen.blit(action_text, (BOARD_WIDTH + 20, 370 + i * 25))
        
        # Instructions
        instructions = info_font.render("Press SPACE to advance", True, TEXT_COLOR)
        screen.blit(instructions, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - instructions.get_width() // 2, 490))
        
        quit_instr = info_font.render("or Q to quit", True, TEXT_COLOR)
        screen.blit(quit_instr, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - quit_instr.get_width() // 2, 520))
        
        # Game result (if game is over)
        if done:
            if "winner" in env.info:
                winner = env.info["winner"]
                if winner == "RED":
                    result_text = title_font.render("RED WINS!", True, RED_COLOR)
                elif winner == "YELLOW":
                    result_text = title_font.render("YELLOW WINS!", True, YELLOW_COLOR)
                else:
                    result_text = title_font.render("DRAW!", True, TEXT_COLOR)
                
                screen.blit(result_text, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - result_text.get_width() // 2, 560))
            else:
                # Determine winner based on piece count
                if red_count > yellow_count:
                    result_text = title_font.render("RED WINS!", True, RED_COLOR)
                elif yellow_count > red_count:
                    result_text = title_font.render("YELLOW WINS!", True, YELLOW_COLOR)
                else:
                    result_text = title_font.render("DRAW!", True, TEXT_COLOR)
                
                screen.blit(result_text, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - result_text.get_width() // 2, 560))
    
    running = True
    auto_play = False
    last_action_time = 0
    
    while running:
        current_time = time.time()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  # Quit
                    running = False
                elif event.key == pygame.K_SPACE:  # Advance manually
                    if not done:
                        auto_play = False
                        last_action_time = 0  # Force the next action
                elif event.key == pygame.K_a:  # Automatic mode
                    auto_play = not auto_play
        
        # Automatic mode or manual action
        if (auto_play and current_time - last_action_time > delay) or (not auto_play and last_action_time == 0):
            if not done:
                current_player = obs["current_player"]
                current_agent = agent1 if current_player == 0 else agent2
                
                action = current_agent.choose_action(env)
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                actions_history.append((action, current_player))
                
                if current_player == 0:  # RED
                    red_reward += reward
                else:  # YELLOW
                    yellow_reward += reward
                
                if terminated:
                    env.info = info
                
                obs = next_obs
                done = terminated or truncated
                
                last_action_time = current_time
            else:
                # Game over, wait before continuing
                if auto_play:
                    time.sleep(3)
                    running = False
        
        draw_board()
        draw_info_panel()
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()