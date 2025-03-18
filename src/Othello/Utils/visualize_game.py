import pygame
import time
from Othello.Env.env import OthelloEnv, BOARD_SIZE, BLACK, WHITE


def visualize_game(agent1, agent2, delay=0.5):
    """Visualize a game between two agents."""
    # Pygame configuration
    pygame.init()
    
    # Constants for the graphical interface
    SQUARE_SIZE = 60
    BOARD_WIDTH = BOARD_SIZE * SQUARE_SIZE
    INFO_PANEL_WIDTH = 300
    WINDOW_WIDTH = BOARD_WIDTH + INFO_PANEL_WIDTH
    WINDOW_HEIGHT = BOARD_WIDTH
    BACKGROUND_COLOR = (0, 120, 0)  # Dark green
    LINE_COLOR = (0, 0, 0)  # Black
    BLACK_COLOR = (0, 0, 0)  # Black
    WHITE_COLOR = (255, 255, 255)  # White
    INFO_PANEL_COLOR = (50, 50, 50)  # Dark gray
    TEXT_COLOR = (255, 255, 255)  # White
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
    valid_move_surface.fill((0, 255, 0, 100))  # Semi-transparent green
    
    # Clock to control the speed
    clock = pygame.time.Clock()
    
    # Initialize the environment
    env = OthelloEnv()
    obs, _ = env.reset()
    done = False
    black_reward = 0
    white_reward = 0
    actions_history = []
    
    def draw_board():
        """Draw the game board."""
        # Board background
        screen.fill(BACKGROUND_COLOR, (0, 0, BOARD_WIDTH, WINDOW_HEIGHT))
        
        # Board lines
        for i in range(BOARD_SIZE + 1):
            pygame.draw.line(screen, LINE_COLOR, (i * SQUARE_SIZE, 0), 
                             (i * SQUARE_SIZE, BOARD_WIDTH), 2)
            pygame.draw.line(screen, LINE_COLOR, (0, i * SQUARE_SIZE), 
                             (BOARD_WIDTH, i * SQUARE_SIZE), 2)
        
        # Get the current state
        board = obs["board"]
        valid_moves_array = obs["valid_moves"]
        
        # Convert the valid_moves array to a list of tuples
        valid_moves = []
        for i in range(len(valid_moves_array)):
            if valid_moves_array[i] == 1:
                row, col = i // BOARD_SIZE, i % BOARD_SIZE
                valid_moves.append((row, col))
        
        # Highlight valid moves
        for row, col in valid_moves:
            screen.blit(valid_move_surface, (col * SQUARE_SIZE, row * SQUARE_SIZE))
        
        # Draw the pieces
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
        """Draw the information panel."""
        # Panel background
        pygame.draw.rect(screen, INFO_PANEL_COLOR, 
                         (BOARD_WIDTH, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT))
        
        # Title
        title = title_font.render("OTHELLO", True, TEXT_COLOR)
        screen.blit(title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - title.get_width() // 2, 20))
        
        # Agents
        agents_title = info_font.render(f"{agent1.name} (Black) vs {agent2.name} (White)", True, TEXT_COLOR)
        screen.blit(agents_title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - agents_title.get_width() // 2, 60))
        
        # Score
        black_count, white_count = env._get_score()
        
        score_title = info_font.render("SCORE", True, TEXT_COLOR)
        screen.blit(score_title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - score_title.get_width() // 2, 100))
        
        score_black = score_font.render(f"Black: {black_count}", True, TEXT_COLOR)
        screen.blit(score_black, (BOARD_WIDTH + 20, 130))
        
        score_white = score_font.render(f"White: {white_count}", True, TEXT_COLOR)
        screen.blit(score_white, (BOARD_WIDTH + 20, 160))
        
        # Current player's turn
        current_player = "Black" if obs["current_player"] == 0 else "White"
        current_agent = agent1.name if obs["current_player"] == 0 else agent2.name
        player_text = info_font.render(f"Turn: {current_player} ({current_agent})", True, TEXT_COLOR)
        screen.blit(player_text, (BOARD_WIDTH + 20, 200))
        
        # Cumulative rewards
        reward_title = info_font.render("REWARDS", True, TEXT_COLOR)
        screen.blit(reward_title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - reward_title.get_width() // 2, 240))
        
        reward_black = info_font.render(f"Black: {black_reward:.1f}", True, TEXT_COLOR)
        screen.blit(reward_black, (BOARD_WIDTH + 20, 270))
        
        reward_white = info_font.render(f"White: {white_reward:.1f}", True, TEXT_COLOR)
        screen.blit(reward_white, (BOARD_WIDTH + 20, 300))
        
        # Action history
        history_title = info_font.render("LAST ACTIONS", True, TEXT_COLOR)
        screen.blit(history_title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - history_title.get_width() // 2, 340))
        
        # Display the last 5 actions
        for i, (action, player) in enumerate(actions_history[-5:]):
            row, col = action // BOARD_SIZE, action % BOARD_SIZE
            player_name = "Black" if player == 0 else "White"
            action_text = info_font.render(f"{player_name}: ({row}, {col})", True, TEXT_COLOR)
            screen.blit(action_text, (BOARD_WIDTH + 20, 370 + i * 25))
        
        # Instructions
        instructions = info_font.render("Press SPACE to advance", True, TEXT_COLOR)
        screen.blit(instructions, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - instructions.get_width() // 2, 490))
        
        quit_instr = info_font.render("or Q to quit", True, TEXT_COLOR)
        screen.blit(quit_instr, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - quit_instr.get_width() // 2, 520))
    
    # Main visualization loop
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
                # Determine which agent plays
                current_player = obs["current_player"]
                current_agent = agent1 if current_player == 0 else agent2
                
                # Choose an action
                action = current_agent.choose_action(env)
                
                # Execute the action
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Record the action in the history
                actions_history.append((action, current_player))
                
                # Update rewards
                if current_player == 0:  # BLACK
                    black_reward += reward
                else:  # WHITE
                    white_reward += reward
                
                # Update the state
                obs = next_obs
                done = terminated or truncated
                
                # Update the time of the last action
                last_action_time = current_time
            else:
                # Game over, display the result
                black_count, white_count = env._get_score()
                if black_count > white_count:
                    print(f"Black ({agent1.name}) won! {black_count}-{white_count}")
                elif white_count > black_count:
                    print(f"White ({agent2.name}) won! {white_count}-{black_count}")
                else:
                    print(f"Draw! {black_count}-{white_count}")
                
                # Wait a bit before closing
                if auto_play:
                    time.sleep(3)
                    running = False
        
        # Draw the game
        draw_board()
        draw_info_panel()
        
        # Update the display
        pygame.display.flip()
        
        # Limit the frame rate
        clock.tick(60)
    
    # Quit pygame
    pygame.quit()