import pygame
import sys
import numpy as np
from Othello.Env.env import OthelloEnv, BLACK, WHITE, EMPTY, BOARD_SIZE

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
HIGHLIGHT_COLOR = (255, 255, 0)  # Yellow
INFO_PANEL_COLOR = (50, 50, 50)  # Dark gray
TEXT_COLOR = (255, 255, 255)  # White
HIGHLIGHT_ALPHA = 100  # Highlight transparency
VALID_MOVE_COLOR = (0, 255, 0, 150)  # Semi-transparent green

class OthelloGame:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Othello - Human Play")
        
        # Create the environment
        self.env = OthelloEnv(render_mode="human")
        
        # Initialize fonts
        self.title_font = pygame.font.SysFont("Arial", 30, bold=True)
        self.info_font = pygame.font.SysFont("Arial", 20)
        self.score_font = pygame.font.SysFont("Arial", 24, bold=True)
        
        # Time management
        self.clock = pygame.time.Clock()
        
        # Create a semi-transparent surface for highlighting
        self.highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        self.highlight_surface.fill((HIGHLIGHT_COLOR[0], HIGHLIGHT_COLOR[1], HIGHLIGHT_COLOR[2], HIGHLIGHT_ALPHA))
        
        # Create a surface for valid moves
        self.valid_move_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        self.valid_move_surface.fill((VALID_MOVE_COLOR[0], VALID_MOVE_COLOR[1], VALID_MOVE_COLOR[2], VALID_MOVE_COLOR[3]))
        
        # Last reward obtained
        self.last_reward = 0
        self.game_over = False
        self.winner = None
        
        # Reset the environment
        self.obs, _ = self.env.reset()
    
    def draw_board(self):
        """Draws the game board."""
        # Board background
        self.screen.fill(BACKGROUND_COLOR, (0, 0, BOARD_WIDTH, WINDOW_HEIGHT))
        
        # Board lines
        for i in range(BOARD_SIZE + 1):
            pygame.draw.line(self.screen, LINE_COLOR, (i * SQUARE_SIZE, 0), 
                             (i * SQUARE_SIZE, BOARD_WIDTH), 2)
            pygame.draw.line(self.screen, LINE_COLOR, (0, i * SQUARE_SIZE), 
                             (BOARD_WIDTH, i * SQUARE_SIZE), 2)
        
        # Get the current state
        board = self.obs["board"]
        valid_moves_array = self.obs["valid_moves"]
        
        # Convert the valid_moves array to a list of tuples
        valid_moves = []
        for i in range(len(valid_moves_array)):
            if valid_moves_array[i] == 1:
                row, col = i // BOARD_SIZE, i % BOARD_SIZE
                valid_moves.append((row, col))
        
        # Draw valid moves with highlight
        for row, col in valid_moves:
            self.screen.blit(self.valid_move_surface, (col * SQUARE_SIZE, row * SQUARE_SIZE))
        
        # Draw pieces
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
        
        # Draw selection
        selected_x, selected_y = self.env.selected_x, self.env.selected_y
        self.screen.blit(self.highlight_surface, (selected_x * SQUARE_SIZE, selected_y * SQUARE_SIZE))
    
    def draw_info_panel(self):
        """Draws the information panel."""
        # Panel background
        pygame.draw.rect(self.screen, INFO_PANEL_COLOR, 
                         (BOARD_WIDTH, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT))
        
        # Title
        title = self.title_font.render("OTHELLO", True, TEXT_COLOR)
        self.screen.blit(title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - title.get_width() // 2, 20))
        
        # Score
        black_count, white_count = self.env._get_score()
        
        score_title = self.info_font.render("SCORE", True, TEXT_COLOR)
        self.screen.blit(score_title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - score_title.get_width() // 2, 80))
        
        score_black = self.score_font.render(f"Black: {black_count}", True, TEXT_COLOR)
        self.screen.blit(score_black, (BOARD_WIDTH + 20, 120))
        
        score_white = self.score_font.render(f"White: {white_count}", True, TEXT_COLOR)
        self.screen.blit(score_white, (BOARD_WIDTH + 20, 150))
        
        # Current player's turn
        current_player = "Black" if self.obs["current_player"] == 0 else "White"
        player_text = self.info_font.render(f"Turn: {current_player}", True, TEXT_COLOR)
        self.screen.blit(player_text, (BOARD_WIDTH + 20, 200))
        
        # Last reward
        reward_text = self.info_font.render(f"Last reward: {self.last_reward}", True, TEXT_COLOR)
        self.screen.blit(reward_text, (BOARD_WIDTH + 20, 230))
        
        # Cumulative rewards
        cumul_black = self.info_font.render(f"Cumulative Black: {self.env.cumulative_rewards['BLACK']:.1f}", True, TEXT_COLOR)
        self.screen.blit(cumul_black, (BOARD_WIDTH + 20, 260))
        
        cumul_white = self.info_font.render(f"Cumulative White: {self.env.cumulative_rewards['WHITE']:.1f}", True, TEXT_COLOR)
        self.screen.blit(cumul_white, (BOARD_WIDTH + 20, 290))
        
        # Commands
        commands_title = self.info_font.render("COMMANDS", True, TEXT_COLOR)
        self.screen.blit(commands_title, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - commands_title.get_width() // 2, 350))
        
        commands = [
            "Arrows: Move selection",
            "Space: Place piece",
            "R: Reset",
            "Q: Quit"
        ]
        
        for i, cmd in enumerate(commands):
            cmd_text = self.info_font.render(cmd, True, TEXT_COLOR)
            self.screen.blit(cmd_text, (BOARD_WIDTH + 20, 390 + i * 30))
        
        # Display the winner if the game is over
        if self.game_over:
            if self.winner == "BLACK":
                winner_text = "Black wins!"
            elif self.winner == "WHITE":
                winner_text = "White wins!"
            else:
                winner_text = "Draw!"
            
            winner_surface = self.title_font.render(winner_text, True, TEXT_COLOR)
            self.screen.blit(winner_surface, (BOARD_WIDTH + INFO_PANEL_WIDTH // 2 - winner_surface.get_width() // 2, 500))
    
    def run(self):
        """Main game loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:  # Quit
                        running = False
                    
                    elif event.key == pygame.K_r:  # Reset
                        self.obs, _ = self.env.reset()
                        self.last_reward = 0
                        self.game_over = False
                        self.winner = None
                    
                    elif not self.game_over:  # Game actions only if the game is not over
                        if event.key == pygame.K_UP:
                            self.env.move_selection("up")
                        
                        elif event.key == pygame.K_DOWN:
                            self.env.move_selection("down")
                        
                        elif event.key == pygame.K_LEFT:
                            self.env.move_selection("left")
                        
                        elif event.key == pygame.K_RIGHT:
                            self.env.move_selection("right")
                        
                        elif event.key == pygame.K_SPACE:
                            # Check if the selection is valid
                            if self.env.is_selected_valid():
                                # Play the move
                                action = self.env.get_selected_position()
                                self.obs, reward, terminated, truncated, info = self.env.step(action)
                                self.last_reward = reward
                                
                                # Check if the game is over
                                if terminated:
                                    self.game_over = True
                                    self.winner = info.get("winner", None)
            
            # Draw the game
            self.draw_board()
            self.draw_info_panel()
            
            # Update the display
            pygame.display.flip()
            
            # Limit the frame rate
            self.clock.tick(60)
        
        # Quit Pygame
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = OthelloGame()
    game.run()