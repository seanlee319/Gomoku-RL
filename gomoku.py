import pygame
import numpy as np
from ai import QLearningAI

class GomokuEnvironment:
    def __init__(self):
        # Game constants
        self.BOARD_SIZE = 15  # 15x15 board
        self.GRID_SIZE = 40  # pixels per grid square
        self.STONE_RADIUS = 15  # stone size in pixels
        self.MARGIN = 40  # board margin in pixels
        self.BOARD_WIDTH = self.BOARD_SIZE * self.GRID_SIZE + self.MARGIN
        self.BOARD_HEIGHT = self.BOARD_SIZE * self.GRID_SIZE + self.MARGIN
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BROWN = (210, 180, 140)  # board color
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 128, 0)
        
        # Game state
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE))
        self.current_player = 1  # 1=black, 2=white
        self.game_over = False
        self.winner = None
        self.move_count = 0
        
        # Initialize pygame
        pygame.init()
        
    def reset(self):
        """Reset the game to initial state."""
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE))
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_count = 0
        return self.board
    
    def place_stone(self, y, x):
        """
        Place a stone at (y,x) and return (reward, done).
        New reward structure:
        - Win: 1 for winner, 0 for loser
        - Draw: 0.5 for both players
        - Open 3-in-a-row: 0.05 bonus
        - Continue: 0.01 for valid move
        - Invalid: 0.0
        """
        if (0 <= x < self.BOARD_SIZE and 
            0 <= y < self.BOARD_SIZE and 
            self.board[y][x] == 0 and 
            not self.game_over):
            
            self.board[y][x] = self.current_player
            self.move_count += 1
            
            reward = 0.01  # Base reward for valid move
            
            # Check for open 3-in-a-row first
            if self.check_three_open(y, x):
                reward += 0.05  # Bonus for creating open 3
            
            if self.check_win(y, x):
                self.game_over = True
                self.winner = self.current_player
                # Override with win/loss rewards
                reward = 1.0 if self.current_player == 1 else 0.0
            elif np.all(self.board != 0):  # Board full
                self.game_over = True
                reward = 0.5  # Medium reward for draw
            else:
                self.current_player = 3 - self.current_player  # Switch player
                
            return reward, self.game_over
        
        return 0.0, self.game_over
    
    def check_three_open(self, y, x):
        """Check if the placed stone creates an open 3-in-a-row formation.
        Returns True if there's an open 3-in-a-row with no blocking stones on either end."""
        directions = [
            [(0, 1), (0, -1)],  # Horizontal
            [(1, 0), (-1, 0)],   # Vertical
            [(1, 1), (-1, -1)],  # Diagonal \
            [(1, -1), (-1, 1)]   # Diagonal /
        ]
        
        player = self.board[y][x]
        
        for direction_pair in directions:
            count = 1  # Count the current stone
            open_ends = 0
            
            # Check both directions in the pair
            for dy, dx in direction_pair:
                ny, nx = y + dy, x + dx
                consecutive = True
                
                while (0 <= ny < self.BOARD_SIZE and 
                    0 <= nx < self.BOARD_SIZE and 
                    consecutive):
                    if self.board[ny][nx] == player:
                        count += 1
                        ny += dy
                        nx += dx
                    else:
                        consecutive = False
                        # Check if the end is open (empty)
                        if (0 <= ny < self.BOARD_SIZE and 
                            0 <= nx < self.BOARD_SIZE and 
                            self.board[ny][nx] == 0):
                            open_ends += 1
            
            # Check if we have exactly 3 in a row with both ends open
            if count == 3 and open_ends == 2:
                return True
        
        return False
    
    def check_win(self, y, x):
        """Check if the last move at (y,x) caused a win."""
        directions = [
            [(0, 1), (0, -1)],  # Horizontal
            [(1, 0), (-1, 0)],  # Vertical
            [(1, 1), (-1, -1)], # Diagonal \
            [(1, -1), (-1, 1)]  # Diagonal /
        ]
        
        player = self.board[y][x]
        
        for direction_pair in directions:
            count = 1  # Count the current stone
            
            for dy, dx in direction_pair:
                ny, nx = y + dy, x + dx
                while (0 <= ny < self.BOARD_SIZE and 
                       0 <= nx < self.BOARD_SIZE and 
                       self.board[ny][nx] == player):
                    count += 1
                    ny += dy
                    nx += dx
            
            if count >= 5:
                return True
        
        return False
    
    def get_available_moves(self):
        """Get all empty positions on the board."""
        moves = []
        for y in range(self.BOARD_SIZE):
            for x in range(self.BOARD_SIZE):
                if self.board[y][x] == 0:
                    moves.append((y, x))
        return moves
    
    def render(self, screen):
        """Render the game board and stones."""
        # Fill background
        screen.fill((255, 255, 255))
        # Draw board
        pygame.draw.rect(screen, self.BROWN, (0, 0, self.BOARD_WIDTH, self.BOARD_HEIGHT))
        
        # Draw grid lines
        for i in range(self.BOARD_SIZE):
            # Horizontal lines
            pygame.draw.line(
                screen, self.BLACK,
                (self.MARGIN, self.MARGIN + i * self.GRID_SIZE),
                (self.BOARD_WIDTH - self.MARGIN, self.MARGIN + i * self.GRID_SIZE), 2
            )
            # Vertical lines
            pygame.draw.line(
                screen, self.BLACK,
                (self.MARGIN + i * self.GRID_SIZE, self.MARGIN),
                (self.MARGIN + i * self.GRID_SIZE, self.BOARD_HEIGHT - self.MARGIN), 2
            )
        
        # Draw star points
        star_points = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        for y, x in star_points:
            center_x = self.MARGIN + x * self.GRID_SIZE
            center_y = self.MARGIN + y * self.GRID_SIZE
            pygame.draw.circle(screen, self.BLACK, (center_x, center_y), 4)
        
        # Draw stones
        for y in range(self.BOARD_SIZE):
            for x in range(self.BOARD_SIZE):
                if self.board[y][x] == 1:  # Black stone
                    pygame.draw.circle(
                        screen, self.BLACK,
                        (self.MARGIN + x * self.GRID_SIZE, self.MARGIN + y * self.GRID_SIZE),
                        self.STONE_RADIUS
                    )
                elif self.board[y][x] == 2:  # White stone
                    pygame.draw.circle(
                        screen, self.WHITE,
                        (self.MARGIN + x * self.GRID_SIZE, self.MARGIN + y * self.GRID_SIZE),
                        self.STONE_RADIUS
                    )
                    pygame.draw.circle(
                        screen, self.BLACK,
                        (self.MARGIN + x * self.GRID_SIZE, self.MARGIN + y * self.GRID_SIZE),
                        self.STONE_RADIUS, 1  # Outline
                    )