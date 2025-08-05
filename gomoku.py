import pygame
import numpy as np
from ai import QLearningAI

class GomokuEnvironment:
    def __init__(self):
        # Constants
        self.BOARD_SIZE = 15
        self.GRID_SIZE = 40
        self.STONE_RADIUS = 15
        self.MARGIN = 40
        self.BOARD_WIDTH = self.BOARD_SIZE * self.GRID_SIZE + self.MARGIN
        self.BOARD_HEIGHT = self.BOARD_SIZE * self.GRID_SIZE + self.MARGIN
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BROWN = (210, 180, 140)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 128, 0)
        
        # Game variables
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE))
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_count = 0
        
        # Initialize pygame if needed
        pygame.init()
        
    def reset(self):
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE))
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_count = 0
        return self.board
    
    def place_stone(self, y, x):
        if 0 <= x < self.BOARD_SIZE and 0 <= y < self.BOARD_SIZE and self.board[y][x] == 0:
            self.board[y][x] = self.current_player
            self.move_count += 1
            
            if self.check_win(y, x):
                self.game_over = True
                self.winner = self.current_player
                reward = 1.0
            elif np.all(self.board != 0):
                self.game_over = True
                reward = 0.1
            else:
                self.current_player = 3 - self.current_player
                reward = 0.01
                
            return reward, self.game_over
        return 0, self.game_over
    
    def check_win(self, y, x):
        directions = [
            [(0, 1), (0, -1)],  # Horizontal
            [(1, 0), (-1, 0)],  # Vertical
            [(1, 1), (-1, -1)], # Diagonal \
            [(1, -1), (-1, 1)]  # Diagonal /
        ]
        
        player = self.board[y][x]
        
        for direction_pair in directions:
            count = 1
            
            for dy, dx in direction_pair:
                ny, nx = y + dy, x + dx
                while 0 <= ny < self.BOARD_SIZE and 0 <= nx < self.BOARD_SIZE and self.board[ny][nx] == player:
                    count += 1
                    ny += dy
                    nx += dx
            
            if count >= 5:
                return True
        
        return False
    
    def get_available_moves(self):
        moves = []
        for y in range(self.BOARD_SIZE):
            for x in range(self.BOARD_SIZE):
                if self.board[y][x] == 0:
                    moves.append((y, x))
        return moves
    
    def render(self, screen):
        # Fill the entire screen with white first
        screen.fill((255, 255, 255))
        # Draw game board background
        pygame.draw.rect(screen, self.BROWN, (0, 0, self.BOARD_WIDTH, self.BOARD_HEIGHT))
        
        # Draw grid lines
        for i in range(self.BOARD_SIZE):
            # Horizontal lines
            pygame.draw.line(screen, self.BLACK, 
                            (self.MARGIN, self.MARGIN + i * self.GRID_SIZE), 
                            (self.BOARD_WIDTH - self.MARGIN, self.MARGIN + i * self.GRID_SIZE), 2)
            # Vertical lines
            pygame.draw.line(screen, self.BLACK, 
                            (self.MARGIN + i * self.GRID_SIZE, self.MARGIN), 
                            (self.MARGIN + i * self.GRID_SIZE, self.BOARD_HEIGHT - self.MARGIN), 2)
        
        # Draw star points
        star_points = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        for y, x in star_points:
            center_x = self.MARGIN + x * self.GRID_SIZE
            center_y = self.MARGIN + y * self.GRID_SIZE
            pygame.draw.circle(screen, self.BLACK, (center_x, center_y), 4)
        
        # Draw stones
        for y in range(self.BOARD_SIZE):
            for x in range(self.BOARD_SIZE):
                if self.board[y][x] == 1:
                    pygame.draw.circle(screen, self.BLACK, 
                                     (self.MARGIN + x * self.GRID_SIZE, self.MARGIN + y * self.GRID_SIZE), 
                                     self.STONE_RADIUS)
                elif self.board[y][x] == 2:
                    pygame.draw.circle(screen, self.WHITE, 
                                     (self.MARGIN + x * self.GRID_SIZE, self.MARGIN + y * self.GRID_SIZE), 
                                     self.STONE_RADIUS)
                    pygame.draw.circle(screen, self.BLACK, 
                                     (self.MARGIN + x * self.GRID_SIZE, self.MARGIN + y * self.GRID_SIZE), 
                                     self.STONE_RADIUS, 1)