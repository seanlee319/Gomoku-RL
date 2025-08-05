import pygame
import sys
import numpy as np

# Initialize pygame
pygame.init()

# Constants
BOARD_SIZE = 15
GRID_SIZE = 40
STONE_RADIUS = 15
MARGIN = 40
WIDTH = BOARD_SIZE * GRID_SIZE + MARGIN
HEIGHT = BOARD_SIZE * GRID_SIZE + MARGIN
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BROWN = (210, 180, 140)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Game variables
board = np.zeros((BOARD_SIZE, BOARD_SIZE))  # 0 for empty, 1 for black, 2 for white
current_player = 1  # Black goes first
game_over = False
winner = None

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Gomoku')
font = pygame.font.SysFont('Arial', 30)

def draw_board():
    screen.fill(BROWN)
    
    # Draw grid lines
    for i in range(BOARD_SIZE):
        # Horizontal lines
        pygame.draw.line(screen, BLACK, 
                         (MARGIN, MARGIN + i * GRID_SIZE), 
                         (WIDTH - MARGIN, MARGIN + i * GRID_SIZE), 2)
        # Vertical lines
        pygame.draw.line(screen, BLACK, 
                         (MARGIN + i * GRID_SIZE, MARGIN), 
                         (MARGIN + i * GRID_SIZE, HEIGHT - MARGIN), 2)
    
    # Draw star points (small dots)
    star_points = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]  # (row, col)
    for y, x in star_points:
        center_x = MARGIN + x * GRID_SIZE
        center_y = MARGIN + y * GRID_SIZE
        pygame.draw.circle(screen, BLACK, (center_x, center_y), 4)
    
    # Draw stones
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == 1:  # Black stone
                pygame.draw.circle(screen, BLACK, 
                                  (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE), 
                                  STONE_RADIUS)
            elif board[y][x] == 2:  # White stone
                pygame.draw.circle(screen, WHITE, 
                                  (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE), 
                                  STONE_RADIUS)
                pygame.draw.circle(screen, BLACK, 
                                  (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE), 
                                  STONE_RADIUS, 1)  # Border

def place_stone(x, y):
    global current_player, game_over, winner
    
    # Convert screen coordinates to board indices
    board_x = round((x - MARGIN) / GRID_SIZE)
    board_y = round((y - MARGIN) / GRID_SIZE)
    
    # Check if the move is valid
    if 0 <= board_x < BOARD_SIZE and 0 <= board_y < BOARD_SIZE and board[board_y][board_x] == 0:
        board[board_y][board_x] = current_player
        
        # Check for win
        if check_win(board_y, board_x):
            game_over = True
            winner = current_player
        else:
            # Switch player
            current_player = 3 - current_player  # Switches between 1 and 2

def check_win(y, x):
    directions = [
        [(0, 1), (0, -1)],  # Horizontal
        [(1, 0), (-1, 0)],  # Vertical
        [(1, 1), (-1, -1)], # Diagonal \
        [(1, -1), (-1, 1)]  # Diagonal /
    ]
    
    player = board[y][x]
    
    for direction_pair in directions:
        count = 1  # The stone just placed
        
        for dy, dx in direction_pair:
            ny, nx = y + dy, x + dx
            while 0 <= ny < BOARD_SIZE and 0 <= nx < BOARD_SIZE and board[ny][nx] == player:
                count += 1
                ny += dy
                nx += dx
        
        if count >= 5:
            return True
    
    return False

def draw_game_status():
    if game_over:
        text = f"{'Black' if winner == 1 else 'White'} wins! Click to restart."
    else:
        text = f"{'Black' if current_player == 1 else 'White'}'s turn"
    
    status_surface = font.render(text, True, BLUE)
    screen.blit(status_surface, (MARGIN, 10))

def reset_game():
    global board, current_player, game_over, winner
    board = np.zeros((BOARD_SIZE, BOARD_SIZE))
    current_player = 1
    game_over = False
    winner = None

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if game_over:
                reset_game()
            else:
                place_stone(event.pos[0], event.pos[1])
    
    draw_board()
    draw_game_status()
    pygame.display.flip()