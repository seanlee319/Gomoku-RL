import pygame
import sys
import numpy as np
import random
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Initialize pygame
pygame.init()

# Constants
BOARD_SIZE = 15
GRID_SIZE = 40
STONE_RADIUS = 15
MARGIN = 40
BOARD_WIDTH = BOARD_SIZE * GRID_SIZE + 2 * MARGIN
BOARD_HEIGHT = BOARD_SIZE * GRID_SIZE + 2 * MARGIN
INFO_WIDTH = 600
WINDOW_WIDTH = BOARD_WIDTH + INFO_WIDTH
WINDOW_HEIGHT = BOARD_HEIGHT
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BROWN = (210, 180, 140)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 128, 0)

# Game variables
board = np.zeros((BOARD_SIZE, BOARD_SIZE))
current_player = 1
game_over = False
winner = None
ai_vs_ai = True
auto_play_delay = 0.1  # seconds between AI moves
move_count = 0

# Set up the display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Gomoku AI vs AI with Learning Visualization')
font = pygame.font.SysFont('Arial', 24)
large_font = pygame.font.SysFont('Arial', 36)

class QLearningAI:
    def __init__(self, player):
        self.player = player
        self.q_table = defaultdict(float)
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.3
        self.exploration_decay = 0.9995
        self.last_state = None
        self.last_action = None
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.q_values = []
        self.rewards = []
    
    def get_state_key(self, board):
        return str(board.flatten().tolist())
    
    def get_available_moves(self, board):
        moves = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if board[y][x] == 0:
                    moves.append((y, x))
        return moves
    
    def choose_action(self, board):
        state_key = self.get_state_key(board)
        available_moves = self.get_available_moves(board)
        
        if not available_moves:
            return None
        
        if random.random() < self.exploration_rate:
            return random.choice(available_moves)
        
        best_move = None
        max_q_value = -float('inf')
        total_q = 0
        count = 0
        
        for move in available_moves:
            action_key = str(move)
            q_value = self.q_table.get((state_key, action_key), 0.0)
            total_q += q_value
            count += 1
            if q_value > max_q_value:
                max_q_value = q_value
                best_move = move
        
        if count > 0:
            self.q_values.append(total_q / count)
        
        return best_move if best_move is not None else random.choice(available_moves)
    
    def learn(self, board, reward):
        if self.last_state is None or self.last_action is None:
            return
        
        old_state_key = self.last_state
        action_key = str(self.last_action)
        new_state_key = self.get_state_key(board)
        
        max_q_new = 0.0
        available_moves = self.get_available_moves(board)
        if available_moves:
            max_q_new = max(
                [self.q_table.get((new_state_key, str(move)), 0.0) 
                 for move in available_moves]
            )
        
        old_q_value = self.q_table.get((old_state_key, action_key), 0.0)
        new_q_value = old_q_value + self.learning_rate * (
            reward + self.discount_factor * max_q_new - old_q_value
        )
        self.q_table[(old_state_key, action_key)] = new_q_value
        self.rewards.append(reward)
        
        self.exploration_rate = max(0.05, self.exploration_rate * self.exploration_decay)
    
    def reset(self):
        self.last_state = None
        self.last_action = None

# Create AIs
black_ai = QLearningAI(1)
white_ai = QLearningAI(2)

# Tracking variables
games_played = 0
black_wins = []
white_wins = []
draws = []
exploration_rates = []
game_numbers = []
game_lengths = []

def draw_board():
    # Draw game board background
    pygame.draw.rect(screen, BROWN, (0, 0, BOARD_WIDTH, WINDOW_HEIGHT))
    
    # Draw grid lines
    for i in range(BOARD_SIZE):
        # Horizontal lines
        pygame.draw.line(screen, BLACK, 
                        (MARGIN, MARGIN + i * GRID_SIZE), 
                        (BOARD_WIDTH - MARGIN, MARGIN + i * GRID_SIZE), 2)
        # Vertical lines
        pygame.draw.line(screen, BLACK, 
                        (MARGIN + i * GRID_SIZE, MARGIN), 
                        (MARGIN + i * GRID_SIZE, WINDOW_HEIGHT - MARGIN), 2)
    
    # Draw star points
    star_points = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
    for y, x in star_points:
        center_x = MARGIN + x * GRID_SIZE
        center_y = MARGIN + y * GRID_SIZE
        pygame.draw.circle(screen, BLACK, (center_x, center_y), 4)
    
    # Draw stones
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == 1:
                pygame.draw.circle(screen, BLACK, 
                                 (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE), 
                                 STONE_RADIUS)
            elif board[y][x] == 2:
                pygame.draw.circle(screen, WHITE, 
                                 (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE), 
                                 STONE_RADIUS)
                pygame.draw.circle(screen, BLACK, 
                                 (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE), 
                                 STONE_RADIUS, 1)

    # Draw game status
    status_text = ""
    if game_over:
        if winner:
            status_text = f"{'Black' if winner == 1 else 'White'} wins!"
        else:
            status_text = "Draw!"
    else:
        status_text = f"{'Black' if current_player == 1 else 'White'}'s turn (Move {move_count})"
    
    status_surface = font.render(status_text, True, BLUE)
    screen.blit(status_surface, (MARGIN, 10))

def draw_info_panel():
    # Draw info panel background
    pygame.draw.rect(screen, WHITE, (BOARD_WIDTH, 0, INFO_WIDTH, WINDOW_HEIGHT))
    
    # Title
    title = large_font.render("Learning Progress", True, BLACK)
    screen.blit(title, (BOARD_WIDTH + 20, 20))
    
    # Game stats
    stats_y = 80
    stats = [
        f"Games Played: {games_played}",
        f"Black Wins: {black_ai.wins} ({black_ai.wins/games_played:.1%})" if games_played > 0 else "Black Wins: 0",
        f"White Wins: {white_ai.wins} ({white_ai.wins/games_played:.1%})" if games_played > 0 else "White Wins: 0",
        f"Draws: {black_ai.draws} ({black_ai.draws/games_played:.1%})" if games_played > 0 else "Draws: 0",
        "",
        f"Exploration Rates:",
        f"  Black: {black_ai.exploration_rate:.3f}",
        f"  White: {white_ai.exploration_rate:.3f}",
        "",
        f"Avg Game Length: {np.mean(game_lengths):.1f} moves" if game_lengths else ""
    ]
    
    for stat in stats:
        text = font.render(stat, True, BLACK)
        screen.blit(text, (BOARD_WIDTH + 20, stats_y))
        stats_y += 30
    
    # Draw graphs if we have enough data
    if games_played > 1:
        draw_graphs(stats_y + 20)

def draw_graphs(y_pos):
    try:
        # Create matplotlib figures
        fig1 = plt.figure(figsize=(5.5, 2.5), dpi=80)
        ax1 = fig1.add_subplot(111)
        
        # Win rate graph
        ax1.plot(game_numbers, black_wins, label='Black Wins', color='black')
        ax1.plot(game_numbers, white_wins, label='White Wins', color='blue')
        ax1.plot(game_numbers, draws, label='Draws', color='gray')
        ax1.set_title('Win Rates Over Time')
        ax1.legend(loc='upper left', fontsize='small')
        
        # Convert to pygame surface
        canvas1 = FigureCanvasAgg(fig1)
        canvas1.draw()
        renderer1 = canvas1.get_renderer()
        raw_data1 = renderer1.tostring_argb()
        size1 = canvas1.get_width_height()
        
        # Create graph surface
        graph_surf = pygame.image.fromstring(raw_data1, size1, "ARGB")
        screen.blit(graph_surf, (BOARD_WIDTH + 20, y_pos))
        
        # Second graph (Q-values)
        fig2 = plt.figure(figsize=(5.5, 2.5), dpi=80)
        ax2 = fig2.add_subplot(111)
        
        window = min(100, len(black_ai.q_values))
        if window > 10:
            ax2.plot(black_ai.q_values[-window:], label='Black Q-values', color='black')
            ax2.plot(white_ai.q_values[-window:], label='White Q-values', color='blue')
            ax2.set_title('Recent Move Quality')
            ax2.legend(loc='upper left', fontsize='small')
            
            canvas2 = FigureCanvasAgg(fig2)
            canvas2.draw()
            raw_data2 = canvas2.get_renderer().tostring_argb()
            size2 = canvas2.get_width_height()
            graph_surf2 = pygame.image.fromstring(raw_data2, size2, "ARGB")
            screen.blit(graph_surf2, (BOARD_WIDTH + 20, y_pos + 220))
        
        plt.close(fig1)
        plt.close(fig2)
    except Exception as e:
        print(f"Error drawing graphs: {e}")

def place_stone(y, x):
    global current_player, game_over, winner, move_count
    
    if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and board[y][x] == 0:
        board[y][x] = current_player
        move_count += 1
        
        if check_win(y, x):
            game_over = True
            winner = current_player
            if winner == 1:
                black_ai.wins += 1
                white_ai.losses += 1
                black_ai.learn(board, 1.0)
                white_ai.learn(board, -1.0)
            else:
                white_ai.wins += 1
                black_ai.losses += 1
                white_ai.learn(board, 1.0)
                black_ai.learn(board, -1.0)
        else:
            if np.all(board != 0):
                game_over = True
                black_ai.draws += 1
                white_ai.draws += 1
                black_ai.learn(board, 0.1)
                white_ai.learn(board, 0.1)
            else:
                if current_player == 1:
                    black_ai.learn(board, 0.01)
                else:
                    white_ai.learn(board, 0.01)
                current_player = 3 - current_player

def check_win(y, x):
    directions = [
        [(0, 1), (0, -1)],  # Horizontal
        [(1, 0), (-1, 0)],  # Vertical
        [(1, 1), (-1, -1)], # Diagonal \
        [(1, -1), (-1, 1)]  # Diagonal /
    ]
    
    player = board[y][x]
    
    for direction_pair in directions:
        count = 1
        
        for dy, dx in direction_pair:
            ny, nx = y + dy, x + dx
            while 0 <= ny < BOARD_SIZE and 0 <= nx < BOARD_SIZE and board[ny][nx] == player:
                count += 1
                ny += dy
                nx += dx
        
        if count >= 5:
            return True
    
    return False

def reset_game():
    global board, current_player, game_over, winner, games_played, move_count
    board = np.zeros((BOARD_SIZE, BOARD_SIZE))
    current_player = 1
    game_over = False
    winner = None
    black_ai.reset()
    white_ai.reset()
    
    games_played += 1
    game_numbers.append(games_played)
    black_wins.append(black_ai.wins)
    white_wins.append(white_ai.wins)
    draws.append(black_ai.draws)
    exploration_rates.append((black_ai.exploration_rate, white_ai.exploration_rate))
    game_lengths.append(move_count)
    move_count = 0

def ai_move():
    if current_player == 1:
        ai = black_ai
    else:
        ai = white_ai
    
    state_key = ai.get_state_key(board)
    ai.last_state = state_key
    move = ai.choose_action(board)
    ai.last_action = move
    
    if move:
        place_stone(move[0], move[1])

# Main game loop
last_ai_move_time = 0
clock = pygame.time.Clock()

while True:
    current_time = time.time()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                ai_vs_ai = not ai_vs_ai
            elif event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN and not ai_vs_ai:
            if not game_over:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if mouse_x < BOARD_WIDTH:  # Only process clicks on game board
                    board_x = round((mouse_x - MARGIN) / GRID_SIZE)
                    board_y = round((mouse_y - MARGIN) / GRID_SIZE)
                    place_stone(board_y, board_x)
            elif game_over:
                reset_game()
    
    # AI vs AI mode
    if ai_vs_ai and not game_over and current_time - last_ai_move_time > auto_play_delay:
        ai_move()
        last_ai_move_time = current_time
    elif ai_vs_ai and game_over and current_time - last_ai_move_time > auto_play_delay * 2:
        reset_game()
        last_ai_move_time = current_time
    
    # Draw everything
    draw_board()
    draw_info_panel()
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()