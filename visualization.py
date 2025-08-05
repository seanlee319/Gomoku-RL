import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np 

class GomokuVisualization:
    def __init__(self, env, black_ai, white_ai):
        self.env = env
        self.black_ai = black_ai
        self.white_ai = white_ai
        self.font = pygame.font.SysFont('Arial', 24)
        self.large_font = pygame.font.SysFont('Arial', 36)
        
        # Tracking variables
        self.games_played = 0
        self.black_wins = []
        self.white_wins = []
        self.draws = []
        self.exploration_rates = []
        self.game_numbers = []
        self.game_lengths = []
    
    def draw_info_panel(self, screen, info_width):
        # Draw info panel background
        pygame.draw.rect(screen, (255, 255, 255), (self.env.BOARD_WIDTH, 0, info_width, self.env.BOARD_HEIGHT))
        
        # Title
        title = self.large_font.render("Learning Progress", True, (0, 0, 0))
        screen.blit(title, (self.env.BOARD_WIDTH + 20, 20))
        
        # Game stats
        stats_y = 80
        stats = [
            f"Games Played: {self.games_played}",
            f"Black Wins: {self.black_ai.wins} ({self.black_ai.wins/self.games_played:.1%})" if self.games_played > 0 else "Black Wins: 0",
            f"White Wins: {self.white_ai.wins} ({self.white_ai.wins/self.games_played:.1%})" if self.games_played > 0 else "White Wins: 0",
            f"Draws: {self.black_ai.draws} ({self.black_ai.draws/self.games_played:.1%})" if self.games_played > 0 else "Draws: 0",
            "",
            f"Exploration Rates:",
            f"  Black: {self.black_ai.exploration_rate:.3f}",
            f"  White: {self.white_ai.exploration_rate:.3f}",
            "",
            f"Avg Game Length: {np.mean(self.game_lengths):.1f} moves" if self.game_lengths else ""
        ]
        
        for stat in stats:
            text = self.font.render(stat, True, (0, 0, 0))
            screen.blit(text, (self.env.BOARD_WIDTH + 20, stats_y))
            stats_y += 30
        
        # Draw graphs if we have enough data
        if self.games_played > 1:
            self.draw_graphs(screen, stats_y + 20, info_width)
    
    def draw_graphs(self, screen, y_pos, info_width):
        try:
            # Create matplotlib figures
            fig1 = plt.figure(figsize=(5.5, 2.5), dpi=80)
            ax1 = fig1.add_subplot(111)
            
            # Win rate graph
            ax1.plot(self.game_numbers, self.black_wins, label='Black Wins', color='black')
            ax1.plot(self.game_numbers, self.white_wins, label='White Wins', color='blue')
            ax1.plot(self.game_numbers, self.draws, label='Draws', color='gray')
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
            screen.blit(graph_surf, (self.env.BOARD_WIDTH + 20, y_pos))
            
            # Second graph (Q-values)
            fig2 = plt.figure(figsize=(5.5, 2.5), dpi=80)
            ax2 = fig2.add_subplot(111)
            
            window = min(100, len(self.black_ai.q_values))
            if window > 10:
                ax2.plot(self.black_ai.q_values[-window:], label='Black Q-values', color='black')
                ax2.plot(self.white_ai.q_values[-window:], label='White Q-values', color='blue')
                ax2.set_title('Recent Move Quality')
                ax2.legend(loc='upper left', fontsize='small')
                
                canvas2 = FigureCanvasAgg(fig2)
                canvas2.draw()
                raw_data2 = canvas2.get_renderer().tostring_argb()
                size2 = canvas2.get_width_height()
                graph_surf2 = pygame.image.fromstring(raw_data2, size2, "ARGB")
                screen.blit(graph_surf2, (self.env.BOARD_WIDTH + 20, y_pos + 220))
            
            plt.close(fig1)
            plt.close(fig2)
        except Exception as e:
            print(f"Error drawing graphs: {e}")
    
    def update_stats(self, game_result, move_count):
        self.games_played += 1
        self.game_numbers.append(self.games_played)
        self.black_wins.append(self.black_ai.wins)
        self.white_wins.append(self.white_ai.wins)
        self.draws.append(self.black_ai.draws)
        self.exploration_rates.append((self.black_ai.exploration_rate, self.white_ai.exploration_rate))
        self.game_lengths.append(move_count)