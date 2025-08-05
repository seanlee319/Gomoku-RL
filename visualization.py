import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

class GomokuVisualization:
    def __init__(self, env, black_ai, white_ai):
        self.env = env
        self.black_ai = black_ai
        self.white_ai = white_ai
        self.font = pygame.font.SysFont('Arial', 20)
        self.large_font = pygame.font.SysFont('Arial', 24)
        
        # Tracking variables
        self.games_played = 0
        self.black_win_counts = [0]  # Start with 0 for game 0
        self.white_win_counts = [0]  # Start with 0 for game 0
        self.draw_counts = [0]       # Start with 0 for game 0
        self.exploration_rates = []
        self.game_numbers = []
        self.game_lengths = []
    
    def draw_info_panel(self, screen, info_width):
        """Draw the right-side panel with learning statistics."""
        # Panel background
        pygame.draw.rect(screen, (240, 240, 240), (self.env.BOARD_WIDTH, 0, info_width, self.env.BOARD_HEIGHT))
        
        # Title
        title = self.large_font.render("Learning Progress", True, (0, 0, 0))
        screen.blit(title, (self.env.BOARD_WIDTH + 20, 20))
        
        # Game stats - three column layout
        stats_y = 70
        line_height = 22
        column1_x = self.env.BOARD_WIDTH + 20    # Games and results
        column2_x = self.env.BOARD_WIDTH + 200   # Exploration rates
        column3_x = self.env.BOARD_WIDTH + 350   # Avg Length
        
        # Left column: Games and results
        left_stats = [
            f"Games: {self.games_played}",
            f"Black Wins: {self.black_ai.wins} ({self.black_ai.wins/max(1, self.games_played):.1%})",
            f"White Wins: {self.white_ai.wins} ({self.white_ai.wins/max(1, self.games_played):.1%})",
            f"Draws: {self.black_ai.draws} ({self.black_ai.draws/max(1, self.games_played):.1%})",
        ]
        
        # Middle column: Exploration
        middle_stats = [
            f"Exploration:",
            f"Black: {self.black_ai.exploration_rate:.3f}",
            f"White: {self.white_ai.exploration_rate:.3f}",
        ]
        
        # Right column: Avg Length
        right_stats = [
            f"Avg Length:",
            f"{np.mean(self.game_lengths):.1f} moves" if self.game_lengths else "0 moves"
        ]
        
        # Draw left column stats
        for i, stat in enumerate(left_stats):
            text = self.font.render(stat, True, (0, 0, 0))
            screen.blit(text, (column1_x, stats_y + i * line_height))
        
        # Draw middle column stats
        for i, stat in enumerate(middle_stats):
            text = self.font.render(stat, True, (0, 0, 0))
            screen.blit(text, (column2_x, stats_y + i * line_height))
        
        # Draw right column stats
        for i, stat in enumerate(right_stats):
            text = self.font.render(stat, True, (0, 0, 0))
            screen.blit(text, (column3_x, stats_y + i * line_height))
        
        # Draw graphs below all columns
        max_lines = max(len(left_stats), len(middle_stats), len(right_stats))
        graph_height = (self.env.BOARD_HEIGHT - stats_y - 30 - max_lines*line_height) // 2
        self.draw_win_graph(screen, stats_y + max_lines*line_height + 20, info_width - 40, graph_height)
        self.draw_reward_graph(screen, stats_y + max_lines*line_height + 30 + graph_height, info_width - 40, graph_height)

    def draw_win_graph(self, screen, y_pos, width, height):
        """Draw the cumulative wins graph."""
        try:
            fig = plt.figure(figsize=(width/100, height/100), dpi=100)
            ax = fig.add_subplot(111)

            # Plot cumulative wins starting from game 0
            games = range(0, len(self.black_win_counts))
            ax.plot(games, self.black_win_counts, label='Black Wins', color='black', linewidth=1.5)
            ax.plot(games, self.white_win_counts, label='White Wins', color='blue', linewidth=1.5)
            ax.plot(games, self.draw_counts, label='Draws', color='gray', linewidth=1.5, linestyle='--')

            # Formatting
            ax.set_title('Cumulative Wins Over Games', fontsize=10)
            ax.set_xlabel('Game Number', fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            ax.set_xlim(0, max(1, len(self.black_win_counts)-1))
            ax.set_ylim(0, max(1, max(self.black_win_counts + self.white_win_counts + self.draw_counts)))
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.4)

            plt.tight_layout(pad=1)

            # Convert to Pygame surface
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            raw_data = renderer.tostring_argb()
            size = canvas.get_width_height()
            
            graph_surf = pygame.image.fromstring(raw_data, size, "ARGB")
            screen.blit(graph_surf, (self.env.BOARD_WIDTH + 20, y_pos))

            plt.close(fig)
        except Exception as e:
            print(f"Error drawing win graph: {e}")
    
    def draw_reward_graph(self, screen, y_pos, width, height):
        """Draw the final rewards per game graph for both AIs."""
        try:
            fig = plt.figure(figsize=(width/100, height/100), dpi=100)
            ax = fig.add_subplot(111)

            # Plot final rewards from both AIs if they have data
            if hasattr(self.black_ai, 'final_rewards') and self.black_ai.final_rewards:
                games = range(1, len(self.black_ai.final_rewards)+1)
                ax.plot(games, self.black_ai.final_rewards, 
                    label='Black Rewards', color='black', linewidth=1.5, marker='o', markersize=3)

            if hasattr(self.white_ai, 'final_rewards') and self.white_ai.final_rewards:
                games = range(1, len(self.white_ai.final_rewards)+1)
                ax.plot(games, self.white_ai.final_rewards, 
                    label='White Rewards', color='blue', linewidth=1.5, marker='o', markersize=3)

            # Formatting
            ax.set_title('Final Rewards per Game', fontsize=10)
            ax.set_xlabel('Game Number', fontsize=8)
            ax.set_ylabel('Reward', fontsize=8)
            
            # Set appropriate axis limits
            black_r = getattr(self.black_ai, 'final_rewards', [])
            white_r = getattr(self.white_ai, 'final_rewards', [])
            all_r = black_r + white_r
            
            if all_r:
                ax.set_xlim(1, max(len(black_r), len(white_r)))
                ax.set_ylim(min(all_r) - 0.2, max(all_r) + 0.2)
            else:
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 1.5)
                
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.4)

            plt.tight_layout(pad=1)

            # Convert to Pygame surface
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            raw_data = renderer.tostring_argb()
            size = canvas.get_width_height()
            
            graph_surf = pygame.image.fromstring(raw_data, size, "ARGB")
            screen.blit(graph_surf, (self.env.BOARD_WIDTH + 20, y_pos))

            plt.close(fig)
        except Exception as e:
            print(f"Error drawing reward graph: {e}")
        
    def update_stats(self, game_result, move_count):
        """Update statistics after each game."""
        self.games_played += 1
        self.game_lengths.append(move_count)
        self.game_numbers.append(self.games_played)
        
        # Update cumulative win counts
        prev_black = self.black_win_counts[-1]
        prev_white = self.white_win_counts[-1]
        prev_draw = self.draw_counts[-1]
        
        self.black_win_counts.append(prev_black + (1 if game_result == 1 else 0))
        self.white_win_counts.append(prev_white + (1 if game_result == 2 else 0))
        self.draw_counts.append(prev_draw + (1 if game_result is None else 0))
        
        self.exploration_rates.append((self.black_ai.exploration_rate, self.white_ai.exploration_rate))