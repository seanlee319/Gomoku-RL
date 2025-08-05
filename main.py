import pygame
import sys
import time
from gomoku import GomokuEnvironment
from ai import QLearningAI
from visualization import GomokuVisualization

# Constants
INFO_WIDTH = 600

def main():
    # Initialize game components
    env = GomokuEnvironment()
    black_ai = QLearningAI(1)
    white_ai = QLearningAI(2)
    visualizer = GomokuVisualization(env, black_ai, white_ai)
    
    # Set up the display
    WINDOW_WIDTH = env.BOARD_WIDTH + INFO_WIDTH
    WINDOW_HEIGHT = env.BOARD_HEIGHT
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption('Gomoku AI vs AI with Learning Visualization')
    
    # Game variables
    ai_vs_ai = True
    auto_play_delay = 0.1  # seconds between AI moves
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
                if not env.game_over:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if mouse_x < env.BOARD_WIDTH:  # Only process clicks on game board
                        board_x = round((mouse_x - env.MARGIN) / env.GRID_SIZE)
                        board_y = round((mouse_y - env.MARGIN) / env.GRID_SIZE)
                        reward, done = env.place_stone(board_y, board_x)
                        if done:
                            if env.winner == 1:
                                black_ai.wins += 1
                                white_ai.losses += 1
                                black_ai.learn(env.board, 1.0)
                                white_ai.learn(env.board, -1.0)
                            elif env.winner == 2:
                                white_ai.wins += 1
                                black_ai.losses += 1
                                white_ai.learn(env.board, 1.0)
                                black_ai.learn(env.board, -1.0)
                            else:
                                black_ai.draws += 1
                                white_ai.draws += 1
                                black_ai.learn(env.board, 0.1)
                                white_ai.learn(env.board, 0.1)
                elif env.game_over:
                    env.reset()
                    black_ai.reset()
                    white_ai.reset()
                    visualizer.update_stats(env.winner, env.move_count)
        
        # AI vs AI mode
        if ai_vs_ai and not env.game_over and current_time - last_ai_move_time > auto_play_delay:
            if env.current_player == 1:
                ai = black_ai
            else:
                ai = white_ai
            
            state_key = ai.get_state_key(env.board)
            ai.last_state = state_key
            move = ai.choose_action(env.board)
            ai.last_action = move
            
            if move:
                reward, done = env.place_stone(move[0], move[1])
                if done:
                    if env.winner == 1:
                        black_ai.wins += 1
                        white_ai.losses += 1
                        black_ai.learn(env.board, 1.0)
                        white_ai.learn(env.board, -1.0)
                    elif env.winner == 2:
                        white_ai.wins += 1
                        black_ai.losses += 1
                        white_ai.learn(env.board, 1.0)
                        black_ai.learn(env.board, -1.0)
                    else:
                        black_ai.draws += 1
                        white_ai.draws += 1
                        black_ai.learn(env.board, 0.1)
                        white_ai.learn(env.board, 0.1)
                else:
                    ai.learn(env.board, reward)
            
            last_ai_move_time = current_time
        elif ai_vs_ai and env.game_over and current_time - last_ai_move_time > auto_play_delay * 2:
            visualizer.update_stats(env.winner, env.move_count)
            env.reset()
            black_ai.reset()
            white_ai.reset()
            last_ai_move_time = current_time
        
        # Draw everything
        env.render(screen)
        visualizer.draw_info_panel(screen, INFO_WIDTH)
        
        # Draw game status
        status_text = ""
        if env.game_over:
            if env.winner:
                status_text = f"{'Black' if env.winner == 1 else 'White'} wins!"
            else:
                status_text = "Draw!"
        else:
            status_text = f"{'Black' if env.current_player == 1 else 'White'}'s turn (Move {env.move_count})"
        
        status_surface = visualizer.font.render(status_text, True, (0, 0, 255))
        screen.blit(status_surface, (env.MARGIN, 10))
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()