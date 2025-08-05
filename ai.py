from collections import defaultdict
import random
import numpy as np

class QLearningAI:
    def __init__(self, player):
        self.player = player  # 1 for black, 2 for white
        self.q_table = defaultdict(float)  # Stores Q-values for state-action pairs
        self.learning_rate = 0.1  # How much new info overrides old
        self.discount_factor = 0.9  # Importance of future rewards
        self.exploration_rate = 0.45  # Initial exploration probability
        self.exploration_decay = 0.9995  # Rate at which exploration decreases
        self.min_exploration = 0.05  # Minimum exploration rate
        self.last_state = None  # Previous state for learning
        self.last_action = None  # Previous action for learning
        self.wins = 0  # Total wins
        self.losses = 0  # Total losses
        self.draws = 0  # Total draws
        self.q_values = []  # Tracks Q-values during gameplay
        self.rewards = []  # Tracks rewards received
        self.max_q_per_game = []  # Tracks maximum Q-value per game
        self.final_rewards = []  # Tracks final rewards per game

    def get_state_key(self, board):
        """Convert the board state to a string key for Q-table lookup."""
        return str(board.flatten().tolist())

    def get_available_moves(self, board):
        """Get all valid moves (empty positions) on the current board."""
        moves = []
        for y in range(board.shape[0]):
            for x in range(board.shape[1]):
                if board[y][x] == 0:
                    moves.append((y, x))
        return moves

    def choose_action(self, board):
        """
        Choose an action using ε-greedy policy.
        With probability ε, explore randomly; otherwise exploit best known action.
        """
        state_key = self.get_state_key(board)
        available_moves = self.get_available_moves(board)
        
        if not available_moves:
            return None
        
        # Exploration: choose random action
        if random.random() < self.exploration_rate:
            return random.choice(available_moves)
        
        # Exploitation: choose best known action
        best_move = None
        max_q_value = -float('inf')
        
        for move in available_moves:
            action_key = str(move)
            q_value = self.q_table.get((state_key, action_key), 0.0)
            if q_value > max_q_value:
                max_q_value = q_value
                best_move = move
        
        # Track the maximum Q-value for the current state
        if max_q_value != -float('inf'):
            self.q_values.append(max_q_value)
        
        return best_move if best_move is not None else random.choice(available_moves)

    def learn(self, board, reward):
        """
        Update Q-values using the Q-learning algorithm.
        Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        """
        if self.last_state is None or self.last_action is None:
            return
        
        old_state_key = self.last_state
        action_key = str(self.last_action)
        new_state_key = self.get_state_key(board)
        
        # Calculate maximum Q-value for the new state
        max_q_new = 0.0
        available_moves = self.get_available_moves(board)
        if available_moves:
            max_q_new = max(
                [self.q_table.get((new_state_key, str(move)), 0.0) 
                 for move in available_moves]
            )
        
        # Q-learning update rule
        old_q_value = self.q_table.get((old_state_key, action_key), 0.0)
        new_q_value = old_q_value + self.learning_rate * (
            reward + self.discount_factor * max_q_new - old_q_value
        )
        
        # Update Q-table
        self.q_table[(old_state_key, action_key)] = new_q_value
        self.rewards.append(reward)
        
        # Decay exploration rate with lower bound
        self.exploration_rate = max(
            self.min_exploration, 
            self.exploration_rate * self.exploration_decay
        )

    def reset(self):
        """Reset the agent's temporary state between games."""
        if self.q_values:  # Only store if we have values
            self.max_q_per_game.append(max(self.q_values))
        
        # Store the last reward if it exists (from learn() calls)
        if self.rewards:
            self.final_rewards.append(self.rewards[-1])
        
        self.last_state = None
        self.last_action = None
        self.q_values = []
        self.rewards = []  # Clear rewards for next game