from collections import defaultdict
import random
import numpy as np

class QLearningAI:
    def __init__(self, player):
        self.player = player
        self.q_table = defaultdict(float)
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.45
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
        for y in range(board.shape[0]):
            for x in range(board.shape[1]):
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
        
        q_values = []
        best_move = None
        max_q_value = -float('inf')
        
        for move in available_moves:
            action_key = str(move)
            q_value = self.q_table.get((state_key, action_key), 0.0)
            q_values.append(q_value)
            if q_value > max_q_value:
                max_q_value = q_value
                best_move = move
        
        # Track the maximum Q-value for the current state
        if q_values:
            self.q_values.append(max_q_value)
        
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