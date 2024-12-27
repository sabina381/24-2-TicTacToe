import numpy as np
import random
import math
from typing import Tuple
import copy

# parameter
STATE_SIZE = (3, 3)

# State
class State:
    def __init__(self, state=None, enemy_state=None):
        self.n = STATE_SIZE[0]
        self.action_space = np.arange(self.n ** 2)
        self.state = state if state != None else [0] * (self.n ** 2)
        self.enemy_state = enemy_state if enemy_state != None else [0] * (self.n ** 2)

        self.state = np.array(self.state).reshape(STATE_SIZE)
        self.enemy_state = np.array(self.enemy_state).reshape(STATE_SIZE)


    def total_pieces_count(self):
        total_state = self.state + self.enemy_state
        return np.sum(total_state)


    def get_legal_actions(self):
        total_state = (self.state + self.enemy_state).reshape(-1)
        legal_actions = np.array([total_state[x] == 0 for x in self.action_space], dtype = int)
        return legal_actions


    def is_done(self):
        is_done, is_lose = False, False

        # Check draw
        if self.total_pieces_count() == self.n ** 2:
            is_done, is_lose = True, False

        # Check lose
        lose_condition = np.concatenate([self.enemy_state.sum(axis=0), self.enemy_state.sum(axis=1), [self.enemy_state.trace], [np.fliplr(self.enemy_state).trace()]])
        if self.n in lose_condition:
            is_done, is_lose = True, True
        
        return is_done, is_lose


    def next(self, action_idx):
        x, y =np.divmod(action_idx, self.n)
        state = self.state.copy()
        state[x, y] = 1

        state = list(state.reshape(-1))
        enemy_state = list(copy.copy(self.enemy_state).reshape(-1))

        return State(enemy_state, state)


    def is_first_player(self):
        return (self.total_pieces_count() % 2) == 0


    def render(self):
        board = self.state - self.enemy_state if self.is_first_player() else self.enemy_state - self.state
        board = board.reshape(-1)
        board_list = list(map(lambda x: 'X' if board[x] == 1 else 'O' if board[x] == -1 else '.', self.action_space))

        board_string = ' '.join(board_list)
        formatted_string = '\n'.join([board_string[i:i+6] for i in range(0, len(board_string), 6)])

        print(formatted_string)
        # print("-"*7)

    def get_random_action(self):
        legal_actions = self.get_legal_actions()
        legal_action_idxs = np.where(legal_actions != 0)[0]
        action = np.random.choice(legal_action_idxs)
        return action



