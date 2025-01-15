# import
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/Users/seungyeonlee/Documents/GitHub/24-2-TicTacToe'))))

from Environment import Environment

# parameters
env = Environment()
DEPTH = 100 # 알파베타 알고리즘

# Agents
class RandomAgent:
    __slots__ = ()

    def get_action(self, state):
        return state.get_ramdom_action()

class AlphaBetaAgent:
    __slots__ = ('player', 'best_action')
    def __init__(self, is_first_player):
        self.player = is_first_player
        self.best_action = None
        

    def get_action(self, state):
        self.minimax(state, DEPTH, -np.Inf, np.Inf)
        return self.best_action

    def minimax(self, state, depth, alpha, beta):
        is_done, _ = state.check_done()
        reward = 0

        legal_actions = np.where(state.get_legal_actions()==1)

        if is_done or (depth == 0):
            reward = env.get_reward(state)
            return reward

        if state.check_first_player() == self.player: # max player
            max_eval = -np.Inf

            for action in legal_actions:
                next_state, _, _ = env.step(state)

                eval = self.minimax(next_state, depth-1, alpha, beta)
                
                if eval > max_eval:
                    best_action = action
                    max_eval = eval
                
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

                if depth == DEPTH: # 최상위 호출에서만 best action 저장
                    self.best_action = best_action
            
            return max_eval

        else: # min player
            min_eval = np.Inf
            for action in legal_actions:
                next_state, _, _ = env.step(state)

                eval = self.minimax(next_state, depth-1, alpha, beta)
                min_eval = min(min_eval, eval)

                beta = min(beta, eval)
                if beta <= alpha:
                    break
            
            return min_eval



        

    