# import

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/Users/seungyeonlee/Documents/GitHub/24-2-TicTacToe'))))

from Environment import Environment
from Environment import State


# Agents
class EnemyAgents:
    def __init__(self):
        pass

    def get_random_action(self, state):
        return state.get_random_action()