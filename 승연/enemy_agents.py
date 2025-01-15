# import

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/Users/seungyeonlee/Documents/GitHub/24-2-TicTacToe'))))

from Environment import Environment
from Environment import State


# Agents
class RandomAgent:
    __slots__ = ()

    def get_action(self, state):
        return state.get_ramdom_action()

