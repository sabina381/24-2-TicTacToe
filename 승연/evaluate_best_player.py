# import
import torch

from file_save_load import load_model
from mcts import Mcts

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/Users/seungyeonlee/Documents/GitHub/24-2-TicTacToe'))))

from Environment import Environment
from Environment import State

# parameter
env = Environment()
EP_NUM_GAME = 10
file_name = "model1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# best player를 평가하는 함수
# 만약 새로운 모델이 best model로 업데이트 된 경우
# 새로운 best model과 다른 알고리즘의 대국을 진행해 평가한다.

# first player point로 바꾸는 함수
def first_player_reward(final_state):
    is_first_player = final_state.check_first_player()
    reward = env.get_reward(final_state) if is_first_player else -env.get_reward(final_state)
    return reward


# 1 game play하는 함수
def play_game(player_list):
    is_done = False
    state = State()

    while not is_done:
        player = player_list[0] if state.check_first_player() else player_list[1]

        _, action = player.get_action()
        state, is_done, _ = env.step(state, action)

    reward = first_player_reward(state)

    point = 1 if reward == 1 else 0.5 if reward == 0 else 0
    return point # first player point


class RandomAgent:
    __slots__ = ()

    def get_action(self, state):
        return state.get_ramdom_action()


def evaluate_algorithm(label, player_list):
    total_point = 0
    for i in range(EP_NUM_GAME):
        if i % 2 == 0:
            total_point += play_game(player_list)
        else:
            total_point += play_game(player_list[[1, 0]])

    average_point = total_point / EP_NUM_GAME
    print(f"{label}: {average_point}/{total_point}")


def evaluate_best_player():
    model = load_model(f'{file_name}_model_best.pkl').to(device)

    mcts_best = Mcts(model)
    agents_dict = {'random': random_agent(),
                    'minimax': minimax_agent(), 
                    'alpha-beta': alpha_beta_agent(), 
                    'mcs': mcs_agent(),
                    'mcts': mcts_agent()}

    count = 0
    for key in agents_dict.keys:
        agent = agents_dict[key]
        player_list = [mcts_best, agent]
        count += 1
        evaluate_algorithm(f"[{count}/{len(agents_dict)}] vs. {key}", player_list)

    
    
        


