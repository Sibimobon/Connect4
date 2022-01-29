import numpy as np
from agents.common import BoardPiece, apply_player_action, connected_four, PLAYER1, PLAYER2, NO_PLAYER, PlayerAction, SavedState
from typing import Optional, Tuple
import random, time, math
# import time
import pandas as pd

from tensorflow import keras
# new env requirements.txt
model = keras.models.load_model(r"""C:\Users\Simon\Desktop\Projekte\Programmierpraktikum\Connect4\agents\agent_mcts_nn""")


# saving the board might be a dumb idea.... always applying the move could be more memory effiecient
# stop iterating tree at axis after game is won?
class Node:
    def __init__(self, board: np.ndarray, player: BoardPiece, value: int, executed_sims: int, last_move: Optional[int],
                 uct: Optional[float], left_to_expand: Optional[list], win_state, parent_node):
        self.board = board
        self.player = player
        self.value = value
        self.executed_sims = executed_sims
        self.child_nodes = []
        self.left_to_expand = left_to_expand
        self.last_move = last_move
        self.uct = uct
        # None until the board passed in the tree has win/loss/draw
        self.win_state = win_state
        self.parent_node = parent_node

#uct_constant = 7 works kind of well
uct_constant = 10000


def generate_move_mcts(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    time_limit = 5  # time in sec usually is 5
    running_until = time.time() + time_limit
    tree = Node(board, player, 1, 1, None, None, possible_moves(board), None, None)
    running = True
    iterations = 0
    while time.time() < running_until:
        if running:
            running = False
            walk_tree(tree, player)
            running = True
            iterations += 1
    opt = []
    for c in tree.child_nodes:
        opt.append((c.last_move, c.value))
    opt.sort(key=lambda tup: tup[0])
    print(opt)
    print(iterations)

    return select_best_move(tree), None


# select best move for the root node
def select_best_move(root_node: Node):
    # temporary node for the first iteration
    best_node = None
    best_node_value = float('-inf')
    for node in root_node.child_nodes:
        if node.value > best_node_value:
            best_node = node
            best_node_value = node.value
    return best_node.last_move


def walk_tree(node: Node, simulating_player: BoardPiece):
    node.executed_sims += 1
    if node.win_state!= None:
        node.value += node.win_state
        # update uct value!!!
        node.uct = node.value + uct_constant * (math.sqrt(math.log(node.parent_node.executed_sims)/node.executed_sims))
        return
    if possible_moves(node.board) == []:
        # maybe put uct adjustments in extra method
        node.value -= 1
        node.uct = node.value + uct_constant * (
            math.sqrt(math.log(node.parent_node.executed_sims) / node.executed_sims))
        return
    if node.child_nodes == [] and node.left_to_expand != []:
        expand_node(node, simulating_player)
    else:
        best_move = None
        best_uct = float('-inf')
        for child_node in node.child_nodes:
            if child_node.uct > best_uct:
                best_uct = child_node.uct
                best_move = child_node
        if best_uct < node.value + uct_constant * (math.sqrt(math.log(node.executed_sims)/1)) and node.left_to_expand != []:
            expand_node(node, simulating_player)
        else:
            walk_tree(best_move, simulating_player)
            sum_value = 0
            for child_node in node.child_nodes:
                sum_value += child_node.value
            node.value = sum_value
            if node.parent_node == None:
                return
            node.uct = node.value + uct_constant * (math.sqrt(math.log(node.parent_node.executed_sims)/node.executed_sims))


def expand_node(node: Node, simulating_player: BoardPiece):
    move = random.choice(node.left_to_expand)
    node.left_to_expand = [x for x in node.left_to_expand if x != move]
    applied_board = apply_player_action(node.board, move, node.player, True)
    next_player = PLAYER1 if node.player == PLAYER2 else PLAYER2
    value = None
    if node.win_state == None:
        child_win_state = None
        value = predict_game_result(applied_board, next_player, simulating_player)
        if connected_four(applied_board, node.player, move):
            if node.player == simulating_player:
                child_win_state = 1
                value = 1
            else:
                child_win_state = -1
                value = -1
    else:
        child_win_state = node.win_state
        value = node.win_state
    uct = value + uct_constant * (math.sqrt(math.log(node.executed_sims) /1))
    valid = []
    for column in range(applied_board.shape[1]):
        if applied_board[0, column] == NO_PLAYER:
            valid.append(column)
    new_node = Node(applied_board, next_player, value, 1, move, uct, valid, child_win_state, node)
    node.child_nodes.append(new_node)



# simulates game with random moves until win or draw. RETURNS: 0 for draw, 1 for win, -1 for loss
def predict_game_result(board: np.ndarray, player: BoardPiece, simulating_player: BoardPiece):
    copied_board = board.copy()
    last_player = player
    action = -1
    game_still_playing = True
    if connected_four(board, PLAYER1):
        if simulating_player == PLAYER1:
            return 1
        else:
            return -1
    elif connected_four(board, PLAYER2):
        if simulating_player == PLAYER2:
            return 1
        else:
            return -1

    flat_board = copied_board.flatten()
    input_board = pd.DataFrame(flat_board).T.replace(2, -1).astype(float)
    prediction = model.predict(input_board).round()
    print(prediction)
    if prediction[0, 0] == 1:
        return 1 if simulating_player == 2 else -1
    elif prediction[0, 2] == 1:
        return 1 if simulating_player == 1 else -1
    else:
        return 0


def possible_moves(board: np.ndarray):
    not_filled = []
    for x in range(0, 7):
        if board[0, x] == NO_PLAYER:
            not_filled.append(x)
    if len(not_filled) == 0:
        return []
    return not_filled

