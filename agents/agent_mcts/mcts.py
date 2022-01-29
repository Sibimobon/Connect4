import numpy as np
from agents.common import BoardPiece, apply_player_action, connected_four, PLAYER1, PLAYER2, NO_PLAYER, PlayerAction, SavedState
from typing import Optional, Tuple
import random, time, math
# import time


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


def generate_move_mcts(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    time_limit = 5  # time in sec usually is 5
    running_until = time.time() + time_limit
    tree = Node(board, player, 1, 1, None, None, possible_moves(board), None, None)
    iterations = 0
    running = True
    while time.time() < running_until:
        if running:
            iterations += 1

            running = False
            walk_tree(tree, player)
            running = True
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
    if node.left_to_expand:
        expand_node(node, simulating_player)
        return
    if node.win_state is not None:
        node.value += node.win_state
        # update uct value!!!
        node.uct = calculate_uct(node)
        return
    if (not node.child_nodes) and (not node.left_to_expand):
        node.value -= 1
        # update uct value!!!
        node.uct = calculate_uct(node)
        return
    else:
        # select best node
        best_child = select_best_uct(node)
        walk_tree(best_child, simulating_player)
        update_node(node)
        if node.parent_node is None:
            return


def calculate_uct(node):
    # uct_constant = 7 works kind of well
    uct_constant = 10000
    return node.value + uct_constant * (math.sqrt(math.log(node.parent_node.executed_sims)/node.executed_sims))


def select_best_uct(node):
    best_move = None
    best_uct = float('-inf')
    for child_node in node.child_nodes:
        if child_node.uct > best_uct:
            best_uct = child_node.uct
            best_move = child_node
    return best_move


def update_node(node):
    sum_value = 0
    for child_node in node.child_nodes:
        sum_value += child_node.value
    node.value = sum_value
    if node.parent_node is not None:
        node.uct = calculate_uct(node)


def expand_node(node: Node, simulating_player: BoardPiece):
    move = random.choice(node.left_to_expand)
    node.left_to_expand = [x for x in node.left_to_expand if x != move]
    applied_board = apply_player_action(node.board, move, node.player, True)
    next_player = PLAYER1 if node.player == PLAYER2 else PLAYER2
    value = None
    if node.win_state is None:
        child_win_state = None
        value = random_game_result(applied_board, next_player, simulating_player)
        game_ending = board_winner(applied_board, node.player)
        if game_ending != 0:
            child_win_state = game_ending
            value = game_ending
    else:
        child_win_state = node.win_state
        value = node.win_state
    valid = possible_moves(applied_board)
    new_node = Node(applied_board, next_player, value, 1, move, 0, valid, child_win_state, node)
    new_node.uct = calculate_uct(new_node)
    node.child_nodes.append(new_node)


# returns 1 if the player wins for this board, -1 if the player looses, 0 if no one wins
def board_winner(board: np.ndarray, simulating_player: BoardPiece):
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
    else:
        return 0


# use 1 1/2 0
# simulates game with random moves until win or draw. RETURNS: 0 for draw, 1 for win, -1 for loss
def random_game_result(board: np.ndarray, player: BoardPiece, simulating_player: BoardPiece):
    game_ending = board_winner(board, simulating_player)
    if game_ending != 0:
        return game_ending

    copied_board = board.copy()
    last_player = player
    action = -1
    game_still_playing = True
    while game_still_playing:
        possible = possible_moves(copied_board)
        if possible == []:
            return 0
        action = random.choice(possible)
        apply_player_action(copied_board, action, last_player, False)
        if connected_four(copied_board, last_player, action):
            game_still_playing = False
            if last_player == simulating_player:
                return 1
            else:
                return -1
        last_player = PLAYER1 if last_player == PLAYER2 else PLAYER2


def possible_moves(board: np.ndarray):
    not_filled = []
    for x in range(0, 7):
        if board[0, x] == NO_PLAYER:
            not_filled.append(x)
    if len(not_filled) == 0:
        return []
    return not_filled

