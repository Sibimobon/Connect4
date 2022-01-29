import random as random

from typing import Optional, Tuple
import numpy as np

from agents.common import BoardPiece, SavedState, PlayerAction, check_end_state, NO_PLAYER, PLAYER1, PLAYER2,\
    apply_player_action, GameState, connected_four


def generate_move_minimax(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Tries to calculate the optimal move.

    :param board: the board to calculate the move for
    :param player: the player to calculate the move for
    :param saved_state: a optional game state for saving the result of previous calculations
    :return: a column to be played, an optional game result (here always None)
    """
    depth = 6
    optimal_move = float("-inf")
    options = [random_not_filled_column(board)]
    for x in range(board.shape[1]):
        if board[0, x] == NO_PLAYER:
            applied_board = apply_player_action(board, x, player, True)
            if connected_four(applied_board, player, x):
                return x, None
            if player == PLAYER1:
                current_move = min(board, depth-1, x, float('-inf'), float('inf'))
            else:
                current_move = max(board, depth-1, x, float('-inf'), float('inf'))*-1
            print(current_move)

            if current_move > optimal_move:
                optimal_move = current_move
                options = [x]
            elif current_move == optimal_move:
                options.append(x)
    print(optimal_move)
    print(options)
    return random.choice(options), None


def max(board: np.ndarray, depth, action, alpha, beta):
    """
    Is called recursively by the min function to determine the best move from the maximizing player's perspective.
    The evaluated moves are taken from the min function.

    :param board: the state of the board to be modified, then evaluated
    :param depth: the maximum depth of recursion
    :param action: the action to be applied to the board
    :param alpha: the current best for the maximizing player - used for cutoffs
    :param beta: the current best for the minimizing player - used for cutoffs
    :return: the best value from the next moves of the min player, the heuristic value at the end of recursion or -inf
             for impossible moves
    """
    if board[0, action] != NO_PLAYER:
        return float("inf")
    applied_board = apply_player_action(board, action, PLAYER2, True)
    if depth <= 0 or connected_four(applied_board, PLAYER2, action):
        return heuristic(applied_board, PLAYER2, action)
    else:
        optimal_value = float("-inf")
        for x in range(0, 7):
            if board[0, x] == NO_PLAYER:
                current_value = min(applied_board, depth-1, x, alpha, beta)
                if current_value > optimal_value:
                    optimal_value = current_value
                alpha = current_value if current_value > alpha else alpha
                if beta <= alpha:
                    break
        return optimal_value


def min(board: np.ndarray, depth, action, alpha, beta):
    """
    Is called recursively by the max function to determine the best move from the minimizing player's perspective.
    The evaluated moves are taken from the max function.

    :param board: the state of the board to be modified, then evaluated
    :param depth: the maximum depth of recursion
    :param action: the action to be applied to the board
    :param alpha: the current best for the maximizing player - used for cutoffs
    :param beta: the current best for the minimizing player - used for cutoffs
    :return: the best value from the next moves of the max player, the heuristic value at the end of recursion or -inf
             for impossible moves
    """
    if board[0, action] != NO_PLAYER:
        return float("-inf")
    applied_board = apply_player_action(board, action, PLAYER1, True)
    if depth <= 0 or connected_four(applied_board, PLAYER1, action):
        return heuristic(applied_board, PLAYER1, action)
    else:
        optimal_value = float("inf")
        for x in range(0, 7):
            if board[0, x] == NO_PLAYER:
                current_value = max(applied_board, depth-1, x, alpha, beta)
                if current_value < optimal_value:
                    optimal_value = current_value
                beta = current_value if current_value < beta else beta
                if beta <= alpha:
                    break
        return optimal_value


def random_not_filled_column(board: np.ndarray):
    """
    Get a column in a board where a piece can still be dropped.
    :param board: the board
    :return: integer from 0-6
    """
    not_filled = []
    for x in range(0, 7):
        if board[0, x] == NO_PLAYER:
            not_filled.append(x)
    if len(not_filled) == 0:
        return -1
    return random.choice(not_filled)


def heuristic(board: np.ndarray, player: Optional[BoardPiece] = NO_PLAYER, last_action: Optional[PlayerAction] = None):
    """
    Evaluate a board state from a players perspective.
    :param board: the board state
    :param player: the player to evaluate the board for
    :param last_action: the action performed to reach the current board state
    :return: a number from -inf to inf
    """
    if last_action is not None:
        if connected_four(board, player, last_action) and player == PLAYER1:
            return float('inf')
        elif connected_four(board, player, last_action) and player == PLAYER2:
            return float('-inf')
    else:
        if connected_four(board, player) and player == PLAYER1:
            return float('inf')
        elif connected_four(board, player) and player == PLAYER2:
            return float('-inf')
    score = 0
    score = score + check_horizontal(board) + check_vertical(board) + check_diagonal_pos(
        board) + check_diagonal_neg(board)
    return score


def iterate_window(board: np.ndarray, y_pos: int, x_pos: int, y_dir: int, x_dir: int):
    """
    Evaluates a window of four, rating the chance for a player to win in said window.

    :param board: the board which contains the window
    :param y_pos: the starting position on the y-axis. Bottom is 5, top is 0.
    :param x_pos: the starting position on the x-axis. Left is 0, right is 6.
    :param y_dir: the direction to draw the window from on the y axis. -1 for up, 1 for down, 0 for a horizontal window
    :param x_dir: the direction to draw the window from on the x axis. -1 for left, 1 for right, 0 for a vertical window
    :return: a integer value scaled by the number of pieces for a player in the window
             (positive for Player1 - negative for Player2)
    """
    finds = {
        "Player1": 0,
        "Player2": 0,
        "NoPlayer": 0
    }
    for i in range(0, 4):
        piece = board[y_pos + (i*y_dir), x_pos + (i*x_dir)]
        if piece == PLAYER1:
            finds["Player1"] = finds["Player1"] + 1
        elif piece == PLAYER2:
            finds["Player2"] = finds["Player2"] + 1
        else:
            finds["NoPlayer"] = finds["NoPlayer"] + 1
    if finds["Player1"] == 0:
        return -((finds["Player2"]-1)**10)
    elif finds["Player2"] == 0:
        return (finds["Player1"] - 1)**10
    else:
        return 0


def check_horizontal(board: np.ndarray):
    """
    Iterates over a board horizontally for each row, evaluating a window of four for each column, giving a score of the
    player's likeliness to win.

    :param board: The board to be iterated.
    :return: A cumulative score of all the windows.
    """
    score = 0
    for y in range(board.shape[0]-1):
        for x in range(board.shape[1]-3):
            score = score + iterate_window(board, (y-(board.shape[0]-1))*-1, x, 0, 1)
    return score


def check_vertical(board: np.ndarray):
    """
    Iterates over a board vertically for each column, evaluating a window of four for each row, giving a score of the
    player's likeliness to win.

    :param board: The board to be iterated.
    :return: A cumulative score of all the windows.
    """
    score = 0
    for x in range(board.shape[1]-1):
        for y in range(board.shape[0]-3):
            score = score + iterate_window(board, (y-(board.shape[0]-1))*-1, x, -1, 0)
    return score


def check_diagonal_pos(board):
    """
    Iterates over a board diagonally, evaluating all windows with a positive incline, giving a score of the
    player's likeliness to win.

    :param board: The board to be iterated.
    :return: A cumulative score of all the windows.
    """
    score = 0
    x_start = 6
    y_start = 4
    while x_start >= 0:
        x = x_start
        y = 5
        while x+3 <= 6 and y-3 >= 0:
            score = score + iterate_window(board, y, x, -1, +1)
            x = x + 1
            y = y - 1
        x_start = x_start - 1
    while y_start >= 0:
        x = 0
        y = y_start
        while x+3 <= 6 and y-3 >= 0:
            score = score + iterate_window(board, y, x, -1, +1)
            x = x + 1
            y = y - 1
        y_start = y_start - 1
    return score


def check_diagonal_neg(board: np.ndarray):
    """
    Iterates over a board diagonally, evaluating all windows with a negative incline, giving a score of the
    player's likeliness to win.

    :param board: The board to be iterated.
    :return: A cumulative score of all the windows.
    """
    score = 0
    x_start = 0
    y_start = 4
    while x_start <= 6:
        x = x_start
        y = 5
        while x-3 >= 0 and y-3 >= 0:
            score = score + iterate_window(board, y, x, -1, -1)
            x = x - 1
            y = y - 1
        x_start = x_start + 1
    while y_start >= 0:
        x = 6
        y = y_start
        while x-3 >= 0 and y-3 >= 0:
            score = score + iterate_window(board, y, x, -1, -1)
            x = x - 1
            y = y - 1
        y_start = y_start - 1
    return score
