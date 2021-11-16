import random as random

from typing import Optional, Tuple
import numpy as np
from agents.common import BoardPiece, SavedState, PlayerAction, check_end_state ,NO_PLAYER, PLAYER1, PLAYER2, apply_player_action, GameState, connected_four



def generate_move_minimax(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    depth = 4
    # max part
    if player == PLAYER1:
        optimal_move = float("-inf")
        options = [random_not_filled_column(board)]
        #unnötig
        for x in range(0,7):
            if board[0, x] == NO_PLAYER:
                current_move = max(board, depth, x)
                print(current_move)

                if current_move > optimal_move:
                    optimal_move = current_move
                    options = []
                    options.append(x)
                elif current_move == optimal_move:
                    options.append(x)
        print(optimal_move)
        print(options)
        return random.choice(options),None
    # min part
    else:
        optimal_move = float("inf")
        options = [random_not_filled_column(board)]
        for x in range(0,7):
            if board[0, x] == NO_PLAYER:
                current_move = min(board, depth, x)
                print(current_move)

                if current_move < optimal_move:
                    optimal_move = current_move
                    options = []
                    options.append(x)
                elif current_move == optimal_move:
                    options.append(x)
        print(optimal_move)
        print(options)
        return random.choice(options),None


def random_not_filled_column(board: np.ndarray):
    not_filled = []
    for x in range(0,7):
        if board[0, x] == NO_PLAYER:
            not_filled.append(x)
    return random.choice(not_filled)

def new_max(board: np.ndarray, depth, action):
    if depth > 0 and check_end_state(board, PLAYER2, action) == GameState.STILL_PLAYING:
        current_max = float("-inf")
        options = [random_not_filled_column(board)]
        for i in range(0, 7):
            if board[0,i] == NO_PLAYER:
                print(i)
                new_board = apply_player_action(board, i, PLAYER1)
                min = new_min(new_board, depth-1, i)
                if min > current_max:
                    current_max = min
                    options = []
                    options.append(i)
                elif min == current_max:
                    options.append(i)
        print(options)
        #return random.choice(options)
        return current_max
    else:
        return new_heuristic(board, PLAYER2, action)

def new_min(board: np.ndarray, depth, action):
    if depth > 0 and check_end_state(board, PLAYER1, action) == GameState.STILL_PLAYING:
        current_min = float("inf")
        options = [random_not_filled_column(board)]
        for i in range(0, 7):
            if board[0,i] == NO_PLAYER:
                print(i)

                new_board = apply_player_action(board, i, PLAYER2)
                max = new_max(new_board, depth-1, i)
                if max < current_min:
                    current_min = max
                    options = []
                    options.append(i)
                elif max == current_min:
                    options.append(i)
        #print(options)
        #return random.choice(options)
        return current_min
    else:
        return new_heuristic(board, PLAYER1, action)






def max(board: np.ndarray, depth, action):
    if(board[0,action] != NO_PLAYER):
        return float("-inf")
    applied_board = apply_player_action(board, action, PLAYER1, True)
    if depth == 0 or connected_four(applied_board, PLAYER1, action):
        return new_heuristic(applied_board, PLAYER1, action)
    else:
        optimal_move = float("-inf")
        for x in range(0,7):
            if board[0, x] == NO_PLAYER:
                current_move = min(applied_board, depth-1, x)
                if current_move > optimal_move:
                    optimal_move = current_move

        return optimal_move

def min(board: np.ndarray, depth, action):
    if (board[0, action] != NO_PLAYER):
        return float("inf")
    applied_board = apply_player_action(board, action, PLAYER2, True)
    if depth == 0 or connected_four(applied_board, PLAYER2, action):
        return new_heuristic(applied_board, PLAYER2, action)
    else:
        optimal_move = float("inf")
        for x in range(0,7):
            if board[0, x] == NO_PLAYER:
                current_move = max(applied_board, depth-1, x)
                if current_move < optimal_move:
                    optimal_move = current_move
        return optimal_move


def new_heuristic(board: np.ndarray, player: BoardPiece, last_action: PlayerAction):
    if connected_four(board, player, last_action) and player == PLAYER1:
        return float('inf')
    elif connected_four(board, player, last_action) and player == PLAYER2:
        return float('-inf')
    else:
        score = 0

        score = score + new_check_horizontal(board) + new_check_vertical(board) + new_check_diagonal_pos(
            board) + new_check_diagonal_neg(board)

        return score


def evaluate_window_dict(finds: dict):

    #1 spielstein 0
    #print(finds)
    if(finds["Player1"] == 0):
        #print(-finds["Player2"])
        #return -finds["Player2"]**4
        return  -((finds["Player2"]-1)**10)
    elif(finds["Player2"] == 0):
        #print(finds["Player1"])
        #return finds["Player1"]**4
        return ((finds["Player1"] - 1) ** 10)
    else:
        #print(0)
        return 0



def iterate_window(board: np.ndarray, yPos: int, xPos: int, yDir: int, xDir: int):
    finds = {
        "Player1": 0,
        "Player2": 0,
        "NoPlayer": 0
    }
    for i in range(0, 4):
        piece = board[yPos + (i*yDir), xPos + (i*xDir)]
        if (piece == PLAYER1):
            finds["Player1"] = finds["Player1"] + 1
        elif (piece == PLAYER2):
            finds["Player2"] = finds["Player2"] + 1
        else:
            finds["NoPlayer"] = finds["NoPlayer"] + 1

    return evaluate_window_dict(finds)


def new_check_horizontal(board: np.ndarray):
    #print("horizontal")
    score = 0
    for y in range(board.shape[0]-1):
        #print("neue Zeile")
        for x in range(board.shape[1]-3):
            score = score + iterate_window(board, (y-(board.shape[0]-1))*-1, x, 0, 1)
    return score

def new_check_vertical(board: np.ndarray):
    #print("vertical")
    score = 0
    for x in range(board.shape[1]-1):
        #print("neue Spalte")
        for y in range(board.shape[0]-3):
            score = score + iterate_window(board, (y-(board.shape[0]-1))*-1, x, -1, 0)
    return score


def new_check_diagonal_pos(board):
    #print("diagonal pos")
    score = 0
    xStart = 6
    yStart = 4

    while xStart >= 0:
        x = xStart
        y = 5
        #print("start von " + str(xStart))
        while x+3 <= 6 and y-3 >= 0:

            score = score + iterate_window(board, y, x, -1, +1)
            x = x + 1
            y = y - 1
        xStart = xStart - 1

    while yStart >= 0:
        x = 0
        y = yStart
        #print("start von " + str(yStart))
        while x+3 <= 6 and y-3 >= 0:
            score = score + iterate_window(board, y, x, -1, +1)
            x = x + 1
            y = y - 1
        yStart = yStart - 1
    return score


def new_check_diagonal_neg(board: np.ndarray):
    #print("diagonal neg")
    score = 0
    xStart = 0
    yStart = 4

    while xStart <= 6:
        x = xStart
        y = 5
        #print("start von " + str(xStart))
        while x-3 >= 0 and y-3 >= 0:
            score = score + iterate_window(board, y, x, -1, -1)
            x = x - 1
            y = y - 1
        xStart = xStart + 1

    while yStart >= 0:
        x = 6
        y = yStart
        #print("start von " + str(yStart))
        while x-3 >= 0 and y-3 >= 0:
            score = score + iterate_window(board, y, x, -1, -1)
            x = x - 1
            y = y - 1
        yStart = yStart - 1
    return score





#verbessern mit checks nach freien Feldern!!!
def heuristic(board: np.ndarray, player: BoardPiece, last_action: PlayerAction):
    if check_end_state(board, player, last_action) and player == PLAYER1:
        return float('inf')
    elif check_end_state(board, player, last_action) and player == PLAYER2:
        return float('-inf')
    else:
        score = 0

        score = score + check_horizontal(board) + check_vertical(board) + check_diagonal_pos(board) + check_diagonal_neg(board)

        return score

def calculate_streak(x, y, board, last_checked, score, streak):

    if board[y, x] == last_checked and last_checked != NO_PLAYER:
        streak = streak * 2
    elif board[y, x] == NO_PLAYER:
        if last_checked == PLAYER1:
            score = score + streak
        else:
            score = score + streak * (-1)
        streak = 0
    else:
        last_checked == board[y, x]
        streak = 1

    return last_checked, score, streak



def check_horizontal(board: np.ndarray):

    last_checked: BoardPiece = NO_PLAYER
    score = 0
    streak = 0
    for y in range(board.shape[0]-1):
        for x in range(board.shape[1]-1):
            res = calculate_streak(x, y, board, last_checked, score, streak)
            last_checked = res[0]
            score = res[1]
            streak = res[2]

    return score


def check_vertical(board: np.ndarray):
    last_checked: BoardPiece = NO_PLAYER
    score = 0
    streak = 0

    for x in range(board.shape[1] - 1):
        for y in reversed(range(board.shape[0] - 1)):

            res = calculate_streak(x, y, board, last_checked, score, streak)
            last_checked = res[0]
            score = res[1]
            streak = res[2]

    return score


def check_diagonal_pos(board: np.ndarray):

    last_checked: BoardPiece = NO_PLAYER
    score = 0
    streak = 0

    xStart = 6
    yStart = 5

    while xStart >= 0:
        x = xStart
        y = 5
        while x <= 6 and y >= 0:
            res = calculate_streak(x, y, board, last_checked, score, streak)
            last_checked = res[0]
            score = res[1]
            streak = res[2]

            x = x + 1
            y = y - 1

        xStart = xStart - 1

    while yStart >= 0:
        x = 0
        y = yStart
        while x <= 6 and y >= 0:
            res = calculate_streak(x, y, board, last_checked, score, streak)
            last_checked = res[0]
            score = res[1]
            streak = res[2]

            x = x + 1
            y = y - 1

        yStart = yStart - 1

    return score



def check_diagonal_neg(board: np.ndarray):

    last_checked: BoardPiece = NO_PLAYER
    score = 0
    streak = 0

    xStart = 0
    yStart = 5

    while xStart <= 6:
        x = xStart
        y = 5
        while x >= 0 and y >= 0:
            res = calculate_streak(x, y, board, last_checked, score, streak)
            last_checked = res[0]
            score = res[1]
            streak = res[2]

            x = x - 1
            y = y - 1

        xStart = xStart + 1

    while yStart >= 0:
        x = 6
        y = yStart
        while x >= 0 and y >= 0:
            res = calculate_streak(x, y, board, last_checked, score, streak)
            last_checked = res[0]
            score = res[1]
            streak = res[2]

            x = x - 1
            y = y - 1

        yStart = yStart - 1

    return score


