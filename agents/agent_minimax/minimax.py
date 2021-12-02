import random as random

from typing import Optional, Tuple
import numpy as np

from agents.common import BoardPiece, SavedState, PlayerAction, check_end_state ,NO_PLAYER, PLAYER1, PLAYER2, apply_player_action, GameState, connected_four


#standard max min
#def generate_move_minimax(
#    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
#) -> Tuple[PlayerAction, Optional[SavedState]]:
#    depth = 4
#    # max part
#    if player == PLAYER1:
#        optimal_move = float("-inf")
#        options = [random_not_filled_column(board)]
#        #unnötig
#        for x in range(0,7):
#            if board[0, x] == NO_PLAYER:
#                current_move = max(board, depth, x)
#                print(current_move)
#
#                if current_move > optimal_move:
#                    optimal_move = current_move
#                    options = []
#                    options.append(x)
#                elif current_move == optimal_move:
#                    options.append(x)
#        print(optimal_move)
#        print(options)
#        return random.choice(options),None
#    # min part
#    else:
#        optimal_move = float("inf")
#        options = [random_not_filled_column(board)]
#        for x in range(0,7):
#            if board[0, x] == NO_PLAYER:
#                current_move = min(board, depth, x)
#                print(current_move)
#
#                if current_move < optimal_move:
#                    optimal_move = current_move
#                    options = []
#                    options.append(x)
#                elif current_move == optimal_move:
#                    options.append(x)
#        print(optimal_move)
#        print(options)
#        return random.choice(options),None
#

#alphabeta
#def generate_move_minimax(
#    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
#) -> Tuple[PlayerAction, Optional[SavedState]]:
#
#    depth = 4
#    best_move = -1
#    max_player = True if player == PLAYER1 else False
#    current_best = float('-inf') if max_player else float('inf')
#    for x in range(board.shape[1]):
#        applied_board = apply_player_action(board, x, player)
#        temp = alphabeta(board, depth, float('-inf'), float('inf'), not max_player)
#        print(temp)
#        if max_player:
#            if temp > current_best:
#                current_best = temp
#                best_move = x
#        else:
#            if temp < current_best:
#                current_best = temp
#                best_move = x
#    return best_move, None
#

#new max/min
def generate_move_minimax(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    depth = 6
    depth = 1 if depth < 1 else depth
    # max part
    if player == PLAYER1:
        optimal_move = float("-inf")
        options = [random_not_filled_column(board)]
        #unnötig
        for x in range(board.shape[1]):
            if board[0, x] == NO_PLAYER:
                applied_board = apply_player_action(board, x, PLAYER1, True)
                if connected_four(applied_board, PLAYER1, x):
                    return (x, None)
                current_move = new_min(board, depth-1, x, float('-inf'), float('inf'))
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
        for x in range(board.shape[1]):
            if board[0, x] == NO_PLAYER:
                applied_board = apply_player_action(board, x, PLAYER2, True)
                if connected_four(applied_board, PLAYER2, x):
                    return (x, None)
                current_move = new_max(board, depth-1, x, float('-inf'), float('inf'))
                print(current_move)
                #print(x)

                if current_move < optimal_move:
                    optimal_move = current_move
                    options = []
                    options.append(x)
                elif current_move == optimal_move:
                    options.append(x)
        print(optimal_move)
        print(options)
        return random.choice(options),None





def new_max(board: np.ndarray, depth, action, alpha, beta):
    if(board[0,action] != NO_PLAYER):
        return float("inf")
    applied_board = apply_player_action(board, action, PLAYER2, True)
    if depth <= 0 or connected_four(applied_board, PLAYER2, action):
        return new_heuristic(applied_board, PLAYER2, action)
    else:
        optimal_value = float("-inf")
        for x in range(0,7):
            if board[0, x] == NO_PLAYER:
                current_value = new_min(applied_board, depth-1, x, alpha, beta)
                #print(current_value)
                if current_value > optimal_value:
                    optimal_value = current_value
                alpha = current_value if current_value > alpha else alpha
                if beta <= alpha:
                    break
        return optimal_value

def new_min(board: np.ndarray, depth, action, alpha, beta):
    if (board[0, action] != NO_PLAYER):
        return float("-inf")
    applied_board = apply_player_action(board, action, PLAYER1, True)
    if depth <= 0 or connected_four(applied_board, PLAYER1, action):
        return new_heuristic(applied_board, PLAYER1, action)
    else:
        optimal_value = float("inf")
        #print('start interation')
        for x in range(0,7):
            if board[0, x] == NO_PLAYER:
                current_value = new_max(applied_board, depth-1, x, alpha, beta)
                #print(current_value)
                if current_value < optimal_value:
                    #print('switched'+str(optimal_value)+"with"+str(current_value))
                    optimal_value = current_value
                beta = current_value if current_value < beta else beta
                if beta <= alpha:
                    break
        #print(optimal_value)
        return optimal_value


def generate_move_negamax(
    board: np.ndarray, player: BoardPiece
) -> Tuple[PlayerAction, Optional[SavedState]]:
    best_move = 0
    for x in range(len(board[1])-1):
        curr_val = negamax(board, player, 4, float('-inf'), float('inf'),x)
        if curr_val > best_move:
            best_move = curr_val
    return best_move

def negamax(board: np.ndarray, player: BoardPiece, depth: int, alpha: int, beta:int, last_action: int):
    if depth == 0 or random_not_filled_column(board) == -1:
        return new_heuristic(board, player, last_action)
    max_value = alpha;
    for x in range(0, 7):
        if board[0, x] == NO_PLAYER:
            boardCopy = apply_player_action(board,x,player,True)
            next_player = PLAYER1 if player == PLAYER2 else PLAYER2
            value = -negamax(board, next_player, depth-1, -beta, -max_value, x)
            if value > max_value:
                max_value = value
                if(max_value>= beta):
                    break
    return max_value


def random_not_filled_column(board: np.ndarray):
    not_filled = []
    for x in range(0,7):
        if board[0, x] == NO_PLAYER:
            not_filled.append(x)
    if len(not_filled) == 0:
        return -1
    return random.choice(not_filled)

def alphabeta(board: np.ndarray, depth: int, alpha: int, beta: int, is_max: bool):
    if is_max:
        if depth == 0:
            return new_heuristic(board, PLAYER2)
        value = alpha
        for x in range(board.shape[1]):
            if board[0, x] != NO_PLAYER:
                return float('-inf')
            temp_board = apply_player_action(board, x, PLAYER1, True)
            current_value = alphabeta(temp_board, depth-1, value, beta, False)
            if current_value > value:
                value = current_value
            if value >= beta:
                break
        return value
    else:
        if depth == 0:
            return new_heuristic(board, PLAYER1)
        value = beta
        for x in range(board.shape[1]):
            if board[0, x] != NO_PLAYER:
                return float('-inf')
            temp_board = apply_player_action(board, x, PLAYER2, True)
            current_value = alphabeta(temp_board, depth-1, alpha, value, True)
            if current_value < value:
                value = current_value
            if value <= alpha:
                break
        return value

#change to max
def old_max(board: np.ndarray, depth, action):
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
#change to min
def old_min(board: np.ndarray, depth, action):
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


def new_heuristic(board: np.ndarray, player: Optional[BoardPiece] = NO_PLAYER, last_action: Optional[PlayerAction] = None):
    if last_action != None:
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
        #start from every column where window can fit to left
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


