from enum import Enum
from typing import Optional, Callable, Tuple
import numpy as np
#from numba import jit


BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

CONNECT_N = 4

PlayerAction = np.int8  # The column to be played

class SavedState:
    pass

GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]

class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0



def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    initialBoard = np.ndarray(shape=(6,7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)
    return initialBoard



def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    if board.shape[0]!= 6 or board.shape[1]!= 7:
        raise ValueError
        return 'raise Error!'
    boardString = '\n|==============|\n'
    for y in range(board.shape[0]):
        current_Line = '|'
        for x in range(board.shape[1]):
            to_add = ''
            piece = board.item(y, x)
            if piece == NO_PLAYER:
                to_add = NO_PLAYER_PRINT
            elif piece == PLAYER1:
                to_add = PLAYER1_PRINT
            elif piece == PLAYER2:
                to_add = PLAYER2_PRINT
            else:
                # invalid board
                raise ValueError("Invalid piece on board")
            current_Line += (to_add + " ")
        boardString += (current_Line + '|\n')
    boardString += '|==============|\n' \
                   '|0 1 2 3 4 5 6 |'

    #print(boardString)
    return boardString



def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """

    ret = np.ndarray(shape=(6,7), dtype=BoardPiece)
    split = pp_board.split('\n')

    y = 0
    for i in reversed(range(2, len(split) - 2)):
        temp = split[i].replace('|', '')

        x = 0
        for j in range(0, len(temp) - 1, 2):

            if(temp[j] == NO_PLAYER_PRINT):
                ret[y, x] = NO_PLAYER
            elif(temp[j] == PLAYER1_PRINT):
                ret[y, x] = PLAYER1
            elif(temp[j] == PLAYER2):
                ret[y, x] = PLAYER2
            else:
                raise ValueError
            x = x + 1
        y = y + 1

    #print(ret)

    return ret



def apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """
    if copy:
        temp_board = board.copy()
        for y in reversed(range(board.shape[0])):
            if temp_board[y, action] == 0:
                temp_board[y, action] = player
                return temp_board
    for y in reversed(range(board.shape[0])):
        if board[y, action] == NO_PLAYER:
            board[y, action] = player
            return board
    raise ValueError('Column full')


def connected_four(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """
    if(last_action == None):

        did_win = False
        for x in range(board.shape[1]-1):
            row = 0
            for y in reversed(range(board.shape[0] - 1)):
                temp = board[y, x]
                if temp != 0:
                    row = row + 1
            if board[row, x] == player and not did_win:
                did_win = connected_four(board, player, x)
        return did_win
    else:
        row = 0
        for y in reversed(range(board.shape[0]-1)):
            temp = board[y, last_action]
            if temp != 0:
                row = row + 1

        if check_connect_left_right(board, player, row) or check_connect_top_bottom(board, player, last_action) or check_connect_topleft_bottomright(board, player, last_action, row) or check_connect_topright_bottomleft(board, player, last_action, row):
            return True
    return False

#checks board for horizontal connections at height of last move
#@jit(nopython=True)
def check_connect_left_right(board: np.ndarray, player: BoardPiece, yPos: np.int8):
    counter = 0
    yPos = (yPos-5)*-1
    for x in range(board.shape[1]):
        if board[yPos, x] == player:
            counter = counter + 1
        elif board[yPos, x] != player and counter >= 4:
            pass
        else:
            counter = 0
    if counter >= 4:
        return True
    return False

# checks board for vertical connections at xpos of last action
#@jit(nopython=True)
def check_connect_top_bottom(board: np.ndarray, player: BoardPiece, last_action: PlayerAction):
    counter = 0
    for y in range(board.shape[0]):
        if board[y, last_action] == player:
            counter = counter + 1
        elif board[y, last_action] != player and counter >= 4:
            pass
        else:
            counter = 0
        if counter >= 4:
            return True
    return False

#holds the logic to check in diagonal diections from a point on the edge of the board indicated by xDir and yDir
#@jit(nopython=True)
def diagonal_check(board: np.ndarray, player: BoardPiece, xPos: np.int8, yPos: np.int8, xDir: np.int8, yDir: np.int8):
    counter = 0
    x = xPos
    y = yPos
    while x < board.shape[1] and y < board.shape[0] and x >= 0 and y >= 0:
        #print(str(y)+";"+str(x))
        #print(str(y)+","+str(x)+":"+str(counter))
        if board[y, x] == player:

            counter = counter + 1
            #print(counter)
        elif board[y, x] != player and counter >= 4:

            pass
        else:
            counter = 0
        if counter >= 4:

            return True
        x = x + xDir
        y = y + yDir
    return False

#checks the diagonals with a negative pitch for 4 pieces in a row
#@jit(nopython=True)
def check_connect_topright_bottomleft(board: np.ndarray, player: BoardPiece,  last_action: PlayerAction , yPos: np.int8):

    x = 0
    y = 0
    if last_action >= yPos:
        x = last_action - yPos
        if diagonal_check(board, player, x, (y-5)*-1, +1, -1):
            return True
        return False
    else:
        y = yPos - last_action
        if diagonal_check(board, player, x, (y-5)*-1, +1, -1):
            return True
        return False

#checks the diagonals with a positive pitch for 4 pieces in a row
#@jit(nopython=True)
def check_connect_topleft_bottomright(board: np.ndarray, player: BoardPiece,  last_action: PlayerAction, yPos: np.int8):

    x = 0
    y = 0
    if last_action >= (yPos-5)*-1:
        x = last_action - (yPos-5)*-1
        #print(str((y-5)*-1)+","+str(x))
        if diagonal_check(board, player, x, y, +1, +1):
            return True
        return False
    else:
        y = (yPos-5)*-1 - last_action

        if diagonal_check(board, player, x, y, +1, +1):
            return True
        return False


def check_end_state(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    draw = True


    for x in range(board.shape[1]):
        if board[0, x] == 0:
            draw = False

    if connected_four(board, player, last_action):
        return GameState.IS_WIN
    elif not draw:
        return GameState.STILL_PLAYING
    else:
        return GameState.IS_DRAW


