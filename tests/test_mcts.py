import numpy as np
from agents.common import NO_PLAYER, BoardPiece, PLAYER2, PLAYER1, string_to_board
from agents.agent_mcts.mcts import walk_tree, Node, random_game_result, select_best_move, board_winner

initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
initialBoard.fill(NO_PLAYER)

def test_random_game_result():

    initialBoard[5, 4] = PLAYER2
    initialBoard[4, 4] = PLAYER2
    initialBoard[3, 4] = PLAYER1
    initialBoard[2, 4] = PLAYER2

    initialBoard[5, 3] = PLAYER1
    initialBoard[4, 3] = PLAYER1
    initialBoard[3, 3] = PLAYER2

    initialBoard[5, 2] = PLAYER2
    initialBoard[4, 2] = PLAYER2
    initialBoard[3, 2] = PLAYER1

    initialBoard[5, 1] = PLAYER1
    initialBoard[4, 1] = PLAYER1
    initialBoard[3, 1] = PLAYER1
    initialBoard[2, 1] = PLAYER1

    initialBoard[5, 0] = PLAYER2

    assert random_game_result(initialBoard, PLAYER1, PLAYER2) == -1

def test_random_game_result2():

    initialBoard[5, 4] = PLAYER2
    initialBoard[4, 4] = PLAYER2
    initialBoard[3, 4] = PLAYER1
    initialBoard[2, 4] = PLAYER2

    initialBoard[5, 3] = PLAYER1
    initialBoard[4, 3] = PLAYER1
    initialBoard[3, 3] = PLAYER2

    initialBoard[5, 2] = PLAYER2
    initialBoard[4, 2] = PLAYER2
    initialBoard[3, 2] = PLAYER1

    initialBoard[5, 1] = PLAYER1
    initialBoard[4, 1] = PLAYER1
    initialBoard[3, 1] = PLAYER1
    initialBoard[2, 1] = PLAYER1

    initialBoard[5, 0] = PLAYER2

    assert random_game_result(initialBoard, PLAYER1, PLAYER1) == 1

def test_random_game_result_loose():
    copy_board = initialBoard

    copy_board[5, 3] = PLAYER2
    copy_board[4, 3] = PLAYER2
    copy_board[3, 3] = PLAYER2

    copy_board[5, 2] = PLAYER1
    copy_board[4, 2] = PLAYER1

    copy_board[5, 4] = PLAYER1
    copy_board[4, 4] = PLAYER1

    sum = 0
    for i in range(99):
        sum += random_game_result(copy_board, PLAYER2, PLAYER1)
    print(sum)

    assert sum < 0


def test_random_game_result_win():
    copy_board = initialBoard

    copy_board[5, 3] = PLAYER2
    copy_board[4, 3] = PLAYER2
    copy_board[3, 3] = PLAYER2

    copy_board[5, 2] = PLAYER1
    copy_board[4, 2] = PLAYER1

    copy_board[5, 4] = PLAYER1
    copy_board[4, 4] = PLAYER1

    sum = 0
    for i in range(99):
        sum += random_game_result(copy_board, PLAYER2, PLAYER2)
    print(sum)


def test_board_winner():

    initialBoard[5, 4] = PLAYER2
    initialBoard[4, 4] = PLAYER2
    initialBoard[3, 4] = PLAYER1
    initialBoard[2, 4] = PLAYER2

    initialBoard[5, 3] = PLAYER1
    initialBoard[4, 3] = PLAYER1
    initialBoard[3, 3] = PLAYER2

    initialBoard[5, 2] = PLAYER2
    initialBoard[4, 2] = PLAYER2
    initialBoard[3, 2] = PLAYER1

    initialBoard[5, 1] = PLAYER1
    initialBoard[4, 1] = PLAYER1
    initialBoard[3, 1] = PLAYER1
    initialBoard[2, 1] = PLAYER1

    initialBoard[5, 0] = PLAYER2

    assert board_winner(initialBoard, PLAYER1) == 1

def test_board_loser():

    initialBoard[5, 4] = PLAYER2
    initialBoard[4, 4] = PLAYER2
    initialBoard[3, 4] = PLAYER1
    initialBoard[2, 4] = PLAYER2

    initialBoard[5, 3] = PLAYER1
    initialBoard[4, 3] = PLAYER1
    initialBoard[3, 3] = PLAYER2

    initialBoard[5, 2] = PLAYER2
    initialBoard[4, 2] = PLAYER2
    initialBoard[3, 2] = PLAYER1

    initialBoard[5, 1] = PLAYER1
    initialBoard[4, 1] = PLAYER1
    initialBoard[3, 1] = PLAYER1
    initialBoard[2, 1] = PLAYER1

    initialBoard[5, 0] = PLAYER2

    assert board_winner(initialBoard, PLAYER2) == -1