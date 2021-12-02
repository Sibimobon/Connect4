from agents.agent_minimax.minimax import heuristic, check_horizontal, check_vertical, check_diagonal_pos, check_diagonal_neg, calculate_streak
import numpy as np
from agents.common import NO_PLAYER, BoardPiece, PLAYER2, PLAYER1, string_to_board

def test_check_horizontal_empty():
    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    assert check_horizontal(initialBoard) == 0

def test_check_horizontal_2():
    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 0] = PLAYER1
    initialBoard[5, 1] = PLAYER1

    assert check_horizontal(initialBoard) == 2

def test_eval_window_none():
    from agents.agent_minimax.minimax import evaluate_window_dict

    test_finds = {
        "Player1": 2,
        "Player2": 1,
        "NoPlayer": 1
    }

    assert evaluate_window_dict(test_finds) == 0

def test_eval_window_pos():
    from agents.agent_minimax.minimax import evaluate_window_dict

    test_finds = {
        "Player1": 2,
        "Player2": 0,
        "NoPlayer": 2
    }

    assert evaluate_window_dict(test_finds) == 2

def test_eval_window_neg():
    from agents.agent_minimax.minimax import evaluate_window_dict

    test_finds = {
        "Player1": 0,
        "Player2": 3,
        "NoPlayer": 1
    }

    assert evaluate_window_dict(test_finds) == -3

def test_iterate_window_none():
    from agents.agent_minimax.minimax import iterate_window
    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    assert iterate_window(initialBoard, 5, 0, 0, +1) == 0

def test_iterate_window_pos():
    from agents.agent_minimax.minimax import iterate_window
    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)
    initialBoard[5, 0] = PLAYER1
    initialBoard[5, 1] = PLAYER1

    assert iterate_window(initialBoard, 5, 0, 0, +1) == 2

def test_iterate_window_neg():
    from agents.agent_minimax.minimax import iterate_window
    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)
    initialBoard[5, 0] = PLAYER2
    initialBoard[5, 1] = PLAYER2
    initialBoard[5, 2] = PLAYER2

    assert iterate_window(initialBoard, 5, 0, 0, +1) == -3

def test_new_horizontal():
    from agents.agent_minimax.minimax import new_check_horizontal

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)
    initialBoard[5,0] = PLAYER1

    assert new_check_horizontal(initialBoard) == 1


def test_new_vertical():
    from agents.agent_minimax.minimax import new_check_vertical

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)
    initialBoard[5, 0] = PLAYER1

    assert new_check_vertical(initialBoard) == 1

def test_new_check_diagonal_pos():
    from agents.agent_minimax.minimax import new_check_diagonal_pos

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)
    initialBoard[5, 0] = PLAYER1
    initialBoard[4, 1] = PLAYER1

    assert new_check_diagonal_pos(initialBoard) == 3


def test_new_check_diagonal_neg():
    from agents.agent_minimax.minimax import new_check_diagonal_neg

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)
    initialBoard[5, 6] = PLAYER1
    initialBoard[4, 5] = PLAYER1

    assert new_check_diagonal_neg(initialBoard) == 3

def test_new_heuristic():
    from agents.agent_minimax.minimax import new_heuristic

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)
    initialBoard[5, 0] = PLAYER1

    assert new_heuristic(initialBoard, PLAYER1, 0) == 3

def test_new_heuristic_compare():
    from agents.agent_minimax.minimax import new_heuristic

    smart_board = np.ndarray(shape=(6,7), dtype=BoardPiece)
    smart_board.fill(NO_PLAYER)

    smart_board[5,0] = PLAYER1
    smart_board[4,0] = PLAYER2
    smart_board[5,1] = PLAYER1

    dumb_board = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    dumb_board.fill(NO_PLAYER)

    dumb_board[5,0] = PLAYER1
    dumb_board[4,0] = PLAYER2
    dumb_board[3,0] = PLAYER1
    print(smart_board)
    print(new_heuristic(smart_board, PLAYER1, 1))
    print(dumb_board)
    print(new_heuristic(dumb_board, PLAYER1, 0))

    assert new_heuristic(smart_board, PLAYER1, 1) > new_heuristic(dumb_board, PLAYER1, 0)

def test_new_heuristic_practical():
    from agents.agent_minimax.minimax import new_heuristic

    smart_board = np.ndarray(shape=(6,7), dtype=BoardPiece)
    smart_board.fill(NO_PLAYER)

    smart_board[5,1] = PLAYER1
    smart_board[4,1] = PLAYER2
    smart_board[3,1] = PLAYER1

    dumb_board = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    dumb_board.fill(NO_PLAYER)

    dumb_board[5,0] = PLAYER1
    dumb_board[4,0] = PLAYER2
    dumb_board[3,0] = PLAYER1
    print(smart_board)
    print(new_heuristic(smart_board, PLAYER1, 1))


    assert new_heuristic(smart_board, PLAYER1, 1) > new_heuristic(dumb_board, PLAYER1, 0)


def test_heuristic_player1_win():
    from agents.agent_minimax.minimax import new_heuristic

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 0] = PLAYER1
    initialBoard[5, 1] = PLAYER1
    initialBoard[5, 2] = PLAYER1
    initialBoard[5, 3] = PLAYER1

    assert new_heuristic(initialBoard, PLAYER1, 2) == float("inf")

def test_heuristic_player2_win():
    from agents.agent_minimax.minimax import new_heuristic

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 0] = PLAYER2
    initialBoard[5, 1] = PLAYER2
    initialBoard[5, 2] = PLAYER2
    initialBoard[5, 3] = PLAYER2

    assert new_heuristic(initialBoard, PLAYER2, 2) == float("-inf")

def test_heuristic_human_start():
    from agents.agent_minimax.minimax import new_heuristic

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 3] = PLAYER1
    initialBoard[4,3] = PLAYER2

    assert new_heuristic(initialBoard, PLAYER2, 3) == float("inf")

def test_heuristic_human_start2_smart():
    from agents.agent_minimax.minimax import new_heuristic

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 3] = PLAYER1
    initialBoard[4 ,3] = PLAYER2
    initialBoard[3, 3] = PLAYER1
    initialBoard[5, 4] = PLAYER2

    assert new_heuristic(initialBoard, PLAYER2, 4) == float("inf")

def test_heuristic_human_start2_dumb():
    from agents.agent_minimax.minimax import new_heuristic

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 3] = PLAYER1
    initialBoard[4 ,3] = PLAYER2
    initialBoard[3, 3] = PLAYER1
    initialBoard[2, 3] = PLAYER2

    assert new_heuristic(initialBoard, PLAYER2, 3) == float("inf")

def test_max_pruning():
    from agents.agent_minimax.minimax import new_max

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 0] = PLAYER2
    initialBoard[5, 1] = PLAYER2
    initialBoard[5, 2] = PLAYER2
    initialBoard[5, 3] = PLAYER2

    assert new_max(initialBoard, 3, 2) == float("-inf")


def test_heuristic_dumb_move():
    from agents.agent_minimax.minimax import new_heuristic

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 6] = PLAYER2
    initialBoard[5, 5] = PLAYER1
    initialBoard[4, 5] = PLAYER2
    initialBoard[5, 4] = PLAYER1
    initialBoard[5, 3] = PLAYER1
    initialBoard[5, 2] = PLAYER1

    assert new_heuristic(initialBoard, PLAYER1, 2) == float("inf")

def test_heuristic_dumb_move2():
    from agents.agent_minimax.minimax import new_heuristic

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[0, 2] = PLAYER1
    initialBoard[1, 3] = PLAYER1
    initialBoard[2, 4] = PLAYER1

    assert new_heuristic(initialBoard, PLAYER1, 2) != float("inf")


def test_heuristic_dumb_move3():
    from agents.agent_minimax.minimax import new_heuristic
    from agents.common import apply_player_action

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)


    initialBoard[1, 3] = PLAYER1
    initialBoard[2, 4] = PLAYER1

    initialBoard[1, 2] = PLAYER2
    initialBoard[2, 2] = PLAYER2
    initialBoard[3, 2] = PLAYER2
    initialBoard[0, 2] = PLAYER2

    assert new_heuristic(initialBoard, PLAYER2, 2) == float("-inf")

def test_heuristic_dumb_move4():
    from agents.agent_minimax.minimax import new_heuristic
    from agents.common import apply_player_action

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)


    initialBoard[1, 3] = PLAYER1
    initialBoard[2, 4] = PLAYER1

    initialBoard[1, 2] = PLAYER2
    initialBoard[2, 2] = PLAYER2
    initialBoard[3, 2] = PLAYER2
    initialBoard[0, 2] = PLAYER2

    assert new_heuristic(initialBoard, PLAYER2, 2) == float("-inf")

def test_heuristic_dumb_move5():
    from agents.agent_minimax.minimax import new_heuristic
    from agents.common import apply_player_action

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)


    initialBoard[5, 5] = PLAYER1
    initialBoard[4, 5] = PLAYER2

    initialBoard[5, 4] = PLAYER2
    initialBoard[4, 4] = PLAYER1
    initialBoard[3, 4] = PLAYER2

    initialBoard[5, 3] = PLAYER1
    initialBoard[4, 3] = PLAYER1
    initialBoard[3, 3] = PLAYER1
    initialBoard[2, 3] = PLAYER2

    initialBoard[5, 2] = PLAYER2
    initialBoard[4, 2] = PLAYER2
    initialBoard[3, 2] = PLAYER1
    initialBoard[2, 2] = PLAYER1

    assert new_heuristic(initialBoard, PLAYER1, 2) == float("inf")

def test_generate_dumb_move1():
    from agents.agent_minimax.minimax import generate_move_minimax

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 5] = PLAYER1

    initialBoard[5, 4] = PLAYER2
    initialBoard[4, 4] = PLAYER1
    initialBoard[3, 4] = PLAYER2

    initialBoard[5, 3] = PLAYER1
    initialBoard[4, 3] = PLAYER1
    initialBoard[3, 3] = PLAYER1
    initialBoard[2, 3] = PLAYER2

    initialBoard[5, 2] = PLAYER2
    initialBoard[4, 2] = PLAYER2
    initialBoard[3, 2] = PLAYER1

    print(initialBoard)

    assert generate_move_minimax(initialBoard, PLAYER2, None) == (2, None)



def test_generate_dumb_move2():
    from agents.agent_minimax.minimax import generate_move_minimax

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)


    initialBoard[1, 3] = PLAYER1
    initialBoard[2, 4] = PLAYER1


    initialBoard[1, 2] = PLAYER2
    initialBoard[2, 2] = PLAYER2
    initialBoard[3, 2] = PLAYER2
    initialBoard[4, 2] = PLAYER2
    initialBoard[5, 2] = PLAYER2


    assert generate_move_minimax(initialBoard, PLAYER1, None) == (2, None)


def test_generate_dumb_move3():
    from agents.agent_minimax.minimax import generate_move_minimax

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)


    initialBoard[5, 6] = PLAYER2
    initialBoard[4, 6] = PLAYER2
    initialBoard[3, 6] = PLAYER1
    initialBoard[2, 6] = PLAYER2

    initialBoard[5, 5] = PLAYER2
    initialBoard[4, 5] = PLAYER1
    initialBoard[3, 5] = PLAYER1
    initialBoard[2, 5] = PLAYER2

    initialBoard[5, 4] = PLAYER1
    initialBoard[4, 4] = PLAYER1

    initialBoard[5, 3] = PLAYER1
    initialBoard[4, 3] = PLAYER1

    initialBoard[5, 2] = PLAYER1

    initialBoard[5, 1] = PLAYER2

    initialBoard[5, 0] = PLAYER2

    assert generate_move_minimax(initialBoard, PLAYER2, None) == (2, None)


def test_heuristic_4():
    from agents.agent_minimax.minimax import new_heuristic

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 4] = PLAYER1
    initialBoard[4, 4] = PLAYER2
    initialBoard[3, 4] = PLAYER2
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

    initialBoard[5, 0] = PLAYER2

    print(initialBoard)

    assert new_heuristic(initialBoard, PLAYER2, 4) != float('-inf')

def test_heuristic_5():
    from agents.agent_minimax.minimax import new_heuristic

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 4] = PLAYER1
    initialBoard[4, 4] = PLAYER2
    initialBoard[3, 4] = PLAYER2
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

    print(initialBoard)

    assert new_heuristic(initialBoard, PLAYER1, 1) == float('inf')

def test_min_depth_0():
    from agents.agent_minimax.minimax import new_min

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 4] = PLAYER1
    initialBoard[4, 4] = PLAYER2
    initialBoard[3, 4] = PLAYER2
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


    initialBoard[5, 0] = PLAYER2

    print(initialBoard)

    assert new_min(initialBoard, 0, 1, float('-inf'), float('inf')) == float('inf')


def test_max_depth_1():
    from agents.agent_minimax.minimax import new_max

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 4] = PLAYER1
    initialBoard[4, 4] = PLAYER2
    initialBoard[3, 4] = PLAYER2

    initialBoard[5, 3] = PLAYER1
    initialBoard[4, 3] = PLAYER1
    initialBoard[3, 3] = PLAYER2

    initialBoard[5, 2] = PLAYER2
    initialBoard[4, 2] = PLAYER2
    initialBoard[3, 2] = PLAYER1

    initialBoard[5, 1] = PLAYER1
    initialBoard[4, 1] = PLAYER1
    initialBoard[3, 1] = PLAYER1


    initialBoard[5, 0] = PLAYER2

    print(initialBoard)

    assert new_max(initialBoard, 1, 4, float('-inf'), float('inf')) == float('inf')

def test_generate_dumb_move4():
    from agents.agent_minimax.minimax import generate_move_minimax

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 4] = PLAYER1
    initialBoard[4, 4] = PLAYER2
    initialBoard[3, 4] = PLAYER2

    initialBoard[5, 3] = PLAYER1
    initialBoard[4, 3] = PLAYER1
    initialBoard[3, 3] = PLAYER2

    initialBoard[5, 2] = PLAYER2
    initialBoard[4, 2] = PLAYER2
    initialBoard[3, 2] = PLAYER1

    initialBoard[5, 1] = PLAYER1
    initialBoard[4, 1] = PLAYER1
    initialBoard[3, 1] = PLAYER1

    initialBoard[5, 0] = PLAYER2

    print(initialBoard)

    assert generate_move_minimax(initialBoard, PLAYER2, None) == (1, None)

#Test for scenario in which an immediate win should be blocked
def test_generate_move_block_win():
    from agents.agent_minimax.minimax import generate_move_minimax

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 6] = PLAYER2

    initialBoard[5, 5] = PLAYER2

    initialBoard[5, 3] = PLAYER1
    initialBoard[4, 3] = PLAYER1
    initialBoard[3, 3] = PLAYER1

    print(initialBoard)

    assert generate_move_minimax(initialBoard, PLAYER2, None) == (3, None)


def test_new_max_block_win():
    #for depth 1
    from agents.agent_minimax.minimax import new_max

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 6] = PLAYER2

    initialBoard[5, 5] = PLAYER2

    initialBoard[5, 3] = PLAYER1
    initialBoard[4, 3] = PLAYER1
    initialBoard[3, 3] = PLAYER1

    print(initialBoard)

    assert new_max(initialBoard, 1, 5, float('-inf'), float('inf')) == float('inf')

