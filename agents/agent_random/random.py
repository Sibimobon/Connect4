import random as random

from typing import Optional, Tuple
import numpy as np
from agents.common import BoardPiece, SavedState, PlayerAction, NO_PLAYER


def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`

    possible_moves = {0,1,2,3,4,5,6}

    for x in range(board.shape[1]-1):
        if board[0, x] != NO_PLAYER:
            possible_moves.remove(x)

    possible_moves = list(possible_moves)
    action = random.choice(possible_moves)

    return action, None