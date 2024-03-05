from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from othelloUtil import *

class PlayerType(Enum):
    """
    The type of player assigned to a playerTurn. Human allows for manual input,
    Agent allows for computed input from our RL agent, and AI is for implemented AI
    that will be developed later from the Agent we train.
    """
    Human = 0
    AI = 1
    Agent = 2

class OthelloPlayer(ABC):
    """
    An abstract class which can be extended to respond to a game state given by othelloGame and select
    a move to perform that othelloGame can execute
    """
    def __init__(self, playerType:PlayerType, mode:GameMove):
        self.type = playerType
        self.mode = mode

    @abstractmethod
    def selectMove(self, board:np.ndarray, coords:tuple[int,int], availableMoves:list[GameMove]) -> GameMove:
        """
        Called when self.mode in [MoveMode.Directions8,MoveMode.Directions4].
        Based on the game state of the board, and the player's coordinates for selecting a square
        to place a tile, select a move to adjust its coordinates or place a tile ending their turn.
        """
        pass

    @abstractmethod
    def selectMove(self, board:np.ndarray, availableMoves:list[tuple[int,int]]) -> tuple[int,int]:
        """
        Called when self.move == MoveMode.FullBoard to select a coordinate to place a new tile.
        """
        pass

    @abstractmethod
    def reset(self, score:tuple[int,int]) -> None:
        """
        This is called when the game is over and the player needs to reset. It also retrieves the final score.
        """
        pass