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

class HumanPlayer(OthelloPlayer):
    """
    Human Player to play the game manually using the command line interface (cli)
    """
    def __init__(self):
        super().__init__(PlayerType.Human)


    def selectMove(self, board:np.ndarray=None, coords:tuple[int,int]=None, availableMoves:list[GameMove]=None) -> GameMove:
        if len(availableMoves) == 0:
            print("No moves available")
            return None
        
        print("Please select one of the following available moves:")
        for i, move in enumerate(availableMoves):
            print(f"[{i}] {move.name}")

        while True:
            selectedOption = input("Enter the number corresponding to your desired action => ")
            try:
                if 0 <= int(selectedOption) < len(availableMoves):
                    break
                else: 
                    print("Invalid option.")
            except:
                print("Invalid input. Please enter a valid integer.")
        return availableMoves[int(selectedOption)]
       
            
    def reset(self, score:tuple[int,int]) -> None:
        """
        Not exactly sure a reset is needed for players at this moment.
        """
        pass

    
class AgentPlayer(OthelloPlayer):
    """
    An RL Agent Player that plays the game
    """
    def __init__(self, agent, savedModelPath:str=None):
        """
        Takes a agent object (waiting on this object)
        savedModelPath will be used to load the model's parameters
        """
        super().__init__(PlayerType.Agent)
        self.agent = agent
        if savedModelPath:
            self.agent.load_model()


    def selectMove(self, board:np.ndarray, coords:tuple[int,int], availableMoves:list[GameMove]) -> GameMove:
        """
        Calls the agents action method. 
        Will the agent now take the board (state) and availableMoves?
        get_action should return GameMove
        """
        return self.agent.get_action(board)

    def reset(self, score:tuple[int,int]) -> None:
        """
        Not exactly sure a reset is needed for players at this moment.
        """
        pass


if __name__ == "__main__":
    # Just testing inputs work. Can be used to test OthelloGame
    human = HumanPlayer()
    human.selectMove(availableMoves=[GameMove.North, GameMove.East, GameMove.West])
