from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from othelloUtil import *
from agent import DeepAgent

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
    def __init__(self, playerType:PlayerType, mode:MoveMode):
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
    def __init__(self, mode:MoveMode):
        super().__init__(PlayerType.Human, mode)

    def selectMove(self, board:np.ndarray, availableMoves:list[GameMove], coords:tuple[int,int]=None) -> GameMove:
        if len(availableMoves) == 0:
            print('No moves available.')
            return None
        
        if self.mode == MoveMode.FullBoardSelect:
            print('Please select one of the following available coordinates:')
            for i, move in enumerate(availableMoves):
                print(f'[{i}] {move}')
        else:
            print('Please select one of the following available moves:')
            for i, move in enumerate(availableMoves):
                print(f'[{i}] {move.name}')

        # Loop until we get a valid input
        while (selectedOption := int(input('Enter the number corresponding to your desired action => '))) >= len(availableMoves) \
            or selectedOption < 0:
            print('Invalid option.')

        return availableMoves[int(selectedOption)]
            
    def reset(self, score:tuple[int,int]) -> None:
        """
        Display the score and say who won.
        """
        if score[0] > score[1]:
            print('Game over. Player 1 won.')
        elif score[0] < score[1]:
            print('Game over. Player 2 won.')
        else:
            print('Game over. Players tied!')
        print(f'Player 1: {score[0]}')
        print(f'Player 2: {score[1]}')

    
class AgentPlayer(OthelloPlayer):
    """
    An RL Agent Player that plays the game
    """
    def __init__(self, mode:MoveMode, agent:DeepAgent, savedModelPath:str=None):
        """
        Takes a agent object (waiting on this object)
        savedModelPath will be used to load the model's parameters
        """
        super().__init__(PlayerType.Agent, mode)
        self.agent = agent
        if savedModelPath:
            self.agent.load_model()
            
    def setAgent(self,agent,savedModelPath:str=None):
        self.agent = agent
        if savedModelPath:
            self.agent.load_model()
            

    def selectMove(self, board:np.ndarray, availableMoves:list[GameMove], coords:tuple[int,int]) -> GameMove:
        """
        Calls the agents action method. 
        Will the agent now take the board (state) and availableMoves?
        get_action should return GameMove
        """
        action = self.agent.get_action(board)
        allMoves = getDirectionMoves_8() + [GameMove.PlaceTile]
        move = allMoves[action]
        if move not in availableMoves:
            print(f'Not an available move: {move}')
            exit(1)
        return move
    
    def selectMoveFullBoardSelect(self, board:np.ndarray, availableMoves:list[tuple[int,int]]) -> GameMove:
        """
        Calls the agents action method. 
        Will the agent now take the board (state) and availableMoves?
        get_action should return GameMove
        """
        
        action = self.agent.get_action(board,availableMoves)
        return action

    def reset(self, score:tuple[int,int]) -> None:
        """
        Somehow tell the agent that the game is over, who won, and what the score is (if applicable)
        """
        pass


if __name__ == "__main__":
    # Just testing inputs work. Can be used to test OthelloGame
    human = HumanPlayer(MoveMode.Directions8)
    human.selectMove(board=np.zeros((8,8)), availableMoves=[GameMove.North, GameMove.East, GameMove.West], coords=(3,3))
