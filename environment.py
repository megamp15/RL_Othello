from abc import ABC, abstractmethod
import numpy as np

class Environment(ABC):
    name = 'abstract environment'
    
    def __init__(self) -> None:
        self.state_space : int = 0
        self.num_actions : int = 0

    @abstractmethod
    def reset(self) -> None:
        """
        Used to reset the game to it's initial state
        """
        pass

    @abstractmethod
    def getState(self) -> np.ndarray:
        """
        Used to return the current state of the game
        """
        pass

    @abstractmethod
    def getAvailableMoves(self) -> list:
        """
        Used to retrieve a list of available moves, either for this specific turn, or for the whole game.
        """
        pass

    @abstractmethod
    def getReward(self) -> float:
        """
        Used to calculate the reward of the last action performed
        """
        pass

    @abstractmethod
    def step(self, action:int) -> None:
        """
        Used to process some action to update the environment
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Used to shut the environment down and free any resources it uses
        """