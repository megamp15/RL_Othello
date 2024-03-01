import pygame
import numpy as np
from enum import Enum
from othelloPlayer import OthelloPlayer
from othelloUtil import GameMove

class PlayerTurn(Enum):
    """
    To keep track of whose turn we are referring to.
    """
    Player1 = 0
    Player2 = 1


class Othello():
    def __init__(self, player1:OthelloPlayer, player2:OthelloPlayer, board_size:tuple[int,int]=(8,8)) -> None:
        pygame.init()
        self.board_size = np.array(board_size)
        self.resetBoard(board_size)
        self.player1 = player1
        self.player2 = player2
        self.player_turn = PlayerTurn.Player1

    def resetBoard(self) -> None:
        """
        Resets the board to the initial game state
        """
        half_size = self.board_size//2
        board = np.zeros(self.board_size)
        board[half_size[0],half_size[1]] = 1
        board[half_size[0]-1,half_size[1]-1] = 1
        board[half_size[0]-1,half_size[1]] = -1
        board[half_size[0],half_size[1]-1] = -1

        self.board = board
    
    def displayBoard(self) -> None:
        """
        Displays the game board graphically (will be filled in later if we have time)
        """
        pass

    def startGame(self) -> None:
        """
        Starts a new game of Othello
        """
        self.resetBoard()
        while not self.checkGameOver():
            self.takeTurn()
        self.resetPlayer()

    def performMove(self, move:GameMove) -> None:
        """
        Adjusts the current players position based on the move they chose
        """
        pass

    def placeTile(self, coords:tuple[int,int]) -> None:
        """
        Places a tile at the given coordinates and flips any applicable tiles in each direction.
        """
        pass

    def checkGameOver(self) -> bool:
        """
        Checks to see if the game is over, either by eliminating one player, stalemating one player,
        or filling the entire board with tiles.
        """
        pass

    def countScore(self) -> tuple[int,int]:
        """
        Checks the active score for each player.
        """
        pass

    def resetPlayer(self) -> None:
        """
        Informs the othelloPlayer objects for player1 and player2 that the game is over and what the score was.
        """
        pass

    def takeTurn(self) -> None:
        """
        Depending on whose turn it is, allows the player to perform their moves and place a tile.
        This will wait on input from the player if the playerType is not AI.
        """
        if self.player_turn == PlayerTurn.Player1:
            current_player = self.player1
            next_player = PlayerTurn.Player2
        elif self.player_turn == PlayerTurn.Player2:
            current_player = self.player2
            next_player = PlayerTurn.Player1
        else:
            raise FileNotFoundError
        
        coords = self.board_size//2
        while (move := current_player.selectMove(self.board, coords) != GameMove.PlaceTile):
            self.performMove(move, coords)
        self.placeTile(coords)
        
        self.player_turn = next_player
