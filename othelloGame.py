import pygame
import numpy as np
from enum import Enum
from othelloPlayer import OthelloPlayer
from othelloUtil import GameMove
from humanPlayer import humanPlayer

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
        self.resetBoard()
        self.player1 = player1
        self.player2 = player2
        self.player_turn = PlayerTurn.Player1

    def resetBoard(self) -> None:
        """
        Resets the board to the initial game state
        """
        half_size = self.board_size//2
        self.board = np.zeros(self.board_size)
        self.board[half_size[0],half_size[1]] = 1
        self.board[half_size[0]-1,half_size[1]-1] = 1
        self.board[half_size[0]-1,half_size[1]] = -1
        self.board[half_size[0],half_size[1]-1] = -1
    
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

    def performMove(self, move:GameMove, coords:tuple[int,int]) -> tuple[int,int]:
        """
        Adjusts the current players position based on the move they chose
        """
        match move:
            case GameMove.North:
                offset = [0,1]
            case GameMove.South:
                offset = [0,-1]
            case GameMove.East:
                offset = [-1,0]
            case GameMove.West:
                offset = [1,0]
            case GameMove.NorthEast:
                offset = [-1,1]
            case GameMove.NorthWest:
                offset = [1,1]
            case GameMove.SouthEast:
                offset = [-1,-1]
            case GameMove.SouthWest:
                offset = [1,-1]
            case _:
                raise FileNotFoundError
        return coords + offset
    
    def isWithinBounds(self, coords:tuple[int,int]) -> bool:
        """
        A simple helper to check if a coordinate pair is within the bounds of the current board
        """
        if coords[0] < 0 or coords[0] > self.board_size[0]-1:
            return False
        elif coords[1] < 0 or coords[1] > self.board_size[1]-1:
            return False
        else:
            return True

    def placeTile(self, coords:tuple[int,int]) -> None:
        """
        Places a tile at the given coordinates and flips any applicable tiles in each direction.
        """
        if self.board[*coords] == 0:
            if self.player_turn == PlayerTurn.Player1:
                self.board[*coords] = 1
                current_player_tile = 1
            elif self.player_turn == PlayerTurn.Player2:
                self.board[*coords] = -1
                current_player_tile = -1
            else:
                raise FileNotFoundError
        else:
            raise FileNotFoundError
        
        
    
    def findFlippableTiles(self,coords:tuple[int,int],playerTurn:PlayerTurn) -> np.ndarray[bool,bool]:
        """
        Finds a logical mask of the tiles to be flipped if the given player were to place a tile at the
        given coordinates. To be used for finding valid moves and for actual tile placement.
        """
        if playerTurn == PlayerTurn.Player1:
            current_player_tile = 1
        elif playerTurn == PlayerTurn.Player2:
            current_player_tile = -1
        else:
            raise FileNotFoundError
        
        directions = [[0,1],
                      [1,1],
                      [1,0],
                      [1,-1],
                      [0,-1],
                      [-1,-1],
                      [-1,0],
                      [-1,1]]
        full_mask = np.zeros(self.board_size,dtype=bool)
        for d in directions:
            offset = np.array(coords)
            temp_mask = np.zeros(self.board_size,dtype=bool)
            while self.isWithinBounds(offset:= offset + d) and self.board[*offset] != current_player_tile:
                if self.board[*offset] == -current_player_tile:
                    temp_mask[*offset] = True
                else:
                    temp_mask = np.zeros(self.board_size,dtype=bool)
                    break
            if self.isWithinBounds(offset):
                full_mask &= temp_mask
        return full_mask

    def checkGameOver(self) -> bool:
        """
        Checks to see if the game is over, either by eliminating one player, stalemating one player,
        or filling the entire board with tiles.
        """
        # The whole board is full
        if np.sum(self.board == 0) == 0:
            return True
        # The nonactive player has no more tiles on the board
        elif np.sum(self.board == 1) == 0:
            return True
        # The nonactive player has no available positions to place a tile
        # elif len(self.getAvailableMoves())
        else:
            return False

    def countScore(self) -> tuple[int,int]:
        """
        Checks the active score for each player.
        """
        player1_score = np.sum(self.board == 1)
        player2_score = np.sum(self.board == -1)
        return (player1_score,player2_score)

    def resetPlayer(self) -> None:
        """
        Informs the othelloPlayer objects for player1 and player2 that the game is over and what the score was.
        """
        score = self.countScore()
        self.player1.reset(score)
        self.player2.reset(score)

    def getAvailableMoves(self, coords:tuple[int,int], active_player:PlayerTurn=None) -> list[GameMove]:
        """
        Returns the available, valid moves that can be performed at the given coordinates.
        """
        if active_player == None:
            active_player = self.player_turn

        availableMoves = list[GameMove]()
        directionMoves = [GameMove.North,GameMove.South,GameMove.East,GameMove.West,
                          GameMove.NorthEast,GameMove.NorthWest,GameMove.SouthEast,GameMove.SouthWest]
        
        for direction in directionMoves:
            if self.isWithinBounds(new_coords := self.performMove(direction,coords)) \
                and np.sum(self.findFlippableTiles(new_coords,active_player)) > 0:
                availableMoves.append(direction)

        if np.sum(self.findFlippableTiles(coords,active_player)) > 0:
            availableMoves.append(GameMove.PlaceTile)

        return availableMoves

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
        
        coords = (self.board_size-[1,1])//2
        availableMoves = self.getAvailableMoves(coords)
        while (move := current_player.selectMove(self.board, coords, availableMoves) != GameMove.PlaceTile):
            if move not in availableMoves:
                raise FileNotFoundError
            coords = self.performMove(move, coords)
            availableMoves = self.getAvailableMoves(coords)
        self.placeTile(coords)
        
        self.player_turn = next_player

if __name__ == '__main__':
    player1 = humanPlayer()
    player2 = humanPlayer()
    game = Othello(player1,player2,(8,8))
    game.findFlippableTiles((2,4),PlayerTurn.Player1)
    # game.startGame()