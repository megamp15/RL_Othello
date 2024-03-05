import pygame
import numpy as np
from enum import Enum
from othelloPlayer import OthelloPlayer, HumanPlayer
from othelloUtil import *
import itertools

class PlayerTurn(Enum):
    """
    To keep track of whose turn we are referring to.
    """
    Player1 = 0
    Player2 = 1


class Othello():
    """
    An Othello (Reversi) game that can be played by humans and AI alike, or for training RL agents how to play!
    """
    def __init__(self, player1:OthelloPlayer, player2:OthelloPlayer, board_size:tuple[int,int]=(8,8), \
                 mode:MoveMode=MoveMode.Directions8) -> None:
        pygame.init()
        self.board_size = np.array(board_size)
        self.resetBoard()
        self.player1 = player1
        self.player2 = player2
        self.activePlayer = PlayerTurn.Player1
        self.mode = mode

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
        print(self.board)

    def startGame(self) -> None:
        """
        Starts a new game of Othello
        """
        self.resetBoard()
        print("Game started.\n Current Position: (3,3)")
        while not self.checkGameOver():
            self.takeTurn()
            # clear() # Clears out the cli 
        self.resetPlayer()

    def performMove(self, move:GameMove, coords:tuple[int,int]) -> tuple[int,int]:
        """
        Adjusts the current players position based on the move they chose
        """
        match move:
            case GameMove.North:
                offset = [-1,0]
            case GameMove.South:
                offset = [1,0]
            case GameMove.East:
                offset = [0,1]
            case GameMove.West:
                offset = [0,-1]
            case GameMove.NorthEast:
                offset = [-1,1]
            case GameMove.NorthWest:
                offset = [-1,-1]
            case GameMove.SouthEast:
                offset = [1,1]
            case GameMove.SouthWest:
                offset = [1,-1]
            case _:
                raise FileNotFoundError
        return  tuple(np.array(coords) + offset)
    
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
        # Default to current player
        if selectedPlayer == None:
            selectedPlayer = self.activePlayer
        # Check if the current coords has no tile, then flip applicable tiles and place new one
        if self.board[*coords] == PlayerTurn.NoPlayer.value:
            mask = self.findFlippableTiles(coords,selectedPlayer)
            self.board[mask] = -self.board[mask]
            self.board[*coords] = selectedPlayer.value
        else:
            raise FileNotFoundError
        
    def findFlippableTiles(self,coords:tuple[int,int],playerTurn:PlayerTurn) -> np.ndarray[bool,bool]:
        """
        Finds a logical mask of the tiles to be flipped if the given player were to place a tile at the
        given coordinates. To be used for finding valid moves and for actual tile placement.
        """
        # Default to active player (can't reference self in function parameter)
        if selectedPlayer == None:
            selectedPlayer = self.activePlayer
        
        directions = getDirectionMoves_8()
        full_mask = np.zeros(self.board_size,dtype=bool)
        for d in directions:
            offset = gameMoveToOffset(d)
            next_position = np.array(coords)
            temp_mask = np.zeros(self.board_size,dtype=bool)
            # Continually apply the direction vector to the offset coordinates, and if its within bounds,
            # and either has a tile of the opponent or no tile, add to the mask if its the opponent tile
            # or reset if there is no tile
            while self.isWithinBounds(next_position:= next_position + offset) and \
                  self.board[*next_position] != selectedPlayer.value:
                if self.board[*next_position] == -selectedPlayer.value:
                    temp_mask[*next_position] = True
                else:
                    temp_mask = np.zeros(self.board_size,dtype=bool)
                    break
            if self.isWithinBounds(offset):
                full_mask |= temp_mask
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
        player1_score = np.sum(self.board == -1)
        player2_score = np.sum(self.board == 1)
        return (player1_score,player2_score)

    def resetPlayer(self) -> None:
        """
        Informs the othelloPlayer objects for player1 and player2 that the game is over and what the score was.
        """
        score = self.countScore()
        self.player1.reset(score)
        self.player2.reset(score)
    
    def getPlayer(self, selectedTurn:PlayerTurn=None) -> OthelloPlayer:
        """
        Retuns the player object corresponding to the selected turn.
        """
        if selectedTurn == None:
            selectedTurn = self.activePlayer
        if selectedTurn == PlayerTurn.Player1:
            return self.player1
        elif selectedTurn == PlayerTurn.Player2:
            return self.player2
    
    def flipTurn(self, selectedPlayer:PlayerTurn=None) -> PlayerTurn:
        """
        Return the next player after the selected one.
        """
        if selectedPlayer == None:
            selectedPlayer = self.activePlayer
        if selectedPlayer == PlayerTurn.Player1:
            return PlayerTurn.Player2
        elif selectedPlayer == PlayerTurn.Player2:
            return PlayerTurn.Player1
        else:
            raise FileNotFoundError

    def getAvailableMoves(self, coords:tuple[int,int]=None, selectedPlayer:PlayerTurn=None) -> list[GameMove]:
        """
        Returns the available, valid moves that can be performed at the given coordinates.
        """
        # Default to the active player
        if selectedPlayer == None:
            selectedPlayer = self.activePlayer
        # We should pass None for coords if using FullBoardSelect
        if coords == None and self.getPlayer(selectedPlayer).mode != MoveMode.FullBoardSelect:
            raise FileNotFoundError

        availableMoves = list[GameMove]()
        if self.mode == MoveMode.Directions8:
            directionMoves = getDirectionMoves_8()
        elif self.mode == MoveMode.Directions4:
            directionMoves = getDirectionMoves_4()
        else:
            moves = np.where(self.findAvailableTilePlacements(selectedPlayer))
            return [tuple(x) for x in moves]

        coords = np.array(coords)
        # Find all available tile placements and check if any of the direction moves can move to one
        moveMask = self.findAvailableTilePlacements(selectedPlayer)
        for direction in directionMoves:
            if self.isWithinBounds(new_coords := self.performMove(direction,coords)) \
                and np.sum(self.findFlippableTiles(new_coords,selectedPlayer)) > 0:
                availableMoves.append(direction)

        if np.sum(self.findFlippableTiles(coords,selectedPlayer)) > 0:
            availableMoves.append(GameMove.PlaceTile)
        return availableMoves

    def takeTurn(self) -> None:
        """
        Depending on whose turn it is, allows the player to perform their moves and place a tile.
        This will wait on input from the player if the playerType is not AI.
        """        
        current_player = self.getPlayer(self.activePlayer)
        self.displayBoard()

        if self.mode == MoveMode.FullBoardSelect:
            coords = (self.board_size-[1,1])//2
            print(f"Current Position: {coords}")
            self.displayBoard()
            availableMoves = self.getAvailableMoves(coords)
            while ((move := current_player.selectMove(self.board, coords, availableMoves)) != GameMove.PlaceTile):
                print(f"move: {move}")
                if move not in availableMoves:
                    raise FileNotFoundError
                coords = self.performMove(move, coords)
                print(f"Current Position: {coords}")
                availableMoves = self.getAvailableMoves(coords)
        else:
            self.displayBoard()
            availableMoves = self.getAvailableMoves()
            coords = current_player.selectMove(self.board, availableMoves)

        self.placeTile(coords)
        self.activePlayer = self.flipTurn(self.activePlayer)

if __name__ == '__main__':
    player1 = HumanPlayer()
    player2 = HumanPlayer()
    game = Othello(player1,player2,(8,8))
    game.findFlippableTiles((2,4),PlayerTurn.Player1)
    # The available moves at (3,3) are north and east
    # If you change the starting coordinate to (4,4) on line 203 the available moves is south, east
    # game.takeTurn()
    game.startGame()