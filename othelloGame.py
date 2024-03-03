import pygame
import numpy as np
from enum import Enum
from othelloPlayer import OthelloPlayer, HumanPlayer
from othelloUtil import GameMove, clear

class PlayerTurn(Enum):
    """
    To keep track of whose turn we are referring to.
    """
    Player1 = 1
    Player2 = -1


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
        self.board[half_size[0],half_size[1]] = PlayerTurn.Player2.value
        self.board[half_size[0]-1,half_size[1]-1] = PlayerTurn.Player2.value
        self.board[half_size[0]-1,half_size[1]] = PlayerTurn.Player1.value
        self.board[half_size[0],half_size[1]-1] = PlayerTurn.Player1.value
    
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

    def placeTile(self, coords:tuple[int,int], playerTurn:PlayerTurn=None) -> None:
        """
        Places a tile at the given coordinates and flips any applicable tiles in each direction.
        """
        if playerTurn == None:
            playerTurn = self.player_turn
        if self.board[*coords] == 0:
            mask = self.findFlippableTiles(coords,playerTurn)
            self.board[mask] = -self.board[mask]
            self.board[*coords] = playerTurn.value
        else:
            raise FileNotFoundError
        
    def findFlippableTiles(self,coords:tuple[int,int],playerTurn:PlayerTurn=None) -> np.ndarray[bool,bool]:
        """
        Finds a logical mask of the tiles to be flipped if the given player were to place a tile at the
        given coordinates. To be used for finding valid moves and for actual tile placement.
        """
        # Default to active player (can't reference self in function parameter)
        if playerTurn == None:
            playerTurn = self.player_turn
        
        directions = [[-1,0],
                      [1,0],
                      [0,1],
                      [0,-1],
                      [-1,1],
                      [-1,-1],
                      [1,1],
                      [1,-1]]
        full_mask = np.zeros(self.board_size,dtype=bool)
        # Check in each direction from the given coords
        for d in directions:
            offset = np.array(coords)
            temp_mask = np.zeros(self.board_size,dtype=bool)
            # Continually apply the direction vector to the offset coordinates, and if its within bounds,
            # and either has a tile of the opponent or no tile, add to the mask if its the opponent tile
            # or reset if there is no tile
            while self.isWithinBounds(offset:= offset + d) and self.board[*offset] != playerTurn.value:
                if self.board[*offset] == -playerTurn.value:
                    temp_mask[*offset] = True
                else:
                    temp_mask = np.zeros(self.board_size,dtype=bool)
                    break
            # Add this direction to the mask to be flipped
            full_mask |= temp_mask
        return full_mask

    def checkGameOver(self) -> bool:
        """
        Checks to see if the game is over, either by eliminating one player, stalemating one player,
        or filling the entire board with tiles.
        """
        # The whole board is full
        if np.sum(np.logical_and(self.board != PlayerTurn.Player1.value, self.board != PlayerTurn.Player2.value)) == 0:
            return True
        # The nonactive player has no more tiles on the board
        elif np.sum(self.board == -self.player_turn.value) == 0:
            return True
        # The nonactive player has no available positions to place a tile
        # elif len(self.getAvailableMoves())
        else:
            return False

    def countScore(self) -> tuple[int,int]:
        """
        Checks the active score for each player.
        """
        player1_score = np.sum(self.board == PlayerTurn.Player1.value)
        player2_score = np.sum(self.board == PlayerTurn.Player2.value)
        return (player1_score,player2_score)

    def resetPlayer(self) -> None:
        """
        Informs the othelloPlayer objects for player1 and player2 that the game is over and what the score was.
        RL agents can use this to reset and play a new game.
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
        
        self.displayBoard()

        coords = (self.board_size-[1,1])//2
        print(f"Current Position: {coords}")
        availableMoves = self.getAvailableMoves(coords)
        while ((move := current_player.selectMove(self.board, coords, availableMoves)) != GameMove.PlaceTile):
            print(f"move: {move}")
            if move not in availableMoves:
                raise FileNotFoundError
            coords = self.performMove(move, coords)
            print(f"Current Position: {coords}")
            availableMoves = self.getAvailableMoves(coords)
        self.placeTile(coords)
        
        self.player_turn = next_player

if __name__ == '__main__':
    player1 = HumanPlayer()
    player2 = HumanPlayer()
    game = Othello(player1,player2,(8,8))
    # game.findFlippableTiles((2,4),PlayerTurn.Player1)
    # The available moves at (3,3) are north and east
    # If you change the starting coordinate to (4,4) on line 203 the available moves is south, east
    # game.takeTurn()
    game.startGame()