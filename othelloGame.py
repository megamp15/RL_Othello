# import pygame
import numpy as np
from enum import Enum
from othelloPlayer import OthelloPlayer, HumanPlayer,AgentPlayer
from othelloUtil import *
from dqn import DDQN,DQN,DuelDQN
import itertools
from tqdm import trange

from neuralNet import PixelNeuralNet,StateNeuralNet
from agent import DeepAgent

from environment import Environment

class PlayerTurn(Enum):
    """
    To keep track of whose turn we are referring to.
    """
    Player1 = 1
    Player2 = -1
    NoPlayer = 0

class Othello(Environment):
    """
    An Othello (Reversi) game that can be played by humans and AI alike, or for training RL agents how to play!
    """
    def __init__(self, player1:OthelloPlayer, player2:OthelloPlayer, board_size:tuple[int,int]=(8,8)) -> None:
        # pygame.init()
        self.board_size = np.array(board_size)
        self.state_space = [1] + list(board_size)
        self.reset()
        self.num_actions = np.prod(self.board_size)
        self.player1 = player1
        self.player2 = player2
        self.activePlayer = PlayerTurn.Player1
        self.last_state = None
        self.reward = 0
        self.last_score = (2,2)

    def reset(self) -> None:
        """
        Resets the board to the initial game state
        """
        half_size = self.board_size//2
        self.board = np.zeros(self.board_size)
        self.board[half_size[0],half_size[1]] = PlayerTurn.Player2.value
        self.board[half_size[0]-1,half_size[1]-1] = PlayerTurn.Player2.value
        self.board[half_size[0]-1,half_size[1]] = PlayerTurn.Player1.value
        self.board[half_size[0],half_size[1]-1] = PlayerTurn.Player1.value
        self.activePlayer = PlayerTurn.Player1
        
    def resetBoard(self):
        self.reset()
    
    def displayBoard(self) -> None:
        """
        Displays the game board graphically (will be filled in later if we have time)
        """
        print(self.board)

    def startGame(self) -> None:
        """
        Starts a new game of Othello
        """
        self.reset()
        print("Game started.")
        while not self.checkGameOver():
            self.takeTurn()
            self.activePlayer = self.flipTurn(self.activePlayer)
            # clear() # Clears out the cli 
        self.resetPlayer()

    def performMove(self, move:GameMove, coords:tuple[int,int]) -> tuple[int,int]:
        """
        Adjusts the current players position based on the move they chose
        """
        offset = gameMoveToOffset(move)
        return tuple(np.array(coords) + offset)
    
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
    
    def findAvailableTilePlacements(self, playerTurn:PlayerTurn=None) -> np.ndarray[bool]:
        """
        Find all locations that the selected player can place a tile that will flip at least
        one opponent tile.
        """
        availablePlacements = np.zeros(self.board_size,dtype=bool)
        for x,y in itertools.product(range(self.board_size[0]),range(self.board_size[1])):
            # For each place on the board, check if we can flip at least one tile
            if np.sum(self.findFlippableTiles((x,y),playerTurn)) > 0:
                availablePlacements[x,y] = True
        return availablePlacements

    def placeTile(self, coords:tuple[int,int], selectedPlayer:PlayerTurn=None) -> None:
        """
        Places a tile at the given coordinates and flips any applicable tiles in each direction.
        """
        # Default to current player
        if selectedPlayer == None:
            selectedPlayer = self.activePlayer
        # Check if the current coords has no tile, then flip applicable tiles and place new one
        if self.board[coords[0],coords[1]] == PlayerTurn.NoPlayer.value:
            mask = self.findFlippableTiles(coords,selectedPlayer)
            self.board[mask] = -self.board[mask]
            self.board[coords[0],coords[1]] = selectedPlayer.value
        else:
            print(f'Not a valid move: {coords}')
            self.displayBoard()
            exit(1)
        
    def findFlippableTiles(self,coords:tuple[int,int],selectedPlayer:PlayerTurn=None) -> np.ndarray[bool,bool]:
        """
        Finds a logical mask of the tiles to be flipped if the given player were to place a tile at the
        given coordinates. To be used for finding valid moves and for actual tile placement.
        """
        # Default to active player (can't reference self in function parameter)
        if selectedPlayer == None:
            selectedPlayer = self.activePlayer
        
        directions = getDirectionMoves_8()
        full_mask = np.zeros(self.board_size,dtype=bool)
        # If there is already a tile here, say we can't flip any tiles (can't place one)
        if self.board[*coords] != PlayerTurn.NoPlayer.value:
            return full_mask
        # Check in each direction from the given coords
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
            # Add this direction to the mask to be flipped
            full_mask |= temp_mask
        return full_mask

    def checkGameOver(self) -> bool:
        """
        Checks to see if the game is over, either by eliminating one player, stalemating one player,
        or filling the entire board with tiles.
        """
        # The whole board is full
        if np.sum(self.board == PlayerTurn.NoPlayer.value) == 0:
            return True
        # Either player has no more tiles on the board
        elif np.sum(self.board == PlayerTurn.Player1.value) == 0 or \
             np.sum(self.board == PlayerTurn.Player2.value) == 0:
            return True
        # The next player has no available positions to place a tile
        elif np.sum(self.findAvailableTilePlacements()) == 0:
            return True
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
        selectedPlayerObj = self.getPlayer(selectedPlayer)
        if selectedPlayerObj.mode == MoveMode.FullBoardSelect and coords != None:
            raise FileNotFoundError

        availableMoves = list[GameMove]()
        if selectedPlayerObj.mode == MoveMode.Directions8:
            directionMoves = getDirectionMoves_8()
        elif selectedPlayerObj.mode == MoveMode.Directions4:
            directionMoves = getDirectionMoves_4()
        else:
            moves = np.array(np.where(self.findAvailableTilePlacements(selectedPlayer))).T
            #print("getAvailMovs")
            #print(self.getState())
            #print(moves)
            return [tuple(x) for x in moves]

        coords = np.array(coords)
        # Find all available tile placements and check if any of the direction moves can move to one
        moveMask = self.findAvailableTilePlacements(selectedPlayer)
        for direction in directionMoves:
            if self.isWithinBounds(self.performMove(direction,coords)):
                availableMoves.append(direction)

        # Check to see if the player can place a tile at the current coords too
        if moveMask[*coords]:
            availableMoves.append(GameMove.PlaceTile)

        return availableMoves

    def takeTurn(self) -> None:
        """
        Depending on whose turn it is, allows the player to perform their moves and place a tile.
        This will wait on input from the player if the playerType is not AI.
        """        
        current_player = self.getPlayer(self.activePlayer)

        if current_player.mode == MoveMode.FullBoardSelect:
            self.displayBoard()
            availableMoves = self.getAvailableMoves()
            state = self.getState()
            coords = current_player.selectMoveFullBoardSelect(state, availableMoves)
        else:
            coords = (self.board_size-[1,1])//2
            print(f"Current Position: {coords}")
            self.displayBoard()
            availableMoves = self.getAvailableMoves(coords)
            state = self.getState()
            while ((move := current_player.selectMove(state, availableMoves, coords)) != GameMove.PlaceTile):
                print(f"move: {move}")
                if move not in availableMoves:
                    raise FileNotFoundError
                coords = self.performMove(move, coords)
                print(f"Current Position: {coords}")
                availableMoves = self.getAvailableMoves(coords)

        self.placeTile(coords)

    def getState(self, playerTurn:PlayerTurn=None) -> np.ndarray:
        """
        Used to return the current state of the environment for the selected player (flipped for player 2).
        """
        if playerTurn == None:
            playerTurn = self.activePlayer
        if playerTurn == PlayerTurn.Player1:
            return self.board
        elif playerTurn == PlayerTurn.Player2:
            return -self.board
        else:
            print(f'Not a player: {playerTurn}')
            return None

    def getReward(self, playerTurn:PlayerTurn=None) -> float:
        """
        Used to calculate the reward value for the last move performed for the selected player.
        """
        current_score = self.countScore()
        last_score = self.last_score
        diff =  np.subtract(current_score,last_score)
        #print("env step prev score",last_score)
        #print("env step now score",current_score)
        #print("env:getreward")
        #print(diff)
        return diff
        

    def step(self, action:int) -> bool:
        """
        Used to train agents in a similar way to the gymnasium's othello environment.
        This only works for MoveMode.FullBoardSelect.
        """
        self.last_state = self.getState()
        self.last_score = self.countScore()
        if self.checkGameOver():
            self.reward = self.getReward()
            self.activePlayer = self.flipTurn()
            return self.getState(),self.reward
        self.placeTile(action)
        self.reward = self.getReward()
        self.activePlayer = self.flipTurn()
        return self.getState(),self.reward

if __name__ == '__main__':
    mode = MoveMode.FullBoardSelect
    player1 = AgentPlayer(mode,agent=None)
    player2 = AgentPlayer(mode,agent=None)
    game = Othello(player1,player2,(8,8))
    #state_shape = (1,1,8,8)
    #state_shape = (1,1,10,10)
    #Params stolen from othello.py to get it running.
    # AGENT PARAMS
    EPSILON = .75
    EPSILON_DECAY_RATE = 0.99
    EPSILON_MIN = 0.01
    ALPHA = 0.01
    GAMMA = 0.9
    SKIP_TRAINING = 1_000
    SAVE_INTERVAL = 500
    SYNC_INTERVAL = 250
    
    # TRAINING PARAMS
    EPISODES = 1
    MAX_STEPS = 10_000

    num_actions=60
    
    # Define agent parameters once so it's not quite so verbose
    params = {'state_shape' : environment.state_space,
              'num_actions' : environment.num_actions,
              'epsilon' : EPSILON,
              'epsilon_decay_rate' : EPSILON_DECAY_RATE,
              'epsilon_min' : EPSILON_MIN,
              'alpha' : ALPHA,
              'gamma' : GAMMA,
              'sync_interval' : SYNC_INTERVAL,
              'skip_training' : SKIP_TRAINING,
              'save_interval' : SAVE_INTERVAL,
              'max_memory' : MEMORY_CAPACITY,
              'save_path' : save_model_path
              }
    p1_dqn = DQN(env=game, state_shape=state_shape, net_type=StateNeuralNet, num_actions=num_actions, epsilon=EPSILON, epsilon_decay_rate=EPSILON_DECAY_RATE, epsilon_min=EPSILON_MIN, alpha=ALPHA, gamma=GAMMA, skip_training=SKIP_TRAINING, save_interval=SAVE_INTERVAL,sync_interval=SYNC_INTERVAL,max_memory=10_000,save_path='./DQN/test.junk')
    p2_dqn = DQN(env=game, state_shape=state_shape, net_type=StateNeuralNet,num_actions=num_actions, epsilon=EPSILON, epsilon_decay_rate=EPSILON_DECAY_RATE, epsilon_min=EPSILON_MIN, alpha=ALPHA, gamma=GAMMA, skip_training=SKIP_TRAINING, save_interval=SAVE_INTERVAL,sync_interval=SYNC_INTERVAL,max_memory=10_000,save_path='./DQN/test.junk')
    player1.setAgent(p1_dqn)    
    player2.setAgent(p2_dqn)
    # The available moves at (3,3) are north and east
    # If you change the starting coordinate to (4,4) on line 203 the available moves is south, east
    game.startGame()