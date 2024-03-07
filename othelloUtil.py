from enum import Enum
from os import system, name

class GameMove(Enum):
    """
    The actions that a player can perform in the process of taking their turn
    and placing a tile.
    """
    North = 0
    South = 1
    East = 2
    West = 3
    NorthEast = 4
    NorthWest = 5
    SouthEast = 6
    SouthWest = 7
    PlaceTile = 8

class MoveMode(Enum):
    """
    To select the method by which the player selects a coordinate to place a tile
    """
    FullBoardSelect = 1
    Directions8 = 2
    Directions4 = 3

def getDirectionMoves_8() -> list[GameMove]:
    return [GameMove.North,GameMove.South,GameMove.East,GameMove.West,GameMove.NorthEast,GameMove.NorthWest,
            GameMove.SouthEast,GameMove.SouthWest]

def getDirectionMoves_4() -> list[GameMove]:
    return [GameMove.North,GameMove.South,GameMove.East,GameMove.West]

def gameMoveToOffset(move:GameMove) -> tuple[int,int]:
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
    return offset

def clear():
    # for windows
    if name == 'nt':
        _ = system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')