from enum import Enum

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