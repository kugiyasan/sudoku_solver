import math
import numpy as np
from typing import Iterable, Set, Tuple


def get_row_column_square(
    puzzle: np.ndarray, x: int, y: int
) -> Tuple[np.ndarray, np.ndarray, Iterable[int]]:
    """
    utility function, returns the row, column and square of the puzzle

    >>> puzzle = np.arange(81).reshape((9, 9))
    >>> rcs = get_row_column_square(puzzle, 0, 0)
    >>> rcs[0]
    array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    >>> rcs[1]
    array([ 0,  9, 18, 27, 36, 45, 54, 63, 72])
    """
    sqrt = int(puzzle.shape[0] ** 0.5)
    sq_y = y // sqrt * sqrt
    sq_x = x // sqrt * sqrt

    # ? Should remove duplicate cells, keeping them is useless
    # ? Although I don't think it would improve significantly the performance
    square = puzzle[sq_y : sq_y + sqrt, sq_x : sq_x + sqrt]
    return puzzle[y], puzzle[:, x], square.flat


def check_grid_validity(puzzle: np.ndarray) -> None:
    """
    Raise an error if the given puzzle isn't the right size
    or has number that aren't possible

    >>> check_grid_validity(np.array([]))
    Traceback (most recent call last):
    utils.SudokuError: The sudoku doesn't have the right shape

    >>> check_grid_validity(np.full(9, 1))
    Traceback (most recent call last):
    utils.SudokuError: The sudoku doesn't have the right shape

    >>> grid = np.full((9, 9), 0)
    >>> grid[0, 0] = -1
    >>> check_grid_validity(grid)
    Traceback (most recent call last):
    utils.SudokuError: Some cell have numbers that are out of range
    """
    SUDOKU_SIZE = puzzle.shape[0]
    # Check to make sure that SUDOKU_SIZE is a square number
    if SUDOKU_SIZE != math.isqrt(SUDOKU_SIZE) ** 2:
        raise ValueError("SUDOKU_SIZE should be a square number")

    if puzzle.shape != (SUDOKU_SIZE, SUDOKU_SIZE):
        raise SudokuError("The sudoku doesn't have the right shape")

    for i in range(len(puzzle)):
        row = puzzle[i]
        if len(set(row)) + (row == 0).sum() != SUDOKU_SIZE + 1:
            raise SudokuError(f"The same number appears twice in row {i}")

        column = puzzle[:, i]
        if len(set(column)) + (column == 0).sum() != SUDOKU_SIZE + 1:
            raise SudokuError(f"The same number appears twice in column {i}")

        # ! Check for the same number in the same square

        for cell in row:
            if cell < 0 or cell > SUDOKU_SIZE:
                raise SudokuError("Some cell have numbers that are out of range")


def candidates_to_bits(candidates: Set[int]) -> int:
    """
    Convert a candidates set into its bits representation
    The bits are in big endian order

    >>> bin(candidates_to_bits({1, 2, 3, 6, 7}))
    '0b1100111'

    >>> bin(candidates_to_bits({6, 9}))
    '0b100100000'
    """
    bits = 0
    for candidate in candidates:
        bits ^= 1 << (candidate - 1)

    return bits


def bits_to_candidates(bits: int) -> Set[int]:
    """
    Convert a bits representation into a set of candidates

    >>> bits_to_candidates(0b111)
    {1, 2, 3}

    >>> bits_to_candidates(0b111000101)
    {1, 3, 7, 8, 9}
    """
    candidates = set()
    i = 1
    while bits:
        bit = bits & 1
        if bit:
            candidates.add(i)

        bits >>= 1
        i += 1

    return candidates


def get_cell_candidates(puzzle: np.ndarray, x: int, y: int) -> Set[int]:
    """
    Discard the numbers
    that are on the same speficied line, column or square

    Returns the bit representation of the possible candidates
    """
    candidates = {*range(1, puzzle.shape[0] + 1)}

    row, column, square = get_row_column_square(puzzle, x, y)
    candidates.difference_update(row)
    candidates.difference_update(column)

    candidates.difference_update(square)

    return candidates


def create_cell_candidates(puzzle: np.ndarray) -> np.ndarray:
    """
    Compute the candidates for each cell
    Candidates will be stored
    as a number from 0 to 2^SUDOKU_SIZE-1 (-1 is used for filled cells)
    """
    SUDOKU_SIZE = puzzle.shape[0]
    cell_candidates = np.full((SUDOKU_SIZE, SUDOKU_SIZE), -1)

    for y in range(len(puzzle)):
        for x in range(len(puzzle[y])):
            if puzzle[y][x] == 0:
                bits = candidates_to_bits(get_cell_candidates(puzzle, x, y))
                cell_candidates[y][x] = bits

    return cell_candidates


class SudokuError(Exception):
    pass
