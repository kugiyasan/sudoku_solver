import numpy as np

# SUDOKU_SIZE should always be a square number
SUDOKU_SIZE = 9


def set_sudoku_size(size: int):
    global SUDOKU_SIZE
    SUDOKU_SIZE = size

    # Check to make sure that SUDOKU_SIZE is a square number
    sqrt = int(SUDOKU_SIZE ** 0.5)
    if sqrt * sqrt != SUDOKU_SIZE:
        raise ValueError("SUDOKU_SIZE should be a square number")


def get_row_column_square(puzzle: np.ndarray, x: int, y: int) -> tuple:
    """utility function, returns the row, column and square of the puzzle"""
    # TODO write a implementation for puzzle: list[list]
    sqrt = int(SUDOKU_SIZE ** 0.5)
    sq_y = y // sqrt * sqrt
    sq_x = x // sqrt * sqrt

    # TODO Remove duplicate cells, keeping them is useless
    square = puzzle[sq_y : sq_y + sqrt, sq_x : sq_x + sqrt]
    return puzzle[y], puzzle[:, x], square.flat


def check_grid_validity(puzzle: np.ndarray) -> None:
    """
    Raise an error if the given puzzle isn't the right size
    or has number that aren't possible

    >>> check_grid_validity(np.array([]))
    Traceback (most recent call last):
    Exception: The sudoku doesn't have the right shape

    >>> check_grid_validity(np.full(SUDOKU_SIZE, 1))
    Traceback (most recent call last):
    Exception: The sudoku doesn't have the right shape

    >>> grid = np.full((SUDOKU_SIZE, SUDOKU_SIZE), 0)
    >>> grid[0, 0] = -1
    >>> check_grid_validity(grid)
    Traceback (most recent call last):
    Exception: Some cell have numbers that are out of range
    """
    # ! Raise Exception that are more appropriate
    if puzzle.shape != (SUDOKU_SIZE, SUDOKU_SIZE):
        raise Exception("The sudoku doesn't have the right shape")

    for i in range(len(puzzle)):
        row = puzzle[i]
        if len(set(row)) + (row == 0).sum() != SUDOKU_SIZE + 1:
            raise Exception("The same number appears twice in a row")

        column = puzzle[:, i]
        if len(set(column)) + (column == 0).sum() != SUDOKU_SIZE + 1:
            raise Exception("The same number appears twice in a column")

        # ! Check for the same number in the same square

        for cell in row:
            if cell < 0 or cell > SUDOKU_SIZE:
                raise Exception("Some cell have numbers that are out of range")


def candidates_to_bits(candidates: set) -> int:
    """
    Convert a candidates set into its bits representation
    The bits are in big endian order

    >>> bin(candidates_to_bits({1, 2, 3, 6, 7}))
    '0b1100111'

    >>> bin(candidates_to_bits({6, 9}))
    '0b100100000'
    """
    bits = 0
    for i in range(SUDOKU_SIZE):
        if i + 1 in candidates:
            bits ^= 1 << i

    return bits


def bits_to_candidates(bits: int) -> set:
    """
    Convert a bits representation into a set of candidates

    >>> bits_to_candidates(0b111)
    {1, 2, 3}

    >>> bits_to_candidates(0b111000101)
    {1, 3, 7, 8, 9}
    """
    candidates = set()
    for i in range(SUDOKU_SIZE):
        bit = bits & (1 << i)
        if bit:
            candidates.add(i + 1)

    return candidates


def contains_zero(array: np.ndarray) -> bool:
    """
    Returns if the 2d array contains zeroes
    If it's not an array, return True

    >>> contains_zero(None)
    True

    >>> grid = np.full((SUDOKU_SIZE, SUDOKU_SIZE), 0)
    >>> contains_zero(grid)
    True

    >>> grid = np.full((SUDOKU_SIZE, SUDOKU_SIZE), 1)
    >>> contains_zero(grid)
    False
    """
    if array is None:
        return True

    for row in array:
        for cell in row:
            if cell == 0:
                return True

    return False


def deep_copy(array: np.ndarray) -> np.ndarray:
    """Returns a deep copy of an numpy ndarray"""
    result = np.ndarray(array.shape, dtype=array.dtype)
    for y in range(len(array)):
        for x in range(len(array[y])):
            result[y][x] = array[y][x]

    return result


def get_cell_candidates(puzzle: np.ndarray, x: int, y: int) -> set:
    """
    Discard the numbers
    that are on the same speficied line, column or square

    Returns the bit representation of the possible candidates
    """
    candidates = {*range(1, SUDOKU_SIZE + 1)}

    row, column, square = get_row_column_square(puzzle, x, y)
    candidates.difference_update(row)
    candidates.difference_update(column)

    candidates.difference_update(square)

    return candidates


def create_cell_candidates(puzzle: np.ndarray) -> np.ndarray:
    cell_candidates = np.full((SUDOKU_SIZE, SUDOKU_SIZE), -1)

    for y in range(len(puzzle)):
        for x in range(len(puzzle[y])):
            if puzzle[y][x] == 0:
                bits = candidates_to_bits(get_cell_candidates(puzzle, x, y))
                cell_candidates[y][x] = bits

    return cell_candidates


def pretty_print(array: np.ndarray) -> None:
    """
    Utility function
    Use it to print the cell_candidates in a binary form
    """

    def to_bit(item):
        if item == -1:
            return "_" * SUDOKU_SIZE
        return bin(item)[2:].zfill(SUDOKU_SIZE)

    print(*[" ".join(lst) for lst in np.vectorize(to_bit)(array)], sep="\n", end="\n\n")
