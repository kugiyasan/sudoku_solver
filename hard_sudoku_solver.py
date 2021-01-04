# To run the doctest, run
# python3 -m pytest --doctest-modules

import numpy as np

# SUDOKU_SIZE should always be a square number
SUDOKU_SIZE = 9


def check_grid_validity(puzzle: list) -> None:
    """
    Raise an error if the given puzzle isn't the right size
    or has number that aren't possible

    This function doesn't check if the sudoku has no or multiple solutions
    ? It might be better to use `raise` instead of `assert`
    ? `assert` will be ignored if the -O flag is passed

    >>> check_grid_validity([])
    Traceback (most recent call last):
    AssertionError: The sudoku doesn't have the right height

    >>> check_grid_validity([[]] * SUDOKU_SIZE)
    Traceback (most recent call last):
    AssertionError: The sudoku doesn't have the right width

    >>> grid = [[-1 for j in range(SUDOKU_SIZE)] for i in range(SUDOKU_SIZE)]
    >>> check_grid_validity(grid)
    Traceback (most recent call last):
    AssertionError: Some cell have numbers that are out of range

    >>> grid = [[1 for j in range(SUDOKU_SIZE)] for i in range(SUDOKU_SIZE)]
    >>> check_grid_validity(grid)
    """
    assert len(puzzle) == SUDOKU_SIZE, "The sudoku doesn't have the right height"
    for row in puzzle:
        assert len(row) == SUDOKU_SIZE, "The sudoku doesn't have the right width"
        for cell in row:
            assert (
                cell >= 0 and cell <= SUDOKU_SIZE
            ), "Some cell have numbers that are out of range"


def get_row_column_square(puzzle: np.ndarray, x: int, y: int) -> tuple:
    """utility function, returns the row, column and square of the puzzle"""
    # TODO write a implementation for puzzle: list[list]
    sqrt = int(SUDOKU_SIZE ** 0.5)
    sq_y = y // sqrt * sqrt
    sq_x = x // sqrt * sqrt

    # TODO remove the of the square cells, they are copies of cells
    # TODO that are already in row or column
    square = puzzle[sq_y : sq_y + sqrt, sq_x : sq_x + sqrt]
    return puzzle[y], puzzle[:, x], square.flat


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


def get_cell_least_candidates(cell_candidates: np.ndarray) -> tuple:
    """
    Returns the coords of the first cell with the smallest amount of candidates
    """
    smallest_len = SUDOKU_SIZE
    coords = (0, 0)

    for y in range(len(cell_candidates)):
        for x in range(len(cell_candidates[y])):
            if cell_candidates[y][x] == -1:
                continue

            candidates_len = bin(cell_candidates[y][x]).count("1")
            if candidates_len < smallest_len:
                smallest_len = candidates_len
                coords = (x, y)

                if smallest_len == 1:
                    return coords

    return coords


def _remove_candidate(cells: np.ndarray, candidate: int) -> None:
    """Remove the candidate from the cells in place"""
    for i in range(len(cells)):
        if cells[i] == -1:
            continue

        # Example of the following code:
        # if we have SUDOKU_SIZE = 9 and candidate = 5,
        # mask       = 0b111111111
        # bit        = 0b000010000
        # mask ^ bit = 0b111101111
        # and so the cell will set to zero the conresponding bit
        mask = (1 << SUDOKU_SIZE) - 1
        bit = 1 << (candidate - 1)
        cells[i] &= mask ^ bit


def generate_cell_candidates(puzzle: np.ndarray):
    cell_candidates = np.full((SUDOKU_SIZE, SUDOKU_SIZE), -1)

    for y in range(len(puzzle)):
        for x in range(len(puzzle[y])):
            if puzzle[y][x] == 0:
                bits = candidates_to_bits(get_cell_candidates(puzzle, x, y))
                cell_candidates[y][x] = bits

    return cell_candidates


class SolveEngine:
    def __init__(self, cell_candidates: np.ndarray, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.cell_candidates = cell_candidates

        self.copy()

    def copy(self) -> None:
        x = self.x
        y = self.y
        rcs = get_row_column_square(self.cell_candidates, x, y)
        self.row, self.column, self.square = rcs

        self.original_row = np.copy(self.row)
        self.original_column = np.copy(self.column)
        self.original_square = np.copy(self.square)

        self.original_cell_candidates = self.cell_candidates[y][x]

    def restore(self) -> None:
        self.cell_candidates[self.y] = self.original_row
        self.cell_candidates[:, self.x] = self.original_column

        sqrt = int(SUDOKU_SIZE ** 0.5)
        sq_y = self.y // sqrt * sqrt
        sq_x = self.x // sqrt * sqrt

        squareArray = self.cell_candidates[sq_y : sq_y + sqrt, sq_x : sq_x + sqrt]
        squareArray.flat = self.original_square

    def remove_candidate(self, candidate: int) -> None:
        _remove_candidate(self.row, candidate)
        _remove_candidate(self.column, candidate)
        _remove_candidate(self.square, candidate)


def solve(puzzle: np.ndarray, cell_candidates: np.ndarray) -> np.ndarray:
    """This is the recursive function to solve the sudoku"""
    # Find the cell with the smallest amount of candidates
    x, y = get_cell_least_candidates(cell_candidates)

    # if every cell has the default value, it means that the puzzle is done
    if cell_candidates[y][x] == -1:
        return puzzle

    # Copy the current row, column and square
    solve_engine = SolveEngine(cell_candidates, x, y)

    # Mark the cell as filled by putting the default value
    candidates = bits_to_candidates(cell_candidates[y][x])

    for candidate in candidates:
        # Fill the cell with a candidate
        puzzle[y][x] = candidate
        cell_candidates[y][x] = -1

        # Remove this candidate from cells on the same row, column or square
        solve_engine.remove_candidate(candidate)

        # Redo the same steps with the new puzzle
        solution = solve(puzzle, cell_candidates)

        if solution is not None:
            return solution

        # Restore the original neighbors candidates
        solve_engine.restore()

    # Every candidate failed, it's time to backtrack
    # ? I don't need to reset puzzle, but it makes things clearer
    puzzle[y][x] = 0
    cell_candidates[y][x] = solve_engine.original_cell_candidates


def sudoku_solver(puzzle: list) -> list:
    """Entry point for the Sudoku solver"""
    puzzle = np.array(puzzle)
    check_grid_validity(puzzle)

    # Compute the candidates for each cell
    # Candidates will be stored
    # as a number from 0 to 2^SUDOKU_SIZE-1 (-1 is used for filled cells)
    cell_candidates = generate_cell_candidates(puzzle)

    # Solve recursively the puzzle
    solution = solve(puzzle, cell_candidates)

    if solution is None:
        raise ValueError("No solution for this sudoku")

    # TODO check for multiple solutions
    return solution.tolist()


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


if __name__ == "__main__":
    # Check to make sure that SUDOKU_SIZE is a square number
    sqrt = int(SUDOKU_SIZE ** 0.5)
    assert sqrt * sqrt == SUDOKU_SIZE

    puzzle = [
        [0, 0, 6, 1, 0, 0, 0, 0, 8],
        [0, 8, 0, 0, 9, 0, 0, 3, 0],
        [2, 0, 0, 0, 0, 5, 4, 0, 0],
        [4, 0, 0, 0, 0, 1, 8, 0, 0],
        [0, 3, 0, 0, 7, 0, 0, 4, 0],
        [0, 0, 7, 9, 0, 0, 0, 0, 3],
        [0, 0, 8, 4, 0, 0, 0, 0, 6],
        [0, 2, 0, 0, 5, 0, 0, 8, 0],
        [1, 0, 0, 0, 0, 2, 5, 0, 0],
    ]

    solution = [
        [3, 4, 6, 1, 2, 7, 9, 5, 8],
        [7, 8, 5, 6, 9, 4, 1, 3, 2],
        [2, 1, 9, 3, 8, 5, 4, 6, 7],
        [4, 6, 2, 5, 3, 1, 8, 7, 9],
        [9, 3, 1, 2, 7, 8, 6, 4, 5],
        [8, 5, 7, 9, 4, 6, 2, 1, 3],
        [5, 9, 8, 4, 1, 3, 7, 2, 6],
        [6, 2, 4, 7, 5, 9, 3, 8, 1],
        [1, 7, 3, 8, 6, 2, 5, 9, 4],
    ]

    print(sudoku_solver(puzzle))
