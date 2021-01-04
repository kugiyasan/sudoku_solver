# To run the doctest, run
# python3 -m pytest --doctest-modules

import numpy as np
from utils import (
    bits_to_candidates,
    check_grid_validity,
    contains_zero,
    create_cell_candidates,
    deep_copy,
    get_row_column_square,
    SUDOKU_SIZE,
    set_sudoku_size,
)


first_solution = None


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
        # if we have SUDOKU_SIZE = 9 and candidate = 4,
        # mask       = 0b111111111
        # bit        = 0b000001000
        # mask ^ bit = 0b111110111
        # and so the cell will set to zero the conresponding bit
        mask = (1 << SUDOKU_SIZE) - 1
        bit = 1 << (candidate - 1)
        cells[i] &= mask ^ bit


class SolveEngine:
    def __init__(self, cell_candidates: np.ndarray, x: int, y: int) -> None:
        """
        A class that helps to save and restore the state of row, column and square
        """
        self.x = x
        self.y = y
        self.cell_candidates = cell_candidates

        self.copy()

    def copy(self) -> None:
        """Make a copy of the row, column and square"""
        x = self.x
        y = self.y
        rcs = get_row_column_square(self.cell_candidates, x, y)
        self.row, self.column, self.square = rcs

        self.original_row = np.copy(self.row)
        self.original_column = np.copy(self.column)
        self.original_square = np.copy(self.square)

        self.original_cell_candidates = self.cell_candidates[y][x]

    def restore(self) -> None:
        """Restore the row, column and square that was previously saved"""
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

    # If every cell has the default value, it means that the puzzle is done
    if cell_candidates[y][x] == -1:
        # If it's the first solution, store it and continue
        global first_solution
        if first_solution is None:
            first_solution = deep_copy(puzzle)
            return
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

    # Every candidate failed, we need to do backtracking
    # ? I don't need to reset puzzle, but it makes things clearer
    puzzle[y][x] = 0
    cell_candidates[y][x] = solve_engine.original_cell_candidates


def sudoku_solver(puzzle: list) -> list:
    """Entry point for the Sudoku solver"""
    set_sudoku_size(len(puzzle))

    puzzle = np.array(puzzle)
    check_grid_validity(puzzle)

    global first_solution
    first_solution = None

    # Compute the candidates for each cell
    # Candidates will be stored
    # as a number from 0 to 2^SUDOKU_SIZE-1 (-1 is used for filled cells)
    cell_candidates = create_cell_candidates(puzzle)

    # Solve recursively the puzzle
    second_solution = solve(puzzle, cell_candidates)

    if first_solution is None:
        raise ValueError("No solution for this sudoku")
    if not contains_zero(second_solution):
        raise ValueError("Multiple solutions for this sudoku")
    return first_solution.tolist()


if __name__ == "__main__":
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

    # print(sudoku_solver(puzzle))

    def stmt():
        sudoku_solver(puzzle)

    import timeit

    times = timeit.repeat(stmt, number=10)
    print(times)
