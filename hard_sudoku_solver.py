# To run the doctest, run
# python3 -m pytest --doctest-modules

import numpy as np
from typing import Generator, List, Sequence, Tuple
from utils import (
    bits_to_candidates,
    check_grid_validity,
    create_cell_candidates,
    get_row_column_square,
    SudokuError,
)


def get_cell_least_candidates(cell_candidates: np.ndarray) -> Tuple[int, int]:
    """
    Returns the coords of the first cell with the smallest amount of candidates
    """
    smallest_len = cell_candidates.shape[0]
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


def _remove_candidate(cells, candidate: int, SUDOKU_SIZE: int) -> None:
    """
    Remove the candidate from the cells in place

    cells: an object supporting __len__ and __getitem__ (np.ndarray or np.flatiter)
    """
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

        self.SUDOKU_SIZE = cell_candidates.shape[0]
        self.sqrt = int(self.SUDOKU_SIZE ** 0.5)

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

        sqrt = self.sqrt
        sq_y = self.y // sqrt * sqrt
        sq_x = self.x // sqrt * sqrt

        squareArray = self.cell_candidates[sq_y : sq_y + sqrt, sq_x : sq_x + sqrt]
        squareArray.flat = self.original_square

    def remove_candidate(self, candidate: int) -> None:
        _remove_candidate(self.row, candidate, self.SUDOKU_SIZE)
        _remove_candidate(self.column, candidate, self.SUDOKU_SIZE)
        _remove_candidate(self.square, candidate, self.SUDOKU_SIZE)


def recursive_solve(
    puzzle: np.ndarray,
    cell_candidates: np.ndarray,
) -> Generator[np.ndarray, None, None]:
    """This is the recursive function to solve the sudoku"""
    # Find the cell with the smallest amount of candidates
    x, y = get_cell_least_candidates(cell_candidates)

    # If every cell has the default value, it means that the puzzle is done
    if cell_candidates[y][x] == -1:
        yield puzzle
        return

    # Copy the current row, column and square
    solve_engine = SolveEngine(cell_candidates, x, y)

    candidates = bits_to_candidates(cell_candidates[y][x])

    for candidate in candidates:
        # Fill the cell with a candidate
        puzzle[y][x] = candidate
        cell_candidates[y][x] = -1

        # Remove this candidate from cells on the same row, column or square
        solve_engine.remove_candidate(candidate)

        # Redo the same steps with the new puzzle
        yield from recursive_solve(puzzle, cell_candidates)

        # Restore the original neighbors candidates
        solve_engine.restore()

    # Every candidate failed, we need to do backtracking
    # ? I don't need to reset puzzle, but it makes things clearer
    puzzle[y][x] = 0
    cell_candidates[y][x] = solve_engine.original_cell_candidates


def sudoku_solver(puzzle: List[List[int]]) -> Sequence[int]:
    """Entry point for Codewars kata"""
    solution_generator = generate_solutions(puzzle)

    first_solution = next(solution_generator)
    if first_solution is None:
        raise SudokuError("No solution for this sudoku")

    first_solution = np.array(first_solution, copy=True)  # type: ignore

    # Raise an error if there is a second solution
    try:
        next(solution_generator)
        raise SudokuError("Multiple solutions for this sudoku")
    except StopIteration:
        pass

    return first_solution.tolist()


def generate_solutions(
    original_puzzle: List[List[int]],
) -> Generator[np.ndarray, None, None]:
    """
    A more general entry point for the sudoku solver
    Returns a generator with every solution
    """
    puzzle = np.array(original_puzzle)
    check_grid_validity(puzzle)

    cell_candidates = create_cell_candidates(puzzle)

    # Solve recursively the puzzle
    return recursive_solve(puzzle, cell_candidates)
