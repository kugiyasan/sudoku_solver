from hard_sudoku_solver import sudoku_solver, generate_solutions

if __name__ == "__main__":
    puzzle1 = [
        [0, 1, 0, 4],
        [4, 0, 0, 0],
        [0, 0, 0, 2],
        [2, 0, 3, 0],
    ]
    solution1 = [
        [3, 1, 2, 4],
        [4, 2, 1, 3],
        [1, 3, 4, 2],
        [2, 4, 3, 1],
    ]

    puzzle2 = [
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
    solution2 = [
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

    puzzle3 = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
    solution3 = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ]

    A = 10
    B = 11
    C = 12
    D = 13
    E = 14
    F = 15
    G = 16

    puzzle4 = [
        [0, 0, 0, 0, G, C, 2, 0, 0, 7, 0, 0, 0, 0, 0, 0],
        [0, 0, 8, 0, 0, F, 3, 9, 0, 1, 0, 5, 4, 6, 0, 0],
        [B, 0, 0, E, D, 4, 0, 0, 6, 2, 0, 0, 3, A, 0, 5],
        [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, G, 0, 0, F, 0, 0],
        [0, 0, E, A, 0, 6, 4, 0, 0, 0, 0, 0, 0, 0, 7, 8],
        [0, 0, 0, 1, A, 9, 0, 3, 4, 0, 0, 0, 0, 0, 0, 6],
        [0, 4, 2, 0, 0, 0, 0, 0, 8, 3, 5, 0, A, 0, D, 9],
        [0, 0, 0, 9, 0, 5, 0, 0, C, 0, D, 0, E, 0, 0, 3],
        [G, 0, 0, 0, 3, 0, C, 0, 0, 0, 0, 2, D, 0, 0, 0],
        [0, 7, 0, 0, 2, A, B, F, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 8, 0, 0, 0, 0, 7, G, 3, 0, 1, B, 0, 2, 0, 0],
        [1, 0, 0, 0, 0, D, 0, 0, F, 0, 0, 0, 0, C, 0, G],
        [0, 2, 5, 0, 8, 0, G, 0, 0, 0, F, 0, 9, 0, 0, 0],
        [A, 0, 3, B, 0, 0, 0, 0, E, D, 0, 0, 0, 0, 8, 0],
        [9, 0, 0, 0, E, 0, A, 0, 0, G, 0, 0, 5, 0, F, 0],
        [0, 0, 1, G, 0, 0, 6, 0, 2, 0, 0, C, 0, B, 4, 0],
    ]

    # solution = sudoku_solver(puzzle1)
    solution = generate_solutions(puzzle4)
    # print(solution)
    print(next(solution))
    # print(next(solution))

    # solution = sudoku_solver(puzzle2)
    # print(solution)
    # print(next(solution))
    # print(next(solution))

    # import timeit

    # times = timeit.repeat(lambda: sudoku_solver(puzzle2), number=10)
    # print(times)
