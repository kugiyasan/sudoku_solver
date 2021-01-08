# A generic sudoku solver in Python

This repo originally started with a Codewars kata, then I just wanted to make the code more efficient and follow the best practices as much as I can.

## Features

- Code following [flake8](https://flake8.pycqa.org/en/latest/) and [black](https://black.readthedocs.io/en/stable/) styling guides
- Test cases written with [pytest](https://docs.pytest.org/en/stable/) (`pytest --doctest-modules` to run the tests)
- [mypy](http://mypy-lang.org/) is used to type check the code. To type check the code, install mypy and data-science-types with `pip3 install mypy data-science-types` (data-science-types add type checking for numpy)
- (Almost) pure functionnal programming
- Can solve any size of sudoku (1x1, 4x4, 9x9, 16x16, 25x25, etc)
- Math, numpy and typing. That's it, no other dependencies to run the code!

## Running the sudoku solver

Make sure you have Python >= 3.8 installed

```bash
git clone https://github.com/kugiyasan/sudoku_solver
cd sudoku_solver
python3 test_cases.py
```

`test_cases.py` contains some sudoku example and their solution

`hard_sudoku_solver.py` contains the recursive function, the entrypoint functions and some class

`utils.py` contains some simple functions needed in order to solve sudoku

## Usage

To generate as many solutions as you want

```python
solutions = generate_solutions(puzzle)
for solution in solutions:
    print(solution)
```

To use the version that I used to complete the Codewars kata (It will raise an error if there is multiple solutions)

```python
print(sudoku_solver(puzzle))
```

## Contributing

~~Nobody will ever contribute so why would I care writing this paragraph~~

Any action will be much appreciated! If you find something that isn't following best practices, a bug, a lack of explanation or anything like that, don't be afraid to create an issue or a pull request!

## License

[MIT](https://choosealicense.com/licenses/mit/) (Check [LICENSE](https://github.com/kugiyasan/sudoku_solver/blob/master/LICENSE) for more informations)
