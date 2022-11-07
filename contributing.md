# Contributing / Pull requests

For information on how to create a pull request refer to
https://github.com/MarcDiethelm/contributing


# Installs
*required*

pip install
- `black` # auto-formatter
- `isort` # automatically sort imports
- `flake8` # style guide
- `pytype` # static type checker
- `pytest` # runs tests

*optional*

- `pytest-xdist` : enables parallel support for pytest
- `pytest-cov` : running `pytest --cov` will give code coverage (of tests)

# Execution
Run in root-dir of repo in just this order.

> black cc 

> isort cc

> flake8

> pytype --config pytype.cfg

> pytest
