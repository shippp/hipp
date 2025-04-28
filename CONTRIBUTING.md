# How to contribute

We welcome to new contributions to hipp ! Below is a guide to contributing to hipp step by step, ensuring tests are passing and the documentation is updated.

## Overview: making a contribution

1. Fork `shippp/hipp` and clone your fork repository locally.
2. Set up the development environment **(see section "Setup" below)**,
3. Create a branch for the new feature or bug fix,
4. Make your changes,
5. Add or modify related tests in `tests/` **(see section "Tests" below)**,
7. Commit your changes,
8. Run `pre-commit` separately if not installed as git hook **(see section "Linting" below)**,
9. Push to your fork,
10. Open a pull request from GitHub to discuss and eventually merge.

## Development environment

Hipp use [hatch](https://hatch.pypa.io/latest/) python project manager. It's highly recommended to use it !

### Setup

#### With `hatch`

Clone the git repo and enter in the hatch dev shell (see how to install `hatch` in the [hatch documentation](https://hatch.pypa.io/latest/install/))
```bash
git clone https://github.com/shippp/hipp.git
cd hipp
hatch shell dev
```

### Tests

To run test, simply run

```bash
hatch run dev:pytest

# if you are already in the hatch shell
pytest
```

### Formatting and linting

Install the `pre-commit` hooks by running (see [pre-commit documentation](https://pre-commit.com/)). Which will use `.pre-commit-config.yaml` to verify spelling errors, import sorting, type checking, formatting and linting:
```bash
hatch run dev:pre-commit install

# if you are already in the hatch shell
pre-commit install
```



