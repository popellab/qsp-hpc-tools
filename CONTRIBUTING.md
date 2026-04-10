# Contributing to QSP HPC Tools

Thank you for your interest in contributing! This document explains how to get involved.

## Reporting Issues

- Use [GitHub Issues](https://github.com/jeliason/qsp-hpc-tools/issues) to report bugs or request features
- Include a minimal reproducible example when reporting bugs
- Describe expected vs. actual behavior

## Development Setup

```bash
git clone https://github.com/jeliason/qsp-hpc-tools.git
cd qsp-hpc-tools

# Install with uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Making Changes

1. Create a feature branch from `main`
2. Write tests for your changes (see `tests/` for examples)
3. Run the test suite: `pytest`
4. Format and lint:
   ```bash
   black qsp_hpc/ tests/
   ruff check qsp_hpc/ tests/
   ```
5. Submit a pull request

## Code Style

- **Formatter:** Black (100 char line length)
- **Linter:** Ruff
- **Naming:** snake_case for functions/variables, PascalCase for classes
- Pre-commit hooks enforce these automatically

## Testing

```bash
# All unit/integration tests
pytest

# With coverage
pytest --cov=qsp_hpc --cov-report=term-missing

# HPC integration tests (requires credentials)
pytest -m hpc -v
```

Tests are organized into:
- **Unit tests** — fast, no I/O (`test_batch_utils.py`, `test_hash_utils.py`)
- **Integration tests** — use temp files (`test_simulation_pool.py`, `test_qsp_simulator.py`)
- **HPC tests** — require real cluster access, skipped by default

## Pull Request Guidelines

- Keep PRs focused on a single change
- Include tests for new functionality
- Ensure CI passes before requesting review
- Update documentation if behavior changes

## Questions?

Open an issue or reach out to the maintainer.
