# Development Guide

## Setup

### Clone and Install

```bash
git clone https://github.com/jeliason/qsp-hpc-tools.git
cd qsp-hpc-tools

# Install with uv (recommended - faster)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Or with pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Pre-commit Hooks

Install pre-commit hooks for automatic code quality checks:

```bash
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

The hooks automatically:
- Format code with Black
- Lint with Ruff (and auto-fix issues)
- Run fast unit tests
- Check for trailing whitespace, merge conflicts, etc.

## Testing

### Running Tests

```bash
# Run all tests (skips HPC integration tests by default)
pytest

# Run with coverage
pytest --cov=qsp_hpc --cov-report=term-missing

# Run specific test file
pytest tests/test_batch_utils.py

# Run HPC integration tests (requires credentials)
pytest -m hpc -v

# Explicitly skip HPC tests
pytest -m "not hpc" -v
```

### Test Categories

1. **Unit Tests** (fast, no I/O)
   - `test_batch_utils.py`: Batch splitting calculations
   - `test_hash_utils.py`: Hash computation and normalization

2. **Integration Tests** (require temp files)
   - `test_simulation_pool.py`: Pool manager with file I/O
   - `test_qsp_simulator.py`: End-to-end workflows

3. **HPC Integration Tests** (marked with `@pytest.mark.hpc`)
   - Require real HPC credentials
   - Connect to actual SLURM cluster
   - Skipped by default

### Test in Fresh Environment (Like CI)

```bash
# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete

# Run tests fresh
pytest tests/
```

## Code Quality

### Manual Commands

```bash
# Format code with black
black qsp_hpc/ tests/

# Lint with ruff
ruff check qsp_hpc/ tests/

# Auto-fix ruff issues
ruff check --fix qsp_hpc/ tests/
```

### Code Style

- **Line length:** 100 characters (configured in `pyproject.toml`)
- **Formatter:** Black
- **Linter:** Ruff (replaces flake8, isort, etc.)
- **Naming:** Snake_case for functions/variables, PascalCase for classes

## Project Structure

```
qsp-hpc-tools/
├── qsp_hpc/                    # Main package
│   ├── simulation/             # Simulation pool and QSP simulator
│   ├── batch/                  # HPC job management
│   └── utils/                  # Hash utilities, logging
├── qsp_hpc/matlab/            # MATLAB HPC workers
├── scripts/                    # Setup and utility scripts
│   ├── setup_credentials.py   # Interactive credential setup
│   └── hpc/                   # HPC environment setup
├── tests/                      # Test suite
├── docs/                       # Documentation
└── pyproject.toml             # Package configuration
```

## Adding New Features

1. **Create feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Write tests first** (TDD approach)
   - Add tests in `tests/`
   - Run `pytest` to verify they fail

3. **Implement feature**
   - Write code in `qsp_hpc/`
   - Run tests frequently

4. **Format and lint**
   ```bash
   black qsp_hpc/ tests/
   ruff check qsp_hpc/ tests/
   pytest
   ```

5. **Commit and push**
   ```bash
   git add -A
   git commit -m "Add feature: description"
   git push origin feature/my-feature
   ```

6. **Submit pull request**

## CI/CD

GitHub Actions runs on every push and PR:
- Tests on Python 3.9, 3.10, 3.11, 3.12
- Code quality checks (ruff, black)
- Coverage reporting

## Debugging Tips

### Import Errors in CI but not Locally

```bash
# Test import directly
python -c "from qsp_hpc.batch.hpc_job_manager import SomeClass"

# Clear cache and test
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
pytest tests/
```

### HPC Job Debugging

```bash
# View logs
qsp-hpc logs --job-id 12345

# Test HPC connection
qsp-hpc test

# Check configuration
qsp-hpc info
```

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag
4. Push to PyPI (maintainers only)

## Contact

- **Issues:** https://github.com/jeliason/qsp-hpc-tools/issues
- **Maintainer:** Joel Eliason
