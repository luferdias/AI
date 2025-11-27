# Contributing to Concrete Crack Detection Pipeline

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs
- Include detailed reproduction steps
- Provide system information (OS, Python version, GPU)
- Include error messages and logs

### Feature Requests
- Open an issue with the "enhancement" label
- Describe the feature and its use case
- Explain why it would be valuable

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/luferdias/AI.git
   cd AI
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the code style (PEP 8)
   - Add tests for new features
   - Update documentation

4. **Run tests and linters**
   ```bash
   make test
   make lint
   make format
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

### Python
- Follow PEP 8
- Use Black for formatting
- Use isort for imports
- Maximum line length: 100 characters
- Add type hints where appropriate

### Documentation
- Docstrings for all public functions/classes
- Google-style docstrings
- Update README.md if needed

### Testing
- Write unit tests for new functionality
- Maintain >80% code coverage
- Use pytest fixtures appropriately

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest pytest-cov black flake8 isort mypy

# Run setup
python setup.py
```

## Project Structure

```
src/
├── data/       # Data pipeline modules
├── models/     # Model implementations
├── api/        # API service
└── utils/      # Utility functions

tests/
├── unit/       # Unit tests
└── integration/  # Integration tests
```

## Review Process

1. All PRs require review
2. Must pass CI/CD checks
3. Must maintain code coverage
4. Documentation must be updated

## Questions?

Open an issue or contact the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
