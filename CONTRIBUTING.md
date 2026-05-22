# Contributing Guide

Thank you for your interest in contributing to the Chemical Property Prediction project!

## How to Contribute

### Reporting Bugs
- Check if the bug has already been reported in [Issues](https://github.com/username/chemical-property-predictor/issues)
- Include Python version, OS, and steps to reproduce
- Include error messages and traceback

### Suggesting Features
- Open an issue with the `enhancement` label
- Describe the feature and its use case
- Discuss implementation approach if possible

### Pull Requests

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/chemical-property-predictor.git
   cd chemical-property-predictor
   ```

2. **Create Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Install Dev Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8 mypy
   ```

4. **Make Changes**
   - Follow PEP 8 style guide
   - Add docstrings to functions and classes
   - Add type hints
   - Include unit tests

5. **Run Tests**
   ```bash
   pytest tests/ -v
   black src/ tests/
   flake8 src/ tests/
   ```

6. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

7. **Create Pull Request**
   - Describe your changes
   - Reference related issues
   - Request review from maintainers

## Code Standards

### Style Guide
- Follow PEP 8
- Use `black` for formatting (line length: 88)
- Maximum line length: 120 characters

### Documentation
- Use Google-style docstrings
- Include type hints
- Add docstring examples for public functions

### Testing
- Write tests for new features
- Maintain >80% code coverage
- Use pytest fixtures for test data

## Commit Message Format

```
type: description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Questions?

Open an issue or contact the maintainers.

Thank you for contributing!
