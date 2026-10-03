# Developer Guide

This guide provides comprehensive information for developers working on the **Time Series Forecasting** project.

## Development Environment

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (latest version; installs Python for you)
- Git
- A code editor (VS Code recommended)

### Setting Up

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd time-series-forecasting
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

## Code Style

We follow strict code style guidelines to maintain code quality and consistency.

### Python Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all modules, classes, and functions
- Keep lines under 88 characters (ruff's default)

### Code Formatting

We use the following tools for code formatting:

- **Ruff**: For formatting (`ruff format`) and linting (`ruff check`)
- **MyPy**: For type checking

Run the formatters:
```bash
uv run ruff format .
uv run ruff check .
uv run mypy src
```

### Pre-commit Hooks

The following pre-commit hooks are configured:
- `ruff-check`: Linting (with auto-fix)
- `ruff-format`: Code formatting
- `mypy`: Type checking

## Testing

### Running Tests

Run the test suite:
```bash
uv run pytest
```

Run tests with coverage:
```bash
uv run pytest --cov=src
```

### Writing Tests

- Write tests for all new functionality
- Follow the `tests/` directory structure
- Use pytest fixtures for common setup
- Aim for high test coverage

Example test structure:
```python
def test_function_name():
    # Arrange
    input_data = ...
    expected_output = ...
    
    # Act
    result = function_name(input_data)
    
    # Assert
    assert result == expected_output
```

## Documentation

### Code Documentation

- Use Google-style docstrings
- Include type hints
- Document all parameters and return values
- Provide usage examples

Example:
```python
def process_data(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Process the input data by applying transformations.
    
    Args:
        data: Input DataFrame to process
        column: Name of the column to transform
    
    Returns:
        Processed DataFrame with transformed column
    
    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 3]})
        >>> process_data(df, 'A')
           A  A_transformed
        0  1             2
        1  2             4
        2  3             6
    """
    # Implementation
```

### Project Documentation

- Keep `README.md` up to date
- Update documentation when adding new features
- Include examples in documentation

## Git Workflow

### Branching Strategy

- `main`: Production-ready code
- `dev`: Development branch
- Feature branches: `feature/feature-name`
- Bug fix branches: `fix/bug-name`
- Release branches: `release/version`

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

### Pull Requests

1. Create a feature branch
2. Make changes
3. Run tests and formatters
4. Create a pull request
5. Request review
6. Address feedback
7. Merge after approval

## Data Science Best Practices

### Data Management

- Use DVC for data versioning
- Keep raw data immutable
- Document data processing steps
- Use appropriate data formats

### Model Development

- Use MLflow for experiment tracking
- Document model parameters
- Save model artifacts
- Version control model files

### Visualization

- Use consistent styling
- Save figures in appropriate formats
- Include proper labels and titles
- Document visualization code

## Common Tasks

### Adding Dependencies

Add a new dependency:
```bash
uv add package-name
```

Add a development dependency:
```bash
uv add --group dev package-name
```

### Running Jupyter

Start Jupyter Lab:
```bash
uv run --with jupyterlab jupyter lab
```

## Troubleshooting

### Common Issues

1. **Virtual Environment Issues**
   - Delete `.venv` directory
   - Run `uv sync` again

2. **Pre-commit Hook Failures**
   - Run `uv run pre-commit run --all-files`

3. **Test Failures**
   - Check test data
   - Verify dependencies
   - Check environment variables

### Getting Help

- Check the documentation
- Search existing issues
- Create a new issue if needed
- Ask in the project chat

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and formatters
5. Submit a pull request

## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](../LICENSE) file for details.
