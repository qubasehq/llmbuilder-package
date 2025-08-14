# Testing Guide

This guide covers testing practices, conventions, and tools used in LLMBuilder development.

## Testing Philosophy

LLMBuilder follows a comprehensive testing approach:

- **Unit Tests**: Test individual functions and classes in isolation
- **Integration Tests**: Test component interactions and workflows
- **Performance Tests**: Ensure performance requirements are met
- **CLI Tests**: Test command-line interface functionality

## Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_config.py
│   ├── test_data_processing.py
│   └── test_tokenizer.py
├── integration/             # Integration tests
│   ├── test_pipeline_workflow.py
│   ├── test_data_processing.py
│   └── test_tokenizer_training_integration.py
├── performance/             # Performance tests
│   ├── test_processing_speed.py
│   └── test_memory_usage.py
├── cli/                     # CLI tests
│   └── test_cli_commands.py
├── fixtures/                # Test data and fixtures
│   ├── sample_data/
│   └── conftest.py
└── conftest.py              # Global test configuration
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v           # Unit tests only
python -m pytest tests/integration/ -v    # Integration tests only
python -m pytest tests/performance/ -v    # Performance tests only

# Run specific test file
python -m pytest tests/test_config.py -v

# Run specific test function
python -m pytest tests/test_config.py::test_config_validation -v
```

### Test Markers

```bash
# Run tests by marker
python -m pytest -m "not slow" -v         # Skip slow tests
python -m pytest -m "integration" -v      # Integration tests only
python -m pytest -m "performance" -v      # Performance tests only
python -m pytest -m "optional_deps" -v    # Tests requiring optional deps

# Run slow tests (requires environment variable)
RUN_SLOW=1 python -m pytest -m "slow" -v

# Run performance tests (requires environment variable)
RUN_PERF=1 python -m pytest -m "performance" -v
```

### Coverage Reports

```bash
# Run tests with coverage
python -m pytest tests/ --cov=llmbuilder --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Writing Tests

### Unit Test Example

```python
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from llmbuilder.config import Config, ModelConfig
from llmbuilder.data.processor import DocumentProcessor

class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Config()
        self.processor = DocumentProcessor(self.config)

    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_process_valid_document_success(self):
        """Test processing valid document returns success."""
        # Arrange
        content = "This is test content for processing."
        test_file = self.temp_dir / "test.txt"
        test_file.write_text(content)

        # Act
        result = self.processor.process_file(str(test_file))

        # Assert
        assert result.success is True
        assert result.items_processed > 0
        assert result.error_message is None

    def test_process_nonexistent_file_raises_error(self):
        """Test processing non-existent file raises FileNotFoundError."""
        # Arrange
        nonexistent_file = str(self.temp_dir / "nonexistent.txt")

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            self.processor.process_file(nonexistent_file)

    @pytest.mark.parametrize("content,expected_lines", [
        ("Single line", 1),
        ("Line 1\nLine 2", 2),
        ("", 0),
        ("Line 1\n\nLine 3", 2),  # Empty lines ignored
    ])
    def test_process_content_line_counting(self, content, expected_lines):
        """Test that content processing counts lines correctly."""
        # Arrange
        test_file = self.temp_dir / "test.txt"
        test_file.write_text(content)

        # Act
        result = self.processor.process_file(str(test_file))

        # Assert
        assert result.items_processed == expected_lines

    @patch('llmbuilder.data.processor.external_service_call')
    def test_process_with_mocked_external_service(self, mock_service):
        """Test processing with mocked external dependencies."""
        # Arrange
        mock_service.return_value = {"status": "success", "data": "processed"}
        test_file = self.temp_dir / "test.txt"
        test_file.write_text("test content")

        # Act
        result = self.processor.process_file(str(test_file))

        # Assert
        assert result.success is True
        mock_service.assert_called_once()
```

### Integration Test Example

```python
import pytest
from pathlib import Path
import tempfile
import shutil

from llmbuilder.data.ingest import IngestionPipeline
from llmbuilder.data.dedup import DeduplicationPipeline
from llmbuilder.config.manager import create_config_from_template

class TestDataProcessingPipeline:
    """Integration tests for complete data processing pipeline."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_processing_pipeline(self):
        """Test complete pipeline from ingestion to deduplication."""
        # Arrange - Create test documents
        self._create_test_documents()

        config = create_config_from_template("basic_config", {
            "data": {
                "ingestion": {"batch_size": 10},
                "deduplication": {"similarity_threshold": 0.8}
            }
        })

        # Act - Run ingestion pipeline
        ingestion = IngestionPipeline(config.data.ingestion)
        ingestion_result = ingestion.process_directory(
            str(self.input_dir),
            str(self.output_dir / "ingested.txt")
        )

        # Act - Run deduplication pipeline
        dedup = DeduplicationPipeline(config.data.deduplication)
        dedup_result = dedup.process_file(
            str(self.output_dir / "ingested.txt"),
            str(self.output_dir / "deduplicated.txt")
        )

        # Assert
        assert ingestion_result["success"] is True
        assert ingestion_result["files_processed"] > 0

        assert dedup_result["success"] is True
        assert dedup_result["final_count"] <= dedup_result["original_count"]

        # Verify output files exist
        assert (self.output_dir / "ingested.txt").exists()
        assert (self.output_dir / "deduplicated.txt").exists()

    def _create_test_documents(self):
        """Create test documents for pipeline testing."""
        documents = [
            ("doc1.txt", "This is the first test document."),
            ("doc2.html", "<html><body><p>HTML document content</p></body></html>"),
            ("doc3.txt", "This is the first test document."),  # Duplicate
            ("doc4.txt", "Another unique document with different content."),
        ]

        for filename, content in documents:
            (self.input_dir / filename).write_text(content)

@pytest.mark.slow
@pytest.mark.integration
class TestTokenizerTrainingIntegration:
    """Integration tests for tokenizer training workflow."""

    @pytest.mark.optional_deps
    def test_sentencepiece_training_workflow(self):
        """Test complete SentencePiece tokenizer training workflow."""
        # This test requires sentence-transformers
        pytest.importorskip("sentencepiece")

        # Test implementation here
        pass
```

### Performance Test Example

```python
import pytest
import time
import psutil
import os
from pathlib import Path

@pytest.mark.performance
class TestProcessingPerformance:
    """Performance tests for data processing components."""

    def test_large_file_processing_performance(self):
        """Test that large file processing meets performance requirements."""
        # Skip if performance tests not enabled
        if not os.getenv("RUN_PERF"):
            pytest.skip("Performance tests not enabled (set RUN_PERF=1)")

        # Arrange
        large_content = "Test line content.\n" * 10000  # 10k lines
        test_file = Path("large_test_file.txt")
        test_file.write_text(large_content)

        try:
            # Act
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss

            # Process the large file
            from llmbuilder.data.processor import DocumentProcessor
            processor = DocumentProcessor()
            result = processor.process_file(str(test_file))

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            # Assert performance requirements
            processing_time = end_time - start_time
            memory_used = end_memory - start_memory

            assert processing_time < 5.0, f"Processing took {processing_time:.2f}s, expected < 5s"
            assert memory_used < 100 * 1024 * 1024, f"Used {memory_used / 1024 / 1024:.1f}MB, expected < 100MB"
            assert result.success is True

        finally:
            test_file.unlink(missing_ok=True)

    @pytest.mark.parametrize("file_size", [1000, 5000, 10000])
    def test_processing_scales_linearly(self, file_size):
        """Test that processing time scales roughly linearly with file size."""
        if not os.getenv("RUN_PERF"):
            pytest.skip("Performance tests not enabled")

        # Implementation here
        pass
```

### CLI Test Example

```python
import pytest
from click.testing import CliRunner
from pathlib import Path
import tempfile

from llmbuilder.cli import main

class TestCLICommands:
    """Test cases for CLI commands."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cli_help_command(self):
        """Test that help command works."""
        result = self.runner.invoke(main, ['--help'])

        assert result.exit_code == 0
        assert "LLMBuilder" in result.output
        assert "Usage:" in result.output

    def test_cli_info_command(self):
        """Test info command displays package information."""
        result = self.runner.invoke(main, ['info'])

        assert result.exit_code == 0
        assert "LLMBuilder" in result.output
        assert "version" in result.output.lower()

    def test_config_templates_command(self):
        """Test config templates command lists available templates."""
        result = self.runner.invoke(main, ['config', 'templates'])

        assert result.exit_code == 0
        assert "basic_config" in result.output
        assert "cpu_optimized_config" in result.output

    def test_config_validate_command_with_valid_config(self):
        """Test config validation with valid configuration."""
        # Create valid config file
        config_content = '{"model": {"vocab_size": 16000}}'
        config_file = self.temp_dir / "valid_config.json"
        config_file.write_text(config_content)

        result = self.runner.invoke(main, ['config', 'validate', str(config_file)])

        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_config_validate_command_with_invalid_config(self):
        """Test config validation with invalid configuration."""
        # Create invalid config file
        config_content = '{"invalid": "json"'  # Missing closing brace
        config_file = self.temp_dir / "invalid_config.json"
        config_file.write_text(config_content)

        result = self.runner.invoke(main, ['config', 'validate', str(config_file)])

        assert result.exit_code != 0
        assert "failed" in result.output.lower() or "error" in result.output.lower()
```

## Test Fixtures and Utilities

### Global Fixtures (conftest.py)

```python
# tests/conftest.py
import pytest
import tempfile
from pathlib import Path
import shutil

@pytest.fixture(scope="session")
def sample_data_dir():
    """Provide directory with sample test data."""
    return Path(__file__).parent / "fixtures" / "sample_data"

@pytest.fixture
def temp_directory():
    """Provide temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def sample_config():
    """Provide sample configuration for tests."""
    from llmbuilder.config import Config, ModelConfig
    return Config(
        model=ModelConfig(
            vocab_size=8000,
            num_layers=4,
            num_heads=4,
            embedding_dim=256
        )
    )

@pytest.fixture
def mock_file_system(temp_directory):
    """Create mock file system structure for testing."""
    # Create directory structure
    (temp_directory / "input").mkdir()
    (temp_directory / "output").mkdir()

    # Create sample files
    (temp_directory / "input" / "test1.txt").write_text("Sample content 1")
    (temp_directory / "input" / "test2.txt").write_text("Sample content 2")

    return temp_directory

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "optional_deps: marks tests that require optional dependencies"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers and environment variables."""
    import os

    # Skip slow tests unless RUN_SLOW is set
    if not os.getenv("RUN_SLOW"):
        skip_slow = pytest.mark.skip(reason="Slow tests disabled (set RUN_SLOW=1 to enable)")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Skip performance tests unless RUN_PERF is set
    if not os.getenv("RUN_PERF"):
        skip_perf = pytest.mark.skip(reason="Performance tests disabled (set RUN_PERF=1 to enable)")
        for item in items:
            if "performance" in item.keywords:
                item.add_marker(skip_perf)
```

### Test Utilities

```python
# tests/utils.py
from pathlib import Path
from typing import List, Dict, Any
import json

def create_sample_documents(directory: Path, documents: List[Dict[str, str]]) -> List[Path]:
    """Create sample documents for testing."""
    created_files = []

    for doc in documents:
        file_path = directory / doc["filename"]
        file_path.write_text(doc["content"])
        created_files.append(file_path)

    return created_files

def create_test_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create test configuration with optional overrides."""
    base_config = {
        "model": {
            "vocab_size": 8000,
            "num_layers": 4,
            "num_heads": 4,
            "embedding_dim": 256
        },
        "training": {
            "batch_size": 8,
            "num_epochs": 2,
            "learning_rate": 1e-3
        }
    }

    if overrides:
        # Deep merge overrides
        def deep_merge(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value

        deep_merge(base_config, overrides)

    return base_config

def assert_file_contains(file_path: Path, expected_content: str):
    """Assert that file contains expected content."""
    assert file_path.exists(), f"File does not exist: {file_path}"
    content = file_path.read_text()
    assert expected_content in content, f"Expected content not found in {file_path}"

def assert_processing_result_valid(result):
    """Assert that processing result has valid structure."""
    assert hasattr(result, 'success')
    assert hasattr(result, 'items_processed')
    assert hasattr(result, 'processing_time')
    assert isinstance(result.success, bool)
    assert isinstance(result.items_processed, int)
    assert isinstance(result.processing_time, (int, float))
    assert result.items_processed >= 0
    assert result.processing_time >= 0
```

## Test Configuration

### pytest.ini

```ini
[tool:pytest]
minversion = 7.0
addopts =
    -ra
    --strict-markers
    --strict-config
    --cov=llmbuilder
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
testpaths = tests
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    performance: marks tests as performance tests
    gpu: marks tests that require GPU
    optional_deps: marks tests that require optional dependencies
filterwarnings =
    error
    ignore::UserWarning
    ignore::DeprecationWarning
```

## Continuous Integration

### GitHub Actions Test Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: [3.8, 3.9, "3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install system dependencies (Ubuntu)
      if: matrix.os == 'ubuntu-latest'
      run: |
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-eng

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"

    - name: Run tests
      run: |
        python -m pytest tests/ -v --cov=llmbuilder --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

## Best Practices

### Test Organization

1. **One test class per class being tested**
2. **Group related tests together**
3. **Use descriptive test names**
4. **Follow AAA pattern (Arrange, Act, Assert)**
5. **Keep tests independent and isolated**

### Test Data Management

1. **Use fixtures for common test data**
2. **Create minimal test data**
3. **Clean up after tests**
4. **Use temporary directories for file operations**
5. **Mock external dependencies**

### Performance Testing

1. **Set clear performance requirements**
2. **Test with realistic data sizes**
3. **Monitor memory usage**
4. **Use environment variables to control execution**
5. **Document performance expectations**

### Mocking Guidelines

1. **Mock external services and APIs**
2. **Mock file system operations when appropriate**
3. **Don't mock the code you're testing**
4. **Use specific assertions on mocks**
5. **Reset mocks between tests**

## Debugging Tests

### Running Tests in Debug Mode

```bash
# Run with verbose output
python -m pytest tests/ -v -s

# Run specific test with debugging
python -m pytest tests/test_specific.py::test_function -v -s --pdb

# Run with coverage and keep temporary files
python -m pytest tests/ --cov=llmbuilder --keep-temp-files
```

### Common Issues

1. **Test isolation**: Ensure tests don't depend on each other
2. **Resource cleanup**: Always clean up temporary files and resources
3. **Mock configuration**: Ensure mocks are properly configured and reset
4. **Environment dependencies**: Handle optional dependencies gracefully
5. **Timing issues**: Use appropriate timeouts and retries for async operations

---

Following these testing practices ensures reliable, maintainable tests that provide confidence in code changes and help catch regressions early.
