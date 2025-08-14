# Code Style Guide

This guide outlines the coding standards and style conventions for LLMBuilder to ensure consistent, readable, and maintainable code.

## Overview

LLMBuilder follows [PEP 8](https://pep8.org/) with some modifications and uses automated tools to enforce consistency:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for style checking
- **mypy** for type checking
- **pre-commit** for automated checks

## Automated Formatting

### Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### Usage

```bash
# Format code automatically
black llmbuilder/ tests/ examples/
isort llmbuilder/ tests/ examples/

# Check style
flake8 llmbuilder/ tests/
mypy llmbuilder/

# Run all pre-commit checks
pre-commit run --all-files
```

## Python Code Style

### General Principles

1. **Readability counts** - Code is read more often than written
2. **Explicit is better than implicit** - Be clear about intentions
3. **Simple is better than complex** - Prefer straightforward solutions
4. **Consistency matters** - Follow established patterns in the codebase

### Naming Conventions

```python
# Classes: PascalCase
class DataProcessor:
    pass

class HTMLProcessor(DataProcessor):
    pass

# Functions and methods: snake_case
def process_documents():
    pass

def load_configuration_file():
    pass

# Variables: snake_case
input_file = "data.txt"
processing_results = []
user_config = {}

# Constants: UPPER_CASE
MAX_BATCH_SIZE = 1000
DEFAULT_TIMEOUT = 30
SUPPORTED_FORMATS = ["html", "pdf", "txt"]

# Private attributes: leading underscore
class MyClass:
    def __init__(self):
        self._internal_state = {}
        self.__private_data = []  # Name mangling for truly private

# Type variables: PascalCase with T suffix
from typing import TypeVar
T = TypeVar('T')
ConfigT = TypeVar('ConfigT', bound='BaseConfig')
```

### Function and Method Design

```python
def process_file(
    input_path: str,
    output_path: str,
    config: Optional[ProcessingConfig] = None,
    *,  # Force keyword-only arguments after this
    validate_output: bool = True,
    progress_callback: Optional[Callable[[int], None]] = None
) -> ProcessingResult:
    """
    Process a single file with the specified configuration.

    Args:
        input_path: Path to the input file
        output_path: Path where processed output will be saved
        config: Processing configuration (uses default if None)
        validate_output: Whether to validate the output file
        progress_callback: Optional callback for progress updates

    Returns:
        ProcessingResult with statistics and status information

    Raises:
        FileNotFoundError: If input_path doesn't exist
        PermissionError: If unable to write to output_path
        ProcessingError: If processing fails

    Example:
        >>> config = ProcessingConfig(batch_size=100)
        >>> result = process_file("input.txt", "output.txt", config)
        >>> print(f"Processed {result.items_count} items")
    """
    # Implementation here
    pass
```

### Class Design

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Use dataclasses for simple data containers
@dataclass
class ProcessingResult:
    """Result of a processing operation."""
    success: bool
    items_processed: int
    processing_time: float
    error_message: Optional[str] = None

    def __post_init__(self):
        """Validate the result after initialization."""
        if self.items_processed < 0:
            raise ValueError("items_processed cannot be negative")

# Use ABC for base classes with abstract methods
class BaseProcessor(ABC):
    """Base class for all document processors."""

    def __init__(self, config: ProcessorConfig):
        self.config = config
        self._stats = ProcessingStats()

    @abstractmethod
    def process_file(self, file_path: str) -> ProcessingResult:
        """Process a single file."""
        pass

    @property
    def stats(self) -> ProcessingStats:
        """Get processing statistics."""
        return self._stats

# Use Protocol for structural typing
@runtime_checkable
class Configurable(Protocol):
    """Protocol for objects that can be configured."""

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the object with the given settings."""
        ...

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        ...
```

### Error Handling

```python
# Define custom exceptions
class LLMBuilderError(Exception):
    """Base exception for LLMBuilder."""
    pass

class ProcessingError(LLMBuilderError):
    """Raised when processing fails."""
    pass

class ConfigurationError(LLMBuilderError):
    """Raised when configuration is invalid."""
    pass

# Use specific exception handling
def process_document(file_path: str) -> ProcessingResult:
    """Process a document with proper error handling."""
    try:
        # Validate input
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Process file
        result = _do_processing(file_path)

        # Validate result
        if not result.success:
            raise ProcessingError(f"Processing failed: {result.error_message}")

        return result

    except FileNotFoundError:
        # Re-raise file system errors as-is
        raise
    except ProcessingError:
        # Re-raise processing errors as-is
        raise
    except Exception as e:
        # Wrap unexpected errors
        raise ProcessingError(f"Unexpected error processing {file_path}: {e}") from e

# Use context managers for resource management
from contextlib import contextmanager

@contextmanager
def processing_context(config: ProcessingConfig):
    """Context manager for processing operations."""
    logger.info("Starting processing context")
    try:
        # Setup
        setup_processing(config)
        yield
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
    finally:
        # Cleanup
        cleanup_processing()
        logger.info("Processing context closed")
```

### Type Hints

```python
from typing import (
    Any, Dict, List, Optional, Union, Tuple, Callable,
    TypeVar, Generic, Protocol, Literal, overload
)
from pathlib import Path

# Use specific types
def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from file."""
    pass

# Use generics for reusable code
T = TypeVar('T')

class Cache(Generic[T]):
    """Generic cache implementation."""

    def __init__(self) -> None:
        self._data: Dict[str, T] = {}

    def get(self, key: str) -> Optional[T]:
        """Get item from cache."""
        return self._data.get(key)

    def set(self, key: str, value: T) -> None:
        """Set item in cache."""
        self._data[key] = value

# Use Literal for constrained values
ProcessingMode = Literal["fast", "accurate", "balanced"]

def set_processing_mode(mode: ProcessingMode) -> None:
    """Set the processing mode."""
    pass

# Use overload for different signatures
@overload
def process_data(data: str) -> str: ...

@overload
def process_data(data: List[str]) -> List[str]: ...

def process_data(data: Union[str, List[str]]) -> Union[str, List[str]]:
    """Process data with different input types."""
    if isinstance(data, str):
        return _process_single(data)
    else:
        return [_process_single(item) for item in data]
```

### Documentation

```python
def complex_processing_function(
    input_data: List[Dict[str, Any]],
    config: ProcessingConfig,
    *,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[float], None]] = None
) -> ProcessingResult:
    """
    Process complex data with configurable parallelization.

    This function processes a list of data items according to the provided
    configuration. It supports parallel processing for improved performance
    on multi-core systems.

    Args:
        input_data: List of data items to process. Each item should be a
            dictionary with required keys 'id' and 'content'.
        config: Processing configuration specifying how to handle the data.
            Must include 'batch_size' and 'processing_mode' settings.
        parallel: Whether to use parallel processing. Defaults to True.
            Set to False for debugging or when processing order matters.
        max_workers: Maximum number of worker processes. If None, uses
            the number of CPU cores. Ignored if parallel=False.
        progress_callback: Optional callback function that receives progress
            updates as a float between 0.0 and 1.0.

    Returns:
        ProcessingResult containing:
        - success: Whether all items were processed successfully
        - items_processed: Number of items successfully processed
        - processing_time: Total processing time in seconds
        - error_message: Error description if success=False
        - detailed_stats: Dictionary with per-item processing statistics

    Raises:
        ValueError: If input_data is empty or contains invalid items
        ConfigurationError: If config is missing required settings
        ProcessingError: If processing fails for any item

    Example:
        Basic usage:
        >>> data = [{"id": "1", "content": "text1"}, {"id": "2", "content": "text2"}]
        >>> config = ProcessingConfig(batch_size=10, processing_mode="fast")
        >>> result = complex_processing_function(data, config)
        >>> print(f"Processed {result.items_processed} items in {result.processing_time:.2f}s")

        With progress tracking:
        >>> def progress_handler(progress: float):
        ...     print(f"Progress: {progress:.1%}")
        >>> result = complex_processing_function(
        ...     data, config, progress_callback=progress_handler
        ... )

        Sequential processing for debugging:
        >>> result = complex_processing_function(data, config, parallel=False)

    Note:
        When using parallel processing, ensure that your data items are
        independent and don't require shared state. The function uses
        multiprocessing, so large data items may impact performance due
        to serialization overhead.

    See Also:
        - ProcessingConfig: Configuration options
        - ProcessingResult: Return value structure
        - simple_processing_function: For basic use cases
    """
    # Implementation here
    pass
```

### Import Organization

```python
# Standard library imports (alphabetical)
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports (alphabetical)
import click
import numpy as np
import torch
from transformers import AutoTokenizer

# Local imports (alphabetical)
from llmbuilder.config import Config, ModelConfig
from llmbuilder.data import DataLoader, TextProcessor
from llmbuilder.utils import setup_logging

# Relative imports (if needed, but prefer absolute)
from .base import BaseProcessor
from .utils import validate_input
```

## Testing Style

### Test Structure

```python
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from llmbuilder.data.processor import DocumentProcessor
from llmbuilder.config import ProcessingConfig

class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = ProcessingConfig(batch_size=10)
        self.processor = DocumentProcessor(self.config)

    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_process_valid_document_returns_success(self):
        """Test that processing a valid document returns success."""
        # Arrange
        document_content = "This is a test document with valid content."
        document_file = self.temp_dir / "test.txt"
        document_file.write_text(document_content)

        # Act
        result = self.processor.process_file(str(document_file))

        # Assert
        assert result.success is True
        assert result.items_processed == 1
        assert result.processing_time > 0
        assert result.error_message is None

    def test_process_nonexistent_file_raises_file_not_found(self):
        """Test that processing non-existent file raises FileNotFoundError."""
        # Arrange
        nonexistent_file = str(self.temp_dir / "nonexistent.txt")

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="File not found"):
            self.processor.process_file(nonexistent_file)

    @pytest.mark.parametrize("content,expected_items", [
        ("Single line", 1),
        ("Line 1\nLine 2", 2),
        ("", 0),
        ("Line 1\n\nLine 3", 2),  # Empty lines ignored
    ])
    def test_process_file_counts_items_correctly(self, content, expected_items):
        """Test that file processing counts items correctly."""
        # Arrange
        test_file = self.temp_dir / "test.txt"
        test_file.write_text(content)

        # Act
        result = self.processor.process_file(str(test_file))

        # Assert
        assert result.items_processed == expected_items

    def test_process_file_with_mock_dependencies(self):
        """Test file processing with mocked dependencies."""
        # Arrange
        test_file = self.temp_dir / "test.txt"
        test_file.write_text("test content")

        with patch.object(self.processor, '_validate_content') as mock_validate:
            mock_validate.return_value = True

            # Act
            result = self.processor.process_file(str(test_file))

            # Assert
            assert result.success is True
            mock_validate.assert_called_once_with("test content")

    @pytest.fixture
    def sample_documents(self):
        """Provide sample documents for testing."""
        documents = []
        for i in range(3):
            doc_file = self.temp_dir / f"doc_{i}.txt"
            doc_file.write_text(f"Document {i} content")
            documents.append(str(doc_file))
        return documents

    def test_batch_processing_with_fixture(self, sample_documents):
        """Test batch processing using fixture."""
        # Act
        results = self.processor.process_batch(sample_documents)

        # Assert
        assert len(results) == 3
        assert all(result.success for result in results)
```

### Test Naming

```python
# Good test names - describe what is being tested
def test_process_valid_html_file_extracts_text_content():
    """Test that HTML processor extracts text from valid HTML."""
    pass

def test_process_empty_file_returns_empty_result():
    """Test that processing empty file returns appropriate result."""
    pass

def test_process_file_with_invalid_encoding_raises_encoding_error():
    """Test that invalid encoding raises appropriate error."""
    pass

# Bad test names - not descriptive
def test_process():  # What does it process? What's expected?
    pass

def test_html():  # What about HTML?
    pass

def test_error():  # What error? When?
    pass
```

## Configuration and Data Classes

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ProcessingConfig:
    """Configuration for document processing operations."""

    # Required fields first
    batch_size: int
    output_format: str

    # Optional fields with defaults
    max_workers: int = 4
    timeout_seconds: int = 300
    enable_validation: bool = True

    # Complex defaults using field()
    supported_formats: List[str] = field(
        default_factory=lambda: ["txt", "html", "pdf"]
    )
    processing_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.output_format not in ["txt", "json", "xml"]:
            raise ValueError(f"Unsupported output format: {self.output_format}")

        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingConfig':
        """Create configuration from dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
```

## CLI Style

```python
import click
from typing import Optional

@click.group()
@click.version_option()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def main(ctx: click.Context, verbose: bool, config: Optional[str]):
    """
    LLMBuilder - A comprehensive toolkit for language model development.

    Use 'llmbuilder COMMAND --help' for more information on specific commands.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config

    if verbose:
        click.echo("Verbose mode enabled")

@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--format', 'output_format',
              type=click.Choice(['txt', 'json', 'xml']),
              default='txt',
              help='Output format')
@click.option('--batch-size', default=100,
              help='Processing batch size')
@click.option('--workers', default=4,
              help='Number of worker processes')
@click.option('--dry-run', is_flag=True,
              help='Show what would be done without executing')
@click.pass_context
def process(ctx: click.Context, input_file: str, output_file: str,
           output_format: str, batch_size: int, workers: int, dry_run: bool):
    """Process documents with specified options."""

    if ctx.obj['verbose']:
        click.echo(f"Processing {input_file} -> {output_file}")
        click.echo(f"Format: {output_format}, Batch: {batch_size}, Workers: {workers}")

    if dry_run:
        click.echo("DRY RUN: Would process file with these settings")
        return

    try:
        # Processing logic here
        with click.progressbar(length=100, label='Processing') as bar:
            # Simulate progress
            for i in range(100):
                # Do work
                bar.update(1)

        click.echo(f"✅ Successfully processed {input_file}")

    except Exception as e:
        click.echo(f"❌ Processing failed: {e}", err=True)
        ctx.exit(1)
```

## Performance Considerations

```python
# Use generators for memory efficiency
def process_large_file(file_path: str) -> Iterator[ProcessedItem]:
    """Process large file without loading everything into memory."""
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():  # Skip empty lines
                yield ProcessedItem(
                    line_number=line_num,
                    content=line.strip(),
                    processed_at=datetime.now()
                )

# Use appropriate data structures
from collections import defaultdict, deque
from typing import DefaultDict

# For counting/grouping
item_counts: DefaultDict[str, int] = defaultdict(int)

# For FIFO operations
processing_queue: deque[ProcessingTask] = deque()

# Use context managers for resources
@contextmanager
def managed_processor(config: ProcessingConfig):
    """Context manager for processor lifecycle."""
    processor = None
    try:
        processor = create_processor(config)
        processor.initialize()
        yield processor
    finally:
        if processor:
            processor.cleanup()
```

## Pre-commit Configuration

The project uses pre-commit hooks to enforce code quality:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        args: [--line-length=88]

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile=black]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]
```

## IDE Configuration

### VS Code Settings

```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.testing.pytestEnabled": true
}
```

## Summary

Following these style guidelines ensures:

- **Consistency** across the codebase
- **Readability** for all contributors
- **Maintainability** over time
- **Quality** through automated checks

The automated tools handle most formatting concerns, allowing you to focus on writing great code. When in doubt, run the pre-commit hooks to catch any style issues before committing.

---

For questions about code style, check existing code for examples or ask in GitHub Discussions.
