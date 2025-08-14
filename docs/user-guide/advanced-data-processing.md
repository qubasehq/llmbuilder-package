# Advanced Data Processing

LLMBuilder provides sophisticated data processing capabilities designed to handle diverse document formats, perform intelligent deduplication, and prepare high-quality training datasets. This guide covers the advanced features that go beyond basic text processing.

## Overview

The advanced data processing system includes:

- **Multi-format Document Ingestion**: Process HTML, Markdown, EPUB, PDF, and plain text files
- **Intelligent Text Extraction**: OCR fallback for scanned PDFs, clean HTML parsing, structured document handling
- **Advanced Deduplication**: Both exact and semantic duplicate detection with configurable thresholds
- **Flexible Tokenizer Training**: Support for multiple algorithms including BPE, SentencePiece, and Unigram
- **GGUF Model Conversion**: Convert trained models to GGUF format with various quantization levels
- **Configuration Management**: Template-based configuration system with validation

## Multi-Format Document Ingestion

### Supported Formats

LLMBuilder can process the following document formats:

| Format | Extension | Features |
|--------|-----------|----------|
| HTML | `.html`, `.htm` | Clean text extraction, script/style removal, configurable parsers |
| Markdown | `.md`, `.markdown` | Syntax removal, structure preservation, metadata extraction |
| EPUB | `.epub` | Chapter organization, table of contents, metadata handling |
| PDF | `.pdf` | Text extraction with OCR fallback, quality assessment |
| Plain Text | `.txt` | Direct processing with encoding detection |

### Basic Usage

```bash
# Process a single file
llmbuilder data load --input document.pdf --output processed.txt

# Process a directory of mixed formats
llmbuilder data load --input documents/ --output dataset.txt --format all

# Enable OCR for scanned PDFs
llmbuilder data load --input scanned.pdf --output text.txt --enable-ocr
```

### Configuration Options

The ingestion system can be configured through the configuration file:

```json
{
  "data": {
    "ingestion": {
      "supported_formats": ["html", "markdown", "epub", "pdf", "txt"],
      "batch_size": 100,
      "num_workers": 4,
      "output_format": "txt",
      "preserve_structure": false,
      "extract_metadata": true,

      "html_parser": "lxml",
      "remove_scripts": true,
      "remove_styles": true,

      "enable_ocr": true,
      "ocr_quality_threshold": 0.5,
      "ocr_language": "eng",
      "pdf_extraction_method": "auto",

      "extract_toc": true,
      "chapter_separator": "\n\n---\n\n"
    }
  }
}
```

### HTML Processing

The HTML processor provides clean text extraction with configurable options:

```python
from llmbuilder.data.ingest import HTMLProcessor

processor = HTMLProcessor(
    parser='lxml',  # html.parser, lxml, html5lib
    remove_scripts=True,
    remove_styles=True
)

text = processor.process_file('webpage.html')
```

**Features:**

- Multiple parser backends (html.parser, lxml, html5lib)
- Script and style tag removal
- Encoding detection and handling
- Malformed HTML recovery

### PDF Processing with OCR

The PDF processor includes intelligent OCR fallback:

```python
from llmbuilder.data.ingest import PDFProcessor

processor = PDFProcessor(
    enable_ocr=True,
    ocr_quality_threshold=0.5,
    ocr_language='eng'
)

text = processor.process_file('document.pdf')
```

**Features:**

- Primary text extraction using PyMuPDF
- Quality assessment to trigger OCR
- OCR fallback using Tesseract
- Mixed content handling (text + scanned pages)

### EPUB Processing

Extract text from EPUB files with chapter organization:

```python
from llmbuilder.data.ingest import EPUBProcessor

processor = EPUBProcessor(
    extract_toc=True,
    chapter_separator='\n\n---\n\n'
)

text = processor.process_file('book.epub')
```

**Features:**

- Chapter-by-chapter extraction
- Table of contents handling
- Metadata preservation
- Multiple EPUB format support

## Advanced Deduplication

### Exact Deduplication

Remove exact duplicates using normalized hashing:

```bash
# Basic exact deduplication
llmbuilder data deduplicate --input dataset.txt --output clean_dataset.txt --method exact

# With custom normalization
llmbuilder data deduplicate --input dataset.txt --output clean_dataset.txt --method exact --normalize
```

### Semantic Deduplication

Detect near-duplicates using sentence embeddings:

```bash
# Semantic deduplication with default threshold
llmbuilder data deduplicate --input dataset.txt --output clean_dataset.txt --method semantic

# Custom similarity threshold
llmbuilder data deduplicate --input dataset.txt --output clean_dataset.txt --method semantic --threshold 0.9

# Use GPU for faster processing
llmbuilder data deduplicate --input dataset.txt --output clean_dataset.txt --method semantic --use-gpu
```

### Combined Deduplication

Use both exact and semantic deduplication:

```bash
llmbuilder data deduplicate --input dataset.txt --output clean_dataset.txt --method both --threshold 0.85
```

### Configuration

Configure deduplication behavior:

```json
{
  "data": {
    "deduplication": {
      "enable_exact_deduplication": true,
      "enable_semantic_deduplication": true,
      "similarity_threshold": 0.85,
      "embedding_model": "all-MiniLM-L6-v2",
      "batch_size": 1000,
      "chunk_size": 512,
      "min_text_length": 50,
      "normalize_text": true,
      "use_gpu_for_embeddings": true,
      "embedding_cache_size": 10000,
      "similarity_metric": "cosine"
    }
  }
}
```

## Flexible Tokenizer Training

### Supported Algorithms

LLMBuilder supports multiple tokenization algorithms:

| Algorithm | Library | Best For |
|-----------|---------|----------|
| BPE | Hugging Face Tokenizers | General purpose, subword tokenization |
| SentencePiece | Google SentencePiece | Language-agnostic, handles any text |
| Unigram | Hugging Face Tokenizers | Probabilistic subword segmentation |
| WordPiece | Hugging Face Tokenizers | BERT-style tokenization |

### Basic Training

```bash
# Train a BPE tokenizer
llmbuilder data tokenizer --input corpus.txt --output tokenizer/ --algorithm bpe --vocab-size 16000

# Train a SentencePiece tokenizer
llmbuilder data tokenizer --input corpus.txt --output tokenizer/ --algorithm sentencepiece --vocab-size 32000

# Train with custom special tokens
llmbuilder data tokenizer --input corpus.txt --output tokenizer/ --algorithm bpe --vocab-size 16000 --special-tokens "<pad>,<unk>,<s>,</s>,<mask>"
```

### Advanced Configuration

```json
{
  "tokenizer_training": {
    "vocab_size": 32000,
    "algorithm": "sentencepiece",
    "min_frequency": 2,
    "special_tokens": ["<pad>", "<unk>", "<s>", "</s>", "<mask>"],
    "character_coverage": 0.9995,
    "max_sentence_length": 4192,
    "shuffle_input_sentence": true,
    "normalization_rule_name": "nmt_nfkc_cf",
    "remove_extra_whitespaces": true,
    "add_dummy_prefix": true,
    "continuing_subword_prefix": "##",
    "end_of_word_suffix": "",
    "num_threads": 4,
    "max_training_time": 3600,
    "validation_split": 0.1
  }
}
```

### Validation and Testing

Validate trained tokenizers:

```bash
# Test tokenizer on sample text
llmbuilder tokenizer test --tokenizer tokenizer/ --text "Hello world, this is a test."

# Benchmark tokenizer performance
llmbuilder tokenizer benchmark --tokenizer tokenizer/ --test-file test_corpus.txt
```

## GGUF Model Conversion

### Basic Conversion

Convert trained models to GGUF format for inference:

```bash
# Basic conversion with Q8_0 quantization
llmbuilder convert gguf model/ --output model.gguf

# Specify quantization level
llmbuilder convert gguf model/ --output model.gguf --quantization Q4_0

# Enable validation
llmbuilder convert gguf model/ --output model.gguf --quantization Q4_0 --validate
```

### Batch Conversion

Convert multiple models with different quantization levels:

```bash
# Convert with all quantization levels
llmbuilder convert gguf model/ --output model.gguf --all-quantizations

# Batch convert multiple models
llmbuilder convert batch --input-dir models/ --output-dir gguf_models/ --quantization Q8_0 Q4_0 Q4_1
```

### Supported Quantization Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| F32 | Full precision | Maximum quality, large size |
| F16 | Half precision | Good quality, moderate size |
| Q8_0 | 8-bit quantization | Balanced quality/size |
| Q5_1 | 5-bit quantization | Good compression |
| Q5_0 | 5-bit quantization | Better compression |
| Q4_1 | 4-bit quantization | High compression |
| Q4_0 | 4-bit quantization | Maximum compression |

### Configuration

```json
{
  "gguf_conversion": {
    "quantization_level": "Q4_0",
    "validate_output": true,
    "conversion_timeout": 3600,
    "preferred_script": "auto",
    "output_naming": "quantization_suffix",
    "preserve_metadata": true
  }
}
```

## Configuration Management

### Using Templates

LLMBuilder provides pre-configured templates for common use cases:

```bash
# List available templates
llmbuilder config templates

# Create config from template
llmbuilder config from-template basic_config --output my_config.json

# Create with overrides
llmbuilder config from-template basic_config --output my_config.json \
  --override model.vocab_size=24000 \
  --override training.batch_size=64
```

### Available Templates

| Template | Description | Best For |
|----------|-------------|----------|
| `basic_config` | Balanced configuration | General use, moderate resources |
| `cpu_optimized_config` | CPU-optimized settings | CPU-only training |
| `advanced_processing_config` | Full feature set | High-end GPU, large datasets |
| `inference_optimized_config` | Inference settings | Production deployment |
| `large_scale_config` | Large model training | High-end hardware, large models |

### Configuration Validation

Validate configuration files:

```bash
# Basic validation
llmbuilder config validate my_config.json

# Detailed validation with summary
llmbuilder config validate my_config.json --detailed

# Show configuration summary
llmbuilder config summary my_config.json
```

## End-to-End Workflows

### Complete Data Processing Pipeline

```bash
# 1. Ingest multi-format documents
llmbuilder data load --input documents/ --output raw_text.txt --format all --clean

# 2. Deduplicate the dataset
llmbuilder data deduplicate --input raw_text.txt --output clean_text.txt --method both --threshold 0.85

# 3. Train a tokenizer
llmbuilder data tokenizer --input clean_text.txt --output tokenizer/ --algorithm sentencepiece --vocab-size 32000

# 4. Train the model (using existing training commands)
llmbuilder train model --data clean_text.txt --tokenizer tokenizer/ --output model/ --config advanced_config.json

# 5. Convert to GGUF format
llmbuilder convert gguf model/ --output model_q4.gguf --quantization Q4_0 --validate
```

### Configuration-Driven Workflow

Create a comprehensive configuration file:

```json
{
  "data": {
    "ingestion": {
      "supported_formats": ["html", "markdown", "epub", "pdf"],
      "enable_ocr": true,
      "batch_size": 200
    },
    "deduplication": {
      "enable_exact_deduplication": true,
      "enable_semantic_deduplication": true,
      "similarity_threshold": 0.9
    }
  },
  "tokenizer_training": {
    "algorithm": "sentencepiece",
    "vocab_size": 32000
  },
  "gguf_conversion": {
    "quantization_level": "Q4_0",
    "validate_output": true
  }
}
```

Then use it throughout the pipeline:

```bash
# All commands will use the same configuration
llmbuilder data load --config my_config.json --input docs/ --output dataset.txt
llmbuilder data deduplicate --config my_config.json --input dataset.txt --output clean.txt
llmbuilder data tokenizer --config my_config.json --input clean.txt --output tokenizer/
```

## Performance Optimization

### Memory Management

For large datasets, optimize memory usage:

```json
{
  "data": {
    "ingestion": {
      "batch_size": 50,
      "num_workers": 2
    },
    "deduplication": {
      "batch_size": 500,
      "embedding_cache_size": 5000,
      "use_gpu_for_embeddings": false
    }
  }
}
```

### GPU Acceleration

Enable GPU acceleration where available:

```json
{
  "data": {
    "deduplication": {
      "use_gpu_for_embeddings": true,
      "batch_size": 2000
    }
  },
  "system": {
    "device": "cuda",
    "mixed_precision": true
  }
}
```

### Parallel Processing

Optimize for multi-core systems:

```json
{
  "data": {
    "ingestion": {
      "num_workers": 8
    }
  },
  "tokenizer_training": {
    "num_threads": 8
  },
  "system": {
    "num_workers": 8
  }
}
```

## Troubleshooting

### Common Issues

**OCR Dependencies Missing**

```bash
# Install Tesseract
sudo apt-get install tesseract-ocr  # Ubuntu/Debian
brew install tesseract              # macOS
```

**Memory Issues with Large Files**

- Reduce batch sizes in configuration
- Use streaming processing for very large files
- Enable disk-based caching for embeddings

**Slow Semantic Deduplication**

- Enable GPU acceleration
- Reduce embedding model size
- Increase batch size
- Use embedding caching

**GGUF Conversion Failures**

- Ensure llama.cpp is installed and accessible
- Check model format compatibility
- Verify sufficient disk space
- Use longer conversion timeout

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Enable debug logging
export LLMBUILDER_LOG_LEVEL=DEBUG

# Run with verbose output
llmbuilder data load --input docs/ --output dataset.txt --verbose
```

## Best Practices

### Data Quality

1. **Always validate input data** before processing
2. **Use appropriate OCR thresholds** for PDF processing
3. **Configure deduplication carefully** to avoid over-removal
4. **Monitor processing statistics** for quality assessment

### Performance

1. **Use GPU acceleration** for semantic deduplication
2. **Optimize batch sizes** for your hardware
3. **Enable parallel processing** for multi-core systems
4. **Use appropriate quantization levels** for your use case

### Configuration Management

1. **Use templates** as starting points
2. **Validate configurations** before use
3. **Version control** your configuration files
4. **Document custom settings** and their rationale

### Workflow Organization

1. **Process data incrementally** for large datasets
2. **Keep intermediate results** for debugging
3. **Use consistent naming conventions** for outputs
4. **Monitor resource usage** during processing

This comprehensive guide covers all aspects of LLMBuilder's advanced data processing capabilities. For specific API documentation, see the [API Reference](../api/core.md).
