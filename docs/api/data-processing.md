# Data Processing API Reference

## Core Classes

### IngestionPipeline

```python
from llmbuilder.data.ingest import IngestionPipeline

pipeline = IngestionPipeline(config)
pipeline.process_directory("data/raw/")
```

### DeduplicationPipeline

```python
from llmbuilder.data.dedup import DeduplicationPipeline

dedup = DeduplicationPipeline(config)
dedup.process_files(["file1.txt", "file2.txt"])
```

### TokenizerTrainer

```python
from llmbuilder.training.tokenizer import TokenizerTrainer

trainer = TokenizerTrainer(config)
trainer.train(corpus_files, output_dir)
```

### GGUFConverter

```python
from llmbuilder.tools.convert_to_gguf import GGUFConverter

converter = GGUFConverter()
converter.convert(model_path, output_path)
```

## Configuration

All components use the unified config system:

```python
from llmbuilder.config import ConfigManager

config = ConfigManager.load_config("config.json")
```
