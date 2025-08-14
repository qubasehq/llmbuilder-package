# LLMBuilder CLI Guide

This document explains how to use the LLMBuilder command-line interface. It lists all available commands, their inputs/outputs, and Windows-friendly examples.

- Executable: `llmbuilder`
- Show help: `llmbuilder --help`
- Version: `llmbuilder --version`

Tip for Windows: wrap prompts in double quotes, and use full paths when in doubt.

---

## Global

- `llmbuilder --help` — Show top-level help.
- `llmbuilder --version` — Show package version.
- `llmbuilder -v` — Enable verbose mode for top-level command (shows extra output in some subcommands).
- `llmbuilder welcome` — Friendly welcome and quick actions.
- `llmbuilder info` — Display package information and module overview.

Examples:

- `llmbuilder info`
- `llmbuilder welcome`

---

## Data Commands

Group: `llmbuilder data`

### 1) Load and preprocess text

`llmbuilder data load` — Load text from files/directories, optionally clean, and save to a single file.

Inputs:

- `--input, -i` (file or directory)
- `--output, -o` (output .txt file)
- `--format` [txt|pdf|docx|all] (default: all)
- `--clean` (flag to clean text)
- `--min-length` (minimum text length to keep; default: 50)
- `--interactive` (guided mode)

Outputs:

- A single text file with processed content; progress messages.

Examples:

- Directory to single file:
  `llmbuilder data load -i D:\data -o D:\out\combined.txt --clean --min-length 80`
- Single file:
  `llmbuilder data load -i D:\docs\paper.pdf -o D:\out\paper.txt`
- Interactive:
  `llmbuilder data load --interactive`

### 2) Ingest multi-format documents

`llmbuilder data ingest` — Batch-process multiple formats with optional OCR fallback.

Inputs:

- `--input, -i` (file or directory)
- `--output, -o` (output directory)
- `--formats` (multiple) [html|markdown|epub|pdf|all] (default: all)
- `--batch-size` (default: 100)
- `--workers` (default: 4)
- `--ocr-fallback` (flag for PDFs)
- `--verbose, -v`

Outputs:

- Processed files written into the output directory and a summary (processed/succeeded/failed).

Example:

- `llmbuilder data ingest -i D:\raw_docs -o D:\processed --formats html --formats pdf --ocr-fallback -v`

### 3) Deduplicate text

`llmbuilder data deduplicate` — Remove exact/semantic duplicates from text.

Inputs:

- `--input, -i` (file or directory containing .txt files)
- `--output, -o` (output file for deduplicated lines)
- `--method` [exact|semantic|both] (default: both)
- `--similarity-threshold` (0.0–1.0, default: 0.85)
- `--batch-size` (default: 1000)
- `--embedding-model` (default: all-MiniLM-L6-v2)
- `--verbose, -v`

Outputs:

- Deduplicated text file, with stats printed.

Example:

- `llmbuilder data deduplicate -i D:\text_corpus -o D:\out\dedup.txt --method both --similarity-threshold 0.9 -v`

---

## Tokenizer Commands

Group: `llmbuilder tokenizer`

### 1) Train tokenizer

`llmbuilder tokenizer train`

Inputs:

- `--input, -i` (text file or directory)
- `--output, -o` (output directory)
- `--vocab-size` (default: 16000)
- `--algorithm` [bpe|unigram|wordpiece|sentencepiece] (default: bpe)
- `--special-tokens` (multiple) e.g. `<pad>` `<unk>`
- `--min-frequency` (default: 2)
- `--coverage` (SentencePiece only, default: 0.9995)
- `--validate` (flag)
- `--verbose, -v`

Outputs:

- Tokenizer model and vocab files saved under the output directory with training stats.

Examples:

- BPE:
  `llmbuilder tokenizer train -i D:\out\combined.txt -o D:\out\tokenizer --vocab-size 8000 --algorithm bpe --validate`
- SentencePiece:
  `llmbuilder tokenizer train -i D:\out\combined.txt -o D:\out\sp_tokenizer --algorithm sentencepiece --coverage 0.999`

### 2) Test tokenizer

`llmbuilder tokenizer test`

Inputs:

- `tokenizer_path` (positional; directory containing tokenizer)
- One of:
  - `--text, -t` (quick test string)
  - `--file, -f` (file to tokenize)
  - `--interactive, -i` (interactive loop)

Outputs:

- Encoded tokens, decoded text, and token counts.

Examples:

- Single text:
  `llmbuilder tokenizer test D:\out\tokenizer -t "Hello world"`
- Interactive:
  `llmbuilder tokenizer test D:\out\tokenizer -i`

---

## Training Commands

Group: `llmbuilder train`

### 1) Train a model from scratch

`llmbuilder train model`

Inputs:

- `--data, -d` (path to tokenized dataset or compatible data file)
- `--tokenizer, -t` (tokenizer directory)
- `--output, -o` (checkpoint output directory)
- `--epochs` (default: 10)
- `--batch-size` (default: 32)
- `--lr` (default: 3e-4)
- `--vocab-size` (default: 16000)
- `--layers` (default: 8)
- `--heads` (default: 8)
- `--dim` (embedding dim; default: 512)
- `--config, -c` (optional config file, if supported)

Outputs:

- Training progress, checkpoints in output directory.

Example:

- `llmbuilder train model -d D:\LLM\Model_Test\output\tokenized_data.pt -t D:\LLM\Model_Test\output\tokenizer -o D:\LLM\Model_Test\output\checkpoints --epochs 5 --batch-size 1 --lr 6e-4 --vocab-size 8000 --layers 4 --heads 8 --dim 256`

### 2) Resume training

`llmbuilder train resume`

Inputs:

- `--checkpoint, -c` (path to existing checkpoint)
- `--data, -d` (dataset path)
- `--output, -o` (optional; defaults to checkpoint directory)

Outputs:

- Training resumed; new checkpoints written.

Example:

- `llmbuilder train resume -c D:\LLM\Model_Test\output\checkpoints\checkpoint_epoch_2.pt -d D:\LLM\Model_Test\output\tokenized_data.pt`

---

## Fine-tuning Commands

Group: `llmbuilder finetune`

### Fine-tune a pre-trained model

`llmbuilder finetune model`

Inputs:

- `--model, -m` (path to pre-trained model)
- `--dataset, -d` (dataset path)
- `--output, -o` (output directory)
- `--epochs` (default: 3)
- `--lr` (default: 5e-5)
- `--batch-size` (default: 4)
- `--use-lora` (flag)
- `--lora-rank` (default: 4)

Outputs:

- Fine-tuned model checkpoints and summary (best validation loss etc.).

Example:

- `llmbuilder finetune model -m D:\models\base.pt -d D:\data\fine_tune.pt -o D:\out\ft --epochs 3 --use-lora`

---

## Generation Commands

Group: `llmbuilder generate`

### Generate text

`llmbuilder generate text`

Inputs:

- `--model, -m` (checkpoint path). Use `latest_checkpoint.pt` or a specific `checkpoint_epoch_X.pt`.
- `--tokenizer, -t` (tokenizer directory)
- `--prompt, -p` (text prompt)
- `--interactive, -i` (interactive chat-like mode)
- `--max-tokens` (default: 100)
- `--temperature` (default: 0.8)
- `--top-k` (default: 50)
- `--top-p` (default: 0.9)
- `--device` (cpu|cuda; optional)
- `--setup` (guided setup)

Outputs:

- Generated text printed to console; in interactive mode, a live session.

Examples:

- One-shot:
  `llmbuilder generate text -m D:\LLM\Model_Test\output\checkpoints\latest_checkpoint.pt -t D:\LLM\Model_Test\output\tokenizer -p "Cybersecurity is important because" --max-tokens 120 --temperature 0.8 --top-k 50 --top-p 0.9`
- Interactive:
  `llmbuilder generate text -m D:\LLM\Model_Test\output\checkpoints\latest_checkpoint.pt -t D:\LLM\Model_Test\output\tokenizer --interactive`
- Guided setup:
  `llmbuilder generate text --setup`

Notes:

- If `best_checkpoint.pt` is missing, use `latest_checkpoint.pt`.
- Ensure your tokenizer matches the model used for training.

---

## Model Management Commands

Group: `llmbuilder model`

### 1) Create a new empty model

`llmbuilder model create`

Inputs:

- `--output, -o` (path to save)
- `--vocab-size` (default: 16000)
- `--layers` (default: 8)
- `--heads` (default: 8)
- `--dim` (default: 512)
- `--config, -c` (optional)

Outputs:

- Saved model file and parameter counts printed.

Example:

- `llmbuilder model create -o D:\models\new.pt --vocab-size 8000 --layers 4 --heads 8 --dim 256`

### 2) Show model info

`llmbuilder model info <model_path>`

Outputs:

- Total/trainable parameters and architecture details (if available).

Example:

- `llmbuilder model info D:\LLM\Model_Test\output\checkpoints\latest_checkpoint.pt`

### 3) Evaluate a model

`llmbuilder model evaluate`

Inputs:

- `--dataset, -d` (evaluation dataset path)
- `--batch-size` (default: 32)

Outputs:

- Evaluation metrics (loss, perplexity if available).

Example:

- `llmbuilder model evaluate D:\LLM\Model_Test\output\checkpoints\latest_checkpoint.pt -d D:\LLM\Model_Test\output\tokenized_data.pt`

---

## Export Commands

Group: `llmbuilder export`

### Export to GGUF (llama.cpp compatibility)

`llmbuilder export gguf`

Inputs:

- `model_path` (positional; path to source model/checkpoint)
- `--output, -o` (output GGUF file path)
- `--quantization` [Q8_0|Q4_0|Q4_1|Q5_0|Q5_1|F16|F32] (default: Q8_0)
- `--validate` (flag; run post-conversion validation if available)
- `--verbose, -v` (flag; print extra diagnostics and script discovery)

Outputs:

- A `.gguf` model file written to the output path. Summary includes size, time, and quantization details. On `--validate`, prints validation result.

Examples:

- Convert with default Q8_0:
  `llmbuilder export gguf D:\LLM\Model_Test\output\checkpoints\latest_checkpoint.pt -o D:\LLM\Model_Test\output\models\latest_q8.gguf`
- Convert with Q4_0 and validate:
  `llmbuilder export gguf D:\models\my_model.pt -o D:\models\my_model_q4.gguf --quantization Q4_0 --validate -v`

Notes:

- Ensure sufficient disk space for the converted file.
- Quantization level affects size and speed/quality tradeoffs.

---

## Tips & Troubleshooting

- **No best checkpoint**: If validation is disabled or failed to split, only `latest_checkpoint.pt` is produced. Use that for generation. After upgrading to a version with robust splits and retraining, `best_checkpoint.pt` will appear.
- **Colors on Windows**: Colors are enabled automatically via colorama. If you don’t see colors, ensure `llmbuilder>=0.4.5` is installed.
- **Progress bars**: Data loading, training, and validation show tqdm bars when console output is enabled.
- **Paths**: Prefer absolute paths on Windows to avoid confusion.
- **Help**: Every command supports `--help`.

---

## Quick Start Workflow

1) Prepare data

```
llmbuilder data load -i D:\data -o D:\LLM\Model_Test\output\processed_data.txt --clean
```

2) Train tokenizer

```
llmbuilder tokenizer train -i D:\LLM\Model_Test\output\processed_data.txt -o D:\LLM\Model_Test\output\tokenizer --vocab-size 8000 --algorithm bpe --validate
```

3) Train model

```
llmbuilder train model -d D:\LLM\Model_Test\output\tokenized_data.pt -t D:\LLM\Model_Test\output\tokenizer -o D:\LLM\Model_Test\output\checkpoints --epochs 5 --batch-size 1 --lr 6e-4 --vocab-size 8000 --layers 4 --heads 8 --dim 256
```

4) Generate text

```
llmbuilder generate text -m D:\LLM\Model_Test\output\checkpoints\latest_checkpoint.pt -t D:\LLM\Model_Test\output\tokenizer -p "Cybersecurity is important because" --max-tokens 120
```

---

## Cross‑platform Quick Commands

Below are ready-to-run snippets for Windows PowerShell and Linux/macOS Bash/Zsh.

### Data: Load

PowerShell

```powershell
llmbuilder data load -i "D:\data" -o "D:\LLM\Model_Test\output\processed_data.txt" --clean --min-length 80
```

Bash/Zsh

```bash
llmbuilder data load -i "/data" -o "/mnt/llm/Model_Test/output/processed_data.txt" --clean --min-length 80
```

### Data: Ingest

PowerShell

```powershell
llmbuilder data ingest -i "D:\raw_docs" -o "D:\processed" --formats html --formats pdf --ocr-fallback -v
```

Bash/Zsh

```bash
llmbuilder data ingest -i "/data/raw_docs" -o "/data/processed" --formats html --formats pdf --ocr-fallback -v
```

### Data: Deduplicate

PowerShell

```powershell
llmbuilder data deduplicate -i "D:\text_corpus" -o "D:\out\dedup.txt" --method both --similarity-threshold 0.9 -v
```

Bash/Zsh

```bash
llmbuilder data deduplicate -i "/data/text_corpus" -o "/data/out/dedup.txt" --method both --similarity-threshold 0.9 -v
```

### Tokenizer: Train (BPE)

PowerShell

```powershell
llmbuilder tokenizer train -i "D:\LLM\Model_Test\output\processed_data.txt" -o "D:\LLM\Model_Test\output\tokenizer" --vocab-size 8000 --algorithm bpe --validate
```

Bash/Zsh

```bash
llmbuilder tokenizer train -i "/mnt/llm/Model_Test/output/processed_data.txt" -o "/mnt/llm/Model_Test/output/tokenizer" --vocab-size 8000 --algorithm bpe --validate
```

### Tokenizer: Test (interactive)

PowerShell

```powershell
llmbuilder tokenizer test "D:\LLM\Model_Test\output\tokenizer" -i
```

Bash/Zsh

```bash
llmbuilder tokenizer test "/mnt/llm/Model_Test/output/tokenizer" -i
```

### Train: From scratch

PowerShell

```powershell
llmbuilder train model -d "D:\LLM\Model_Test\output\tokenized_data.pt" -t "D:\LLM\Model_Test\output\tokenizer" -o "D:\LLM\Model_Test\output\checkpoints" --epochs 5 --batch-size 1 --lr 6e-4 --vocab-size 8000 --layers 4 --heads 8 --dim 256
```

Bash/Zsh

```bash
llmbuilder train model -d "/mnt/llm/Model_Test/output/tokenized_data.pt" -t "/mnt/llm/Model_Test/output/tokenizer" -o "/mnt/llm/Model_Test/output/checkpoints" --epochs 5 --batch-size 1 --lr 6e-4 --vocab-size 8000 --layers 4 --heads 8 --dim 256
```

### Train: Resume

PowerShell

```powershell
llmbuilder train resume -c "D:\LLM\Model_Test\output\checkpoints\checkpoint_epoch_2.pt" -d "D:\LLM\Model_Test\output\tokenized_data.pt"
```

Bash/Zsh

```bash
llmbuilder train resume -c "/mnt/llm/Model_Test/output/checkpoints/checkpoint_epoch_2.pt" -d "/mnt/llm/Model_Test/output/tokenized_data.pt"
```

### Generate: One-shot

PowerShell

```powershell
llmbuilder generate text -m "D:\LLM\Model_Test\output\checkpoints\latest_checkpoint.pt" -t "D:\LLM\Model_Test\output\tokenizer" -p "Cybersecurity is important because" --max-tokens 120 --temperature 0.8 --top-k 50 --top-p 0.9
```

Bash/Zsh

```bash
llmbuilder generate text -m "/mnt/llm/Model_Test/output/checkpoints/latest_checkpoint.pt" -t "/mnt/llm/Model_Test/output/tokenizer" -p "Cybersecurity is important because" --max-tokens 120 --temperature 0.8 --top-k 50 --top-p 0.9
```

### Generate: Interactive

PowerShell

```powershell
llmbuilder generate text -m "D:\LLM\Model_Test\output\checkpoints\latest_checkpoint.pt" -t "D:\LLM\Model_Test\output\tokenizer" --interactive
```

Bash/Zsh

```bash
llmbuilder generate text -m "/mnt/llm/Model_Test/output/checkpoints/latest_checkpoint.pt" -t "/mnt/llm/Model_Test/output/tokenizer" --interactive
```

### Model: Create

PowerShell

```powershell
llmbuilder model create -o "D:\models\new.pt" --vocab-size 8000 --layers 4 --heads 8 --dim 256
```

Bash/Zsh

```bash
llmbuilder model create -o "/models/new.pt" --vocab-size 8000 --layers 4 --heads 8 --dim 256
```

### Model: Info

PowerShell

```powershell
llmbuilder model info "D:\LLM\Model_Test\output\checkpoints\latest_checkpoint.pt"
```

Bash/Zsh

```bash
llmbuilder model info "/mnt/llm/Model_Test/output/checkpoints/latest_checkpoint.pt"
```

### Model: Evaluate

PowerShell

```powershell
llmbuilder model evaluate "D:\LLM\Model_Test\output\checkpoints\latest_checkpoint.pt" -d "D:\LLM\Model_Test\output\tokenized_data.pt" --batch-size 8
```

Bash/Zsh

```bash
llmbuilder model evaluate "/mnt/llm/Model_Test/output/checkpoints/latest_checkpoint.pt" -d "/mnt/llm/Model_Test/output/tokenized_data.pt" --batch-size 8
```

### Export: GGUF (Q8_0)

PowerShell

```powershell
llmbuilder export gguf "D:\LLM\Model_Test\output\checkpoints\latest_checkpoint.pt" -o "D:\LLM\Model_Test\output\models\latest_q8.gguf"
```

Bash/Zsh

```bash
llmbuilder export gguf "/mnt/llm/Model_Test/output/checkpoints/latest_checkpoint.pt" -o "/mnt/llm/Model_Test/output/models/latest_q8.gguf"
```

### Export: GGUF (Q4_0, validate)

PowerShell

```powershell
llmbuilder export gguf "D:\models\my_model.pt" -o "D:\models\my_model_q4.gguf" --quantization Q4_0 --validate -v
```

Bash/Zsh

```bash
llmbuilder export gguf "/models/my_model.pt" -o "/models/my_model_q4.gguf" --quantization Q4_0 --validate -v
```

---

## Screenshots

Add images under `docs/images/` and reference them here or in the README.

- Welcome screen: `![Welcome](docs/images/cli-welcome.png)`
- Data load progress: `![Data Load](docs/images/cli-data-load.png)`
- Training progress bars: `![Training](docs/images/cli-training.png)`
- Validation loop: `![Validation](docs/images/cli-validation.png)`
- Generate (interactive): `![Generate Interactive](docs/images/cli-generate-interactive.png)`
- Export GGUF: `![Export GGUF](docs/images/cli-export-gguf.png)`

Tip (Windows): Alt+PrintScreen captures the active window; save PNG to `docs/images/`.

---

Maintained by Qub△se. For more, see the repository wiki and `llmbuilder info`.
