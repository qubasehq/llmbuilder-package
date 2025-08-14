# Model Card Template

A minimal template for documenting models trained with LLMBuilder.

---

# Model Card: [Model Name]

## Basic Info

- **Model**: [Name] v[Version]
- **Size**: [e.g., 125M parameters]
- **Type**: [e.g., GPT-style Transformer]
- **License**: [e.g., MIT]
- **Date**: [Training date]

## Architecture

- **Layers**: [e.g., 12]
- **Hidden Size**: [e.g., 768]
- **Vocab Size**: [e.g., 32K tokens]
- **Max Length**: [e.g., 2048 tokens]

## Training Data

- **Dataset**: [Name and size]
- **Languages**: [e.g., English]
- **Domains**: [e.g., Web text, books]
- **Processing**: [Brief description of cleaning/dedup]

## Performance

| Metric | Score |
|--------|-------|
| Perplexity | [e.g., 15.2] |
| [Other metrics] | [Values] |

## Usage

```python
import llmbuilder as lb

model = lb.load_model("path/to/model.pt")
tokenizer = lb.load_tokenizer("path/to/tokenizer/")

text = lb.generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt="Your prompt here",
    max_new_tokens=100
)
```

## Limitations

- [Key technical limitations]
- [Known biases or risks]
- [Recommended safety measures]

## Files

- `model.pt` - Model weights
- `config.json` - Configuration
- `tokenizer/` - Tokenizer files
- `model_q4_0.gguf` - Quantized version

## Citation

```bibtex
@misc{[model_name],
  title={[Model Name]},
  author={[Your Name]},
  year={[Year]},
  note={Trained using LLMBuilder}
}
```
