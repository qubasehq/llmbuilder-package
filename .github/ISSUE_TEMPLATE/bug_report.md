---
name: Bug Report
about: Create a report to help us improve LLMBuilder
title: '[BUG] '
labels: bug
assignees: ''
---

## 🐛 Bug Description

A clear and concise description of what the bug is.

## 🔄 Steps to Reproduce

Steps to reproduce the behavior:

1. Install LLMBuilder with: `pip install llmbuilder[...]`
2. Run command: `llmbuilder ...`
3. Use configuration: `...`
4. See error

## ✅ Expected Behavior

A clear and concise description of what you expected to happen.

## ❌ Actual Behavior

A clear and concise description of what actually happened.

## 📋 Environment Information

**System Information:**

- OS: [e.g., Ubuntu 20.04, Windows 10, macOS 12.0]
- Python version: [e.g., 3.9.7]
- LLMBuilder version: [e.g., 1.2.3]

**Installation Method:**

- [ ] pip install llmbuilder
- [ ] pip install llmbuilder[all]
- [ ] Development installation (pip install -e .)
- [ ] Other: ___________

**Optional Dependencies Installed:**

- [ ] PDF processing (pymupdf, pytesseract)
- [ ] EPUB processing (ebooklib)
- [ ] HTML processing (beautifulsoup4, lxml)
- [ ] Semantic deduplication (sentence-transformers)
- [ ] GGUF conversion (llama.cpp available)

**System Dependencies:**

- [ ] Tesseract OCR installed and accessible
- [ ] llama.cpp installed and accessible
- [ ] CUDA available (if using GPU features)

## 📊 Configuration

If applicable, include your configuration:

```json
{
  "model": {
    "vocab_size": 16000,
    ...
  },
  "data": {
    "ingestion": {
      ...
    }
  }
}
```

## 📝 Error Logs

Include relevant error messages and stack traces:

```
Paste error logs here
```

## 🔍 Additional Context

Add any other context about the problem here:

- Sample data that causes the issue (if applicable)
- Workarounds you've tried
- Related issues or discussions
- Screenshots (if applicable)

## 🧪 Minimal Reproduction

If possible, provide a minimal code example that reproduces the issue:

```python
import llmbuilder as lb

# Minimal code that reproduces the bug
config = lb.load_config(preset="cpu_small")
# ... rest of the reproduction code
```

## 📋 Checklist

Before submitting, please check:

- [ ] I have searched existing issues for similar problems
- [ ] I have included all relevant environment information
- [ ] I have provided steps to reproduce the issue
- [ ] I have included error logs and stack traces
- [ ] I have tested with the latest version of LLMBuilder
- [ ] I have checked that all required dependencies are installed---
name: Bug report
about: Create a report to help us improve LLMBuilder
title: '[BUG] '
labels: 'bug'
as
