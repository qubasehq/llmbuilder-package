# Pull Request

## 📋 Description

**What does this PR do?**
A clear and concise description of what this pull request accomplishes.

**Related Issues:**

- Fixes #[issue_number]
- Related to #[issue_number]
- Part of #[issue_number]

## 🎯 Type of Change

What type of change does this PR introduce?

- [ ] **Bug fix** (non-breaking change that fixes an issue)
- [ ] **New feature** (non-breaking change that adds functionality)
- [ ] **Breaking change** (fix or feature that would cause existing functionality to not work as expected)
- [ ] **Documentation update** (changes to documentation only)
- [ ] **Performance improvement** (non-breaking change that improves performance)
- [ ] **Code refactoring** (non-breaking change that doesn't add features or fix bugs)
- [ ] **Test improvements** (adding or improving tests)
- [ ] **CI/CD changes** (changes to build, deploy, or test automation)
- [ ] **Dependency updates** (updating dependencies)

## 🔄 Changes Made

**Detailed list of changes:**

### Code Changes

- [ ] Added new functionality: [describe]
- [ ] Modified existing functionality: [describe]
- [ ] Fixed bug: [describe]
- [ ] Improved performance: [describe]
- [ ] Refactored code: [describe]

### Configuration Changes

- [ ] Added new configuration options
- [ ] Modified existing configuration options
- [ ] Updated configuration templates
- [ ] Added configuration validation

### CLI Changes

- [ ] Added new commands
- [ ] Modified existing commands
- [ ] Improved help text
- [ ] Added new options/flags

### Documentation Changes

- [ ] Updated README.md
- [ ] Updated user guides
- [ ] Added/updated examples
- [ ] Updated API documentation
- [ ] Updated CONTRIBUTING.md

### Test Changes

- [ ] Added unit tests
- [ ] Added integration tests
- [ ] Added performance tests
- [ ] Fixed failing tests
- [ ] Improved test coverage

## 🧪 Testing

**How has this been tested?**

### Test Environment

- OS: [e.g., Ubuntu 20.04, Windows 10, macOS 12.0]
- Python version: [e.g., 3.9.7]
- Dependencies: [list any special dependencies]

### Test Cases

- [ ] **Unit tests**: All new and existing unit tests pass
- [ ] **Integration tests**: All integration tests pass
- [ ] **Manual testing**: Manually tested the changes
- [ ] **Performance tests**: Performance impact assessed
- [ ] **Regression testing**: Verified no existing functionality is broken

### Test Commands Run

```bash
# List the commands you ran to test this PR
python -m pytest tests/ -v
python -m pytest tests/integration/ -v
# Add any specific test commands
```

### Test Results

```
# Paste relevant test output or results
```

## 📊 Performance Impact

**Does this change affect performance?**

- [ ] No performance impact
- [ ] Improves performance
- [ ] May impact performance (explain below)
- [ ] Performance impact unknown

**Performance details:**
[If applicable, describe performance impact, include benchmarks, memory usage, etc.]

## 🔒 Breaking Changes

**Does this PR introduce breaking changes?**

- [ ] No breaking changes
- [ ] Yes, breaking changes (describe below)

**Breaking change details:**
[If yes, describe what breaks and how users should adapt]

**Migration guide:**
[If applicable, provide migration instructions for users]

## 📚 Documentation

**Documentation updates:**

- [ ] Code is self-documenting with clear variable/function names
- [ ] Added/updated docstrings for new/modified functions
- [ ] Updated user documentation
- [ ] Updated API documentation
- [ ] Added/updated examples
- [ ] Updated CHANGELOG.md (for significant changes)

**Documentation locations updated:**

- [ ] README.md
- [ ] docs/user-guide/
- [ ] docs/examples/
- [ ] docs/api/
- [ ] Inline code comments
- [ ] CLI help text

## 🔗 Dependencies

**Dependency changes:**

- [ ] No new dependencies
- [ ] Added new required dependencies
- [ ] Added new optional dependencies
- [ ] Updated existing dependencies
- [ ] Removed dependencies

**New dependencies added:**

```
# List any new dependencies and why they're needed
```

## 📋 Checklist

**Before submitting this PR, please check:**

### Code Quality

- [ ] Code follows the project's style guidelines
- [ ] Code has been formatted with Black
- [ ] Imports have been sorted with isort
- [ ] Code passes flake8 style checks
- [ ] Code passes mypy type checking
- [ ] Pre-commit hooks pass

### Testing

- [ ] All existing tests pass
- [ ] New tests have been added for new functionality
- [ ] Test coverage is maintained or improved
- [ ] Integration tests pass (if applicable)
- [ ] Performance tests pass (if applicable)

### Documentation

- [ ] Documentation has been updated
- [ ] Examples have been added/updated
- [ ] API documentation is current
- [ ] CHANGELOG.md updated (for significant changes)

### Review Readiness

- [ ] PR title is clear and descriptive
- [ ] PR description explains the changes and motivation
- [ ] Commits are logically organized
- [ ] Commit messages are clear and follow conventions
- [ ] No debugging code or commented-out code left in
- [ ] No merge conflicts

## 🎯 Review Focus

**What should reviewers focus on?**

- [ ] Code correctness and logic
- [ ] Performance implications
- [ ] Security considerations
- [ ] API design and usability
- [ ] Documentation clarity
- [ ] Test coverage and quality
- [ ] Breaking change impact
- [ ] Configuration changes

**Specific areas needing attention:**
[Highlight any specific areas where you'd like focused review]

## 📸 Screenshots/Examples

**If applicable, add screenshots or examples:**

```python
# Example usage of new feature
import llmbuilder as lb

# Show how the new feature works
```

## 🤔 Questions for Reviewers

**Questions or concerns:**

- [Any specific questions you have for reviewers]
- [Areas where you're unsure about the implementation]
- [Alternative approaches you considered]

## 📝 Additional Notes

**Additional context:**
[Any other information that would be helpful for reviewers]

---

**By submitting this pull request, I confirm that:**

- [ ] I have read and followed the contributing guidelines
- [ ] My code follows the project's coding standards
- [ ] I have tested my changes thoroughly
- [ ] I have updated documentation as needed
- [ ] I understand that this contribution will be licensed under the project's license# Pull Request

## 📋 Description

**What does this PR do?**
A clear and concise description of what this pull request accomplishes.

**Related Issue(s)**

- Fixes #___
- Closes #___
- Related to #___

## 🔄 Type of Change

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🧪 Test improvements
- [ ] 🔧 Refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] 🎨 Code style/formatting changes

## 🧪 Testing

**How has this been tested?**

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Performance tests added/updated
- [ ] Manual testing performed
- [ ] Existing tests still pass

**Test Configuration**:

- **OS**: [e.g. Ubuntu 20.04, Windows 10, macOS 12.0]
- **Python version**: [e.g. 3.9, 3.10, 3.11]
- **Dependencies**: [e.g. with/without optional dependencies]

**Test Results**:

```bash
# Include relevant test output
pytest tests/ -v
```

## 📝 Changes Made

**Files Changed**:

- `llmbuilder/module/file.py` - Description of changes
- `tests/test_module.py` - Added tests for new functionality
- `docs/user-guide/guide.md` - Updated documentation

**Key Changes**:

1. **Change 1**: Description of what was changed and why
2. **Change 2**: Description of what was changed and why
3. **Change 3**: Description of what was changed and why

## 🔧 Configuration Changes

**Are there any configuration changes?**

- [ ] No configuration changes
- [ ] New configuration options added
- [ ] Existing configuration options modified
- [ ] Configuration options deprecated/removed

**If yes, describe the changes**:

```python
# Example of new configuration options
config = {
    "new_feature": {
        "option1": "default_value",
        "option2": True
    }
}
```

## 📚 Documentation

**Documentation updates included?**

- [ ] Code comments updated
- [ ] API documentation updated
- [ ] User guide updated
- [ ] Examples added/updated
- [ ] README updated
- [ ] CHANGELOG updated

**If documentation is needed but not included, explain why**:

## 🚀 Performance Impact

**Does this change affect performance?**

- [ ] No performance impact
- [ ] Performance improvement
- [ ] Potential performance regression
- [ ] Performance impact unknown

**If there's a performance impact, provide details**:

- **Benchmarks**: Include before/after performance measurements
- **Memory usage**: Any changes in memory consumption
- **Processing speed**: Any changes in processing speed

## 🔄 Backward Compatibility

**Is this change backward compatible?**

- [ ] Yes, fully backward compatible
- [ ] No, breaking changes (explain below)
- [ ] Partially compatible (explain below)

**If not fully compatible, describe the breaking changes**:

## ✅ Checklist

**Before submitting this PR, please make sure**:

- [ ] Code follows the project's style guidelines
- [ ] Self-review of code has been performed
- [ ] Code is commented, particularly in hard-to-understand areas
- [ ] Tests have been added that prove the fix is effective or that the feature works
- [ ] New and existing unit tests pass locally
- [ ] Any dependent changes have been merged and published
- [ ] Documentation has been updated
- [ ] CHANGELOG.md has been updated (if applicable)

**Code Quality**:

- [ ] Code has been formatted with `black`
- [ ] Imports have been sorted with `isort`
- [ ] Code passes `flake8` linting
- [ ] Type hints have been added where appropriate
- [ ] Docstrings follow Google style

**Testing**:

- [ ] All existing tests pass
- [ ] New tests have been added for new functionality
- [ ] Tests cover edge cases and error conditions
- [ ] Integration tests have been considered
- [ ] Performance impact has been evaluated

## 🔍 Review Notes

**Specific areas to focus on during review**:

- Area 1: Specific concern or question
- Area 2: Complex logic that needs careful review
- Area 3: Performance-critical code

**Questions for reviewers**:

1. Question about implementation approach
2. Question about naming or API design
3. Question about test coverage

## 📸 Screenshots (if applicable)

**Before**:
[Screenshot of before state]

**After**:
[Screenshot of after state]

## 🔗 Additional Context

**Related PRs**:

- Depends on #___
- Blocks #___
- Related to #___

**External References**:

- Links to relevant documentation
- Links to research papers or articles
- Links to related issues in other projects

---

**Note for Reviewers**: Please check that all tests pass and documentation is updated before approving. For breaking changes, ensure that the impact is clearly documented and justified.
