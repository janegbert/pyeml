# pyeml

[![Tests](https://github.com/janegbert/pyeml/actions/workflows/tests.yml/badge.svg)](https://github.com/janegbert/pyeml/actions/workflows/tests.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Symbolic regression using a single universal operator: **eml(x, y) = exp(x) - ln(y)**.

Based on [Odrzywołek (2026)](https://arxiv.org/abs/2603.21852): *"All elementary functions from a single binary operator"*. This operator, together with the constant 1, can express every elementary function — sin, cos, sqrt, log, +, -, \*, /, and more. The continuous-math equivalent of NAND in digital logic.

## Install

```bash
pip install pyeml
```

Requires Python 3.12+ and PyTorch 2.0+.

## Quick start

```python
import numpy as np
from pyeml import discover

# Feed it data — it discovers the exact formula
x = np.linspace(0.1, 2.0, 200)
y = np.exp(x)

result = discover(x, y, max_depth=3)
print(result.expression)      # "exp(x)"
print(result.eml_expression)  # "eml(x, 1)"
print(result.is_exact)        # True
```

## Sklearn-style API

```python
from pyeml import EMLRegressor

reg = EMLRegressor(max_depth=3, n_restarts=10)
reg.fit(x, y)

print(reg.expression)    # "exp(x)"
y_pred = reg.predict(x)  # exact predictions
```

## CLI

```bash
# Verify paper identities
pyeml verify

# Compile known expressions to EML form
pyeml compile --expr "ln(x)"
# -> eml(1, eml(eml(1, x), 1))  Depth: 3  Leaves: 4

# Discover from a target function
pyeml discover --expr exp --depth 3

# Run the demo
pyeml demo
```

## How it works

Every mathematical expression becomes a **binary tree of identical nodes**, each computing `eml(x, y) = exp(x) - ln(y)`. The grammar is:

```
S -> 1 | x | eml(S, S)
```

Training uses the **master formula** from the paper: each node input is a softmax-weighted mix of `{1, x, child_output}`. Adam optimizer finds the weights, then a hardening phase snaps them to `{0, 1}`, recovering the exact symbolic expression.

| Function | EML form | Depth |
|----------|----------|-------|
| exp(x) | eml(x, 1) | 1 |
| e | eml(1, 1) | 1 |
| ln(x) | eml(1, eml(eml(1, x), 1)) | 3 |

Recovery rates (from the paper): 100% at depth 2, ~25% at depth 3-4, <1% at depth 5+. Multiple random restarts compensate.

## Package structure

```
src/pyeml/
  _operator.py   # eml() core function
  _tree.py       # EMLTree (PyTorch nn.Module)
  _trainer.py    # Multi-phase training engine
  _compiler.py   # Standard math <-> EML conversion
  _symbolic.py   # Expression extraction from trained trees
  _api.py        # discover() + EMLRegressor
  _cli.py        # Command-line interface
  _config.py     # TrainConfig, SearchConfig
```

## Citation

```bibtex
@article{odrzywołek2026eml,
  title={All elementary functions from a single binary operator},
  author={Odrzywołek, Andrzej},
  journal={arXiv preprint arXiv:2603.21852},
  year={2026}
}
```

## Built with

This package was built with [Claude Code](https://claude.ai/code) (Claude Opus 4.6). It typed fast.

## License

MIT
