# Workaround for Quarto + OpenSim Kernel Death Issue

## Problem

When running `quarto preview model_edits.qmd` or `quarto render model_edits.qmd`, all code blocks execute successfully, but the Jupyter kernel crashes during shutdown with the error:

```
ERROR: Kernel died
WARN: Should not have arrived here:
ERROR: [non-error-thrown] undefined
```

This is caused by OpenSim's C++ destructors interacting poorly with Python's garbage collection during kernel shutdown.

## Solution

We've implemented a 3-step workaround that executes the notebook separately before rendering:

1. **Convert** QMD to Jupyter notebook
2. **Execute** the notebook with graceful kernel handling  
3. **Render** the executed notebook to HTML (without re-execution)

## Usage

### Quick Start

Simply run the provided shell script:

```bash
./render_model_edits.sh
```

This will generate `model_edits_executed.html` with all code outputs.

### With Browser Preview

To automatically open the result in your browser:

```bash
./render_model_edits.sh --preview
```

### Manual Steps

If you prefer to run each step manually:

```bash
# 1. Convert QMD to notebook
quarto convert model_edits.qmd

# 2. Execute notebook with graceful cleanup
python execute_notebook.py

# 3. Render to HTML without execution
quarto render model_edits_executed.ipynb --to html
```

## Files

- `model_edits.qmd` - Original Quarto document
- `execute_notebook.py` - Python script that executes the notebook with proper cleanup handling
- `render_model_edits.sh` - Convenience script that runs the full workflow
- `model_edits_executed.ipynb` - Executed notebook (generated)
- `model_edits_executed.html` - Final HTML output (generated)

## Technical Details

The `execute_notebook.py` script uses:
- `nbformat` to read/write notebooks
- `ExecutePreprocessor` from `nbconvert` to execute cells
- `os._exit(0)` to force immediate exit after saving, avoiding OpenSim's problematic C++ destructors

This bypasses the kernel shutdown issue by terminating the Python process immediately after saving the executed notebook, before OpenSim objects are garbage collected.

## Alternative: Preview Mode

If you need live preview during development, you can still use `quarto preview` on the executed notebook:

```bash
quarto preview model_edits_executed.ipynb
```

This will work because the notebook is already executed and Quarto won't need to run the kernel.
