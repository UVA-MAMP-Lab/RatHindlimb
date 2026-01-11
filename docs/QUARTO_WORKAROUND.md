# Workaround for Quarto + OpenSim Kernel Death Issue

## Problem

When running `quarto preview model_edits.qmd` or `quarto render model_edits.qmd`, all code blocks execute successfully, but the Jupyter kernel crashes during shutdown with the error:

```
ERROR: Kernel died
WARN: Should not have arrived here:
ERROR: [non-error-thrown] undefined
```

I hypothesize that this is caused by OpenSim's C++ destructors interacting poorly with Python's garbage collection during kernel shutdown.

## Solution

We've implemented a 3-step workaround with caching support:

1. **Convert** QMD to Jupyter notebook
2. **Execute** the notebook with graceful kernel handling and caching
3. **Render** the executed notebook to HTML (without re-execution)

## Usage

### Option 1: Manual Render (Recommended)

Simply run the provided shell script:

```bash
./render_model_edits.sh
```

This will generate `model_edits_executed.html` with all code outputs.

To automatically open the result in your browser:

```bash
./render_model_edits.sh --preview
```

### Option 2: File Watcher (Auto-render on save)

For active development, use the file watcher to automatically re-render when you save changes:

```bash
./watch_and_render.py
```

This will:

- Monitor `model_edits.qmd` for changes
- Automatically trigger a full render when you save the file
- Use cached outputs to speed up re-renders
- Print progress to the terminal

**Usage:**

1. Start the watcher: `./watch_and_render.py`
2. Edit `model_edits.qmd` in your text editor
3. Save the file
4. Wait for render to complete (~2-3 minutes with cache)
5. Refresh your browser to see changes

Press `Ctrl+C` to stop the watcher.

### Manual Steps

If you prefer to run each step manually:

```bash
# 1. Convert QMD to notebook
quarto convert model_edits.qmd

# 2. Execute notebook with graceful cleanup
.venv/bin/python execute_notebook.py

# 3. Render to HTML without execution
quarto render model_edits_executed.ipynb --to html
```

## Caching System

The execution script includes a caching system to speed up repeated renders:

### How It Works

- **Cell outputs are cached** in `_freeze/execute/` directory
- Each cell's cache is keyed by MD5 hash of its source code
- When code hasn't changed, cached outputs are loaded instantly
- All cells are re-executed to maintain variable dependencies
- Cache is preserved across renders

### Benefits

- **Fast preview**: See previous outputs immediately while cells re-execute
- **Failure recovery**: If kernel dies mid-execution, cached outputs are preserved
- **Partial results**: Can display some results even if later cells fail

### Cache Management

Clear the cache if you need a completely fresh execution:

```bash
rm -rf _freeze/execute/*.cache
```

## Files

### Source Files

- `model_edits.qmd` - Original Quarto document (edit this!)
- `execute_notebook.py` - Python script that executes notebooks with caching
- `render_model_edits.sh` - Convenience script for manual renders
- `watch_and_render.py` - File watcher for auto-rendering
- `populate_cache.py` - Utility to pre-populate cache from existing notebook

### Generated Files

- `model_edits.ipynb` - Converted notebook (temporary)
- `model_edits_executed.ipynb` - Executed notebook with outputs
- `model_edits_executed.html` - Final HTML output
- `_freeze/execute/*.cache` - Cached cell outputs

## Technical Details

### Execution Script

The `execute_notebook.py` script:

- Uses `nbformat` to read/write notebooks
- Uses `ExecutePreprocessor` from `nbconvert` to execute cells
- Computes MD5 hash of each cell's source code for caching
- Stores outputs as pickled objects in `_freeze/execute/`
- Uses `os._exit(0)` to force immediate exit, avoiding OpenSim cleanup issues

### Timeout

Cells have a 30-minute timeout to accommodate expensive operations:

- Muscle ROM analysis with 10,000+ points
- Mesh registration (ICP algorithm)
- TSL optimization loops

### Known Limitations

1. **Cache doesn't skip execution** - All cells are re-executed even if cached. This maintains variable dependencies between cells.
2. **Memory usage** - Cache stores full cell outputs in memory during load/save
3. **Hash-based invalidation** - Any code change invalidates that cell's cache

## Troubleshooting

### "Kernel died" error during manual render

This is expected with direct Quarto execution. Use the workaround scripts instead.

### Execution takes too long

Check if expensive cells are running:

- Muscle ROM analysis: ~1-5 minutes
- Mesh registration: ~1-3 minutes per mesh
- TSL optimization: ~1-5 minutes

### Cache not working

1. Check if `_freeze/execute/` directory exists and contains `.cache` files
2. Verify cell source code hasn't changed (cache is hash-based)
3. Clear cache and try again: `rm -rf _freeze/execute/*.cache`

### Import errors

If you see `ModuleNotFoundError` for src modules, verify:

- You're using the conda environment: `.venv/bin/python`
- Imports use `from src.module_name import` (not `from src.rathindlimb.`)

## Alternative: Preview Already-Executed Notebook

If you need Quarto's live preview server, you can preview the executed notebook:

```bash
quarto preview model_edits_executed.ipynb
```

This works because the notebook is already executed and Quarto won't need to run the kernel.
