# Cache Management Utilities

## Overview

The execution caching system stores cell outputs in `_freeze/execute/` to speed up repeated renders.

## Quick Reference

### View Cache Status

```bash
# Count cached cells
ls _freeze/execute/*.cache | wc -l

# Show cache files with timestamps
ls -lh _freeze/execute/
```

### Clear Cache

```bash
# Clear all cached outputs
rm -rf _freeze/execute/*.cache

# Clear cache for specific cells (if you know the hash)
rm _freeze/execute/<hash>.cache
```

### Pre-populate Cache

If you have an already-executed notebook, populate the cache from it:

```bash
.venv/bin/python populate_cache.py
```

This reads `model_edits_executed.ipynb` and saves all cell outputs to cache.

## Cache File Format

Each `.cache` file contains:
- `outputs`: List of cell output objects (text, images, etc.)
- `execution_count`: Cell execution counter
- `timestamp`: ISO format timestamp of when cached

Files are named by MD5 hash of the cell's source code.

## How Caching Works

1. **On execution start**: Check each cell's source code hash
2. **If hash matches cached file**: Load outputs from cache
3. **Execute all cells**: Maintains variable dependencies
4. **After execution**: Save all cell outputs to cache

## When Cache is Invalidated

Cache for a cell is invalidated when:
- Cell source code changes (even whitespace!)
- Cache file is deleted
- Cache file is corrupted

## Tips

- **Cache is automatic**: Just run renders normally
- **Cache speeds up failure recovery**: If a late cell fails, early cells show cached outputs
- **Cache doesn't skip execution**: All cells still run (for variable dependencies)
- **Clear cache when debugging**: Ensures fresh execution

## Storage

- Location: `_freeze/execute/`
- Format: Python pickle (`.cache` files)
- Size: Varies by output (text is small, plots are larger)
- Typical: ~15-30 KB per cell with text output, ~30-50 KB with plots
