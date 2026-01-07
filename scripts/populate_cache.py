#!/usr/bin/env python
"""Populate cache from an already-executed notebook."""

import nbformat
import hashlib
import pickle
from pathlib import Path
from datetime import datetime

# Cache directory
CACHE_DIR = Path('_freeze/execute')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def compute_cell_hash(cell):
    """Compute hash of cell source code."""
    if cell.cell_type != 'code':
        return None
    source = cell.source
    return hashlib.md5(source.encode('utf-8')).hexdigest()

def save_to_cache(cell):
    """Save cell outputs to cache."""
    cell_hash = compute_cell_hash(cell)
    if not cell_hash or not cell.outputs:
        return False
    
    cache_path = CACHE_DIR / f"{cell_hash}.cache"
    cache_data = {
        'outputs': cell.outputs,
        'execution_count': cell.execution_count,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        return True
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")
        return False

# Read the executed notebook
print("Reading executed notebook...")
with open('model_edits_executed.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

print(f"Populating cache from executed notebook...")
print(f"Cache directory: {CACHE_DIR.absolute()}\n")

cached_count = 0
for i, cell in enumerate(nb.cells):
    if cell.cell_type != 'code':
        continue
    
    if save_to_cache(cell):
        cached_count += 1
        print(f"Cell {i+1}: Cached ✓")
    else:
        print(f"Cell {i+1}: Skipped (no outputs)")

print(f"\n✓ Populated cache with {cached_count} cells")
print(f"Cache location: {CACHE_DIR.absolute()}")
