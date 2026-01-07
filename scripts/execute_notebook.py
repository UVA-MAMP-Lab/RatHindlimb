#!/usr/bin/env python
"""Execute Jupyter notebook with graceful kernel handling for OpenSim.

Simple caching strategy:
- Load cached outputs before execution (useful for displaying previous results)
- Execute ALL cells to maintain variable dependencies  
- Save outputs to cache after execution
- If all cells are cached and no code changed, skip execution entirely
"""

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
import sys
import signal
import os
import hashlib
import pickle
from pathlib import Path
from datetime import datetime

def signal_handler(sig, frame):
    """Handle interrupt signal."""
    print("\nInterrupted, exiting...")
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Cache directory
CACHE_DIR = Path('_freeze/execute')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def compute_cell_hash(cell):
    """Compute hash of cell source code."""
    if cell.cell_type != 'code':
        return None
    source = cell.source
    return hashlib.md5(source.encode('utf-8')).hexdigest()

def get_cache_path(cell_hash):
    """Get cache file path for a cell hash."""
    return CACHE_DIR / f"{cell_hash}.cache"

def save_to_cache(cell):
    """Save cell outputs to cache."""
    cell_hash = compute_cell_hash(cell)
    if not cell_hash:
        return
    
    cache_path = get_cache_path(cell_hash)
    cache_data = {
        'outputs': cell.outputs,
        'execution_count': cell.execution_count,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

def load_from_cache(cell):
    """Load cell outputs from cache. Returns True if loaded successfully."""
    cell_hash = compute_cell_hash(cell)
    if not cell_hash:
        return False
    
    cache_path = get_cache_path(cell_hash)
    if not cache_path.exists():
        return False
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        cell.outputs = cache_data['outputs']
        if 'execution_count' in cache_data:
            cell.execution_count = cache_data['execution_count']
        return True
    except Exception as e:
        print(f"Warning: Failed to load cache: {e}")
        return False

# Read the notebook
print("Reading notebook...")
with open('model_edits.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

print("Checking cache status...")
print(f"Cache directory: {CACHE_DIR.absolute()}")

# Check which cells are cached
code_cells = [(i, cell) for i, cell in enumerate(nb.cells) if cell.cell_type == 'code']
total_cells = len(code_cells)
cached_count = 0
uncached_cells = []

print()
for idx, (i, cell) in enumerate(code_cells):
    if load_from_cache(cell):
        cached_count += 1
        print(f"Cell {i+1}: Loaded from cache ✓")
    else:
        uncached_cells.append(i)
        print(f"Cell {i+1}: Not cached, needs execution")

print(f"\nCache summary: {cached_count}/{total_cells} cells cached")

if len(uncached_cells) == 0:
    # All cells cached - no execution needed!
    print("\n✓ All cells cached! Skipping execution entirely.")
else:
    # Execute ALL cells to maintain variable dependencies
    print(f"\n{len(uncached_cells)} cells need execution.")
    print(f"Executing entire notebook to maintain variable dependencies...")
    
    try:
        ep = ExecutePreprocessor(
            timeout=1800,  # 30 minutes for expensive cells
            kernel_name='python3', 
            allow_errors=False
        )
        
        print("Starting execution...")
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        
        print("\nExecution complete! Saving to cache...")
        
        # Save all cells to cache (refresh cached ones too)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                save_to_cache(cell)
        
        print(f"✓ Saved {total_cells} cells to cache")
        
    except CellExecutionError as e:
        print(f"\nError executing cell: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Write output notebook
print("\nSaving executed notebook...")
try:
    with open('model_edits_executed.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("✓ Saved to model_edits_executed.ipynb")
except Exception as e:
    print(f"Error saving notebook: {e}")
    sys.exit(1)

# Print final cache stats
cache_files = list(CACHE_DIR.glob('*.cache'))
print(f"\nFinal cache statistics:")
print(f"  - Total cache entries: {len(cache_files)}")
print(f"  - Cells that were cached at start: {cached_count}/{total_cells}")

# Force immediate exit to avoid OpenSim cleanup issues
print("\nForcing immediate exit...")
os._exit(0)
