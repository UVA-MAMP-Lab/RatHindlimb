#!/usr/bin/env python
"""Execute Jupyter notebook with graceful kernel handling and cache support for OpenSim.

Caching Strategy:
- If ALL cells are cached: Skip execution entirely, use cached outputs
- If ANY cell needs execution: Execute from first uncached cell onward
  (to maintain execution context and variable dependencies)
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

def should_skip_cache(cell):
    """Check if cell should skip cache (has cache: false tag)."""
    if cell.cell_type != 'code':
        return False
    
    # Check for #| cache: false comment
    source = cell.source
    if '#| cache: false' in source or '#| cache:false' in source:
        return True
    
    return False

# Read the notebook
print("Reading notebook...")
with open('model_edits.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

print("Executing notebook with caching enabled...")
print(f"Cache directory: {CACHE_DIR.absolute()}")

# Check cache status and find first uncached cell
code_cells = [(i, cell) for i, cell in enumerate(nb.cells) if cell.cell_type == 'code']
total_cells = len(code_cells)
first_uncached_index = None
cached_count = 0
uncached_cells = []

print("\nChecking cache status...")
for idx, (i, cell) in enumerate(code_cells):
    skip_cache = should_skip_cache(cell)
    
    if skip_cache:
        if first_uncached_index is None:
            first_uncached_index = i
        uncached_cells.append(i)
        print(f"Cell {i+1}: Will execute (cache disabled)")
    elif load_from_cache(cell):
        cached_count += 1
        print(f"Cell {i+1}: Loaded from cache ✓")
    else:
        if first_uncached_index is None:
            first_uncached_index = i
        uncached_cells.append(i)
        print(f"Cell {i+1}: Not cached, will execute")

print(f"\nCache summary: {cached_count}/{total_cells} cells from cache")

if first_uncached_index is None:
    # All cells cached - no execution needed!
    print("\n✓ All cells cached! Skipping execution entirely.")
    cells_to_cache = []
else:
    # Need to execute from first_uncached_index onward to maintain context
    cells_to_execute = [i for i, cell in enumerate(nb.cells) if i >= first_uncached_index and cell.cell_type == 'code']
    cells_to_cache = uncached_cells  # Only cache the ones that weren't cached before
    
    print(f"\nFirst uncached cell: {first_uncached_index + 1}")
    print(f"Will execute cells {first_uncached_index + 1} through {len(nb.cells)} ({len(cells_to_execute)} code cells)")
    print(f"Will update cache for {len(cells_to_cache)} cells")
    
    try:
        # Create a custom preprocessor that starts from first_uncached_index
        class PartialExecutePreprocessor(ExecutePreprocessor):
            def __init__(self, *args, start_index=0, **kwargs):
                super().__init__(*args, **kwargs)
                self.start_index = start_index
            
            def preprocess_cell(self, cell, resources, index):
                """Skip cells before start_index."""
                if index < self.start_index:
                    return cell, resources
                return super().preprocess_cell(cell, resources, index)
        
        ep = PartialExecutePreprocessor(
            timeout=600, 
            kernel_name='python3', 
            allow_errors=False,
            start_index=first_uncached_index
        )
        
        print("\nStarting execution...")
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        
        print("\nExecution complete! Updating cache...")
        
        # Save newly executed cells to cache
        for i in cells_to_cache:
            cell = nb.cells[i]
            if cell.cell_type == 'code':
                save_to_cache(cell)
        
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
print(f"  - Cells loaded from cache this run: {cached_count}")
if cells_to_cache:
    print(f"  - Cells newly cached this run: {len(cells_to_cache)}")

# Force immediate exit to avoid OpenSim cleanup issues
print("\nForcing immediate exit...")
os._exit(0)


# Read the notebook
print("Reading notebook...")
with open('model_edits.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

print("Executing notebook with caching enabled...")
print(f"Cache directory: {CACHE_DIR.absolute()}")

# Check cache and load what we can
total_cells = sum(1 for cell in nb.cells if cell.cell_type == 'code')
cached_indices = set()
cells_to_execute = []

print("\nChecking cache...")
for i, cell in enumerate(nb.cells):
    if cell.cell_type != 'code':
        continue
    
    skip_cache = should_skip_cache(cell)
    
    if skip_cache:
        cells_to_execute.append(i)
        print(f"Cell {i+1}: Will execute (cache disabled)")
    elif load_from_cache(cell):
        cached_indices.add(i)
        print(f"Cell {i+1}: Loaded from cache ✓")
    else:
        cells_to_execute.append(i)
        print(f"Cell {i+1}: Not cached, will execute")

cached_count = len(cached_indices)
print(f"\nCache summary: {cached_count}/{total_cells} cells from cache, {len(cells_to_execute)} to execute")

if len(cells_to_execute) == 0:
    print("\nAll cells cached! Skipping execution.")
else:
    print(f"\nExecuting {len(cells_to_execute)} cells...")
    
    try:
        # Create custom executor that skips cached cells
        ep = CachingExecutePreprocessor(
            timeout=600, 
            kernel_name='python3', 
            allow_errors=False,
            cached_indices=cached_indices
        )
        
        # Execute the notebook (cached cells will be skipped)
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        
        print("\nExecution complete! Saving to cache...")
        
        # Save newly executed cells to cache
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code' and i in cells_to_execute:
                save_to_cache(cell)
        
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
print(f"  - Cells loaded from cache this run: {cached_count}")
print(f"  - Cells executed this run: {len(cells_to_execute)}")

# Force immediate exit to avoid OpenSim cleanup issues
print("\nForcing immediate exit...")
os._exit(0)
