#!/usr/bin/env python
"""Execute Jupyter notebook with graceful kernel handling for OpenSim."""

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys
import signal
import os

def signal_handler(sig, frame):
    """Handle interrupt signal."""
    print("\nInterrupted, exiting...")
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Read the notebook
with open('model_edits.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

# Create executor with timeout and allow errors
ep = ExecutePreprocessor(timeout=600, kernel_name='python3', allow_errors=False)

print("Executing notebook...")
try:
    # Execute the notebook
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    print("Execution complete!")
    
    # Write output notebook
    with open('model_edits_executed.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Saved executed notebook to model_edits_executed.ipynb")
    
except Exception as e:
    print(f"Error during execution: {e}")
    # Try to save partial results anyway
    try:
        with open('model_edits_partial.ipynb', 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print("Saved partial notebook to model_edits_partial.ipynb")
    except:
        pass
    sys.exit(1)
finally:
    # Force immediate exit to avoid OpenSim cleanup issues
    print("Forcing immediate exit...")
    os._exit(0)
