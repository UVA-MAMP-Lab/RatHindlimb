#!/usr/bin/env python
"""Simple test - just execute the notebook without any caching logic."""

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

print("Reading notebook...")
with open('model_edits.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

print("Executing notebook...")
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': '.'}})

print("Saving...")
with open('test_output.ipynb', 'w') as f:
    nbformat.write(nb, f)

print("Done!")
os._exit(0)
