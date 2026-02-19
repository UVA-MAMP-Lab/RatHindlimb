# RatHindlimb

## Description

This repository contains code and data to generate a bilateral musculoskeletal model of the rat hindlimb in OpenSim, based on the original model from Johnson et al. (2008) updated to utilize attachment points from the work of Young et al. (2017), a more robust knee joint, muscle parameters from Johnson et al. (2011), and estimated tendon slack lengths based on the methods of Manal & Buchanan (2004) all mirrored to the contralateral limb. The model is intended for use in simulations of rat hindlimb biomechanics, including inverse kinematics, inverse dynamics, and computed muscle control. Quarto handles execution caching for model-generation notebooks/documents.

## Quickstart

- Clone the repository with submodules
- Create and activate the conda environment

### Installation

If not already installed, install:

- [git](https://git-scm.com/install/)
- [conda](https://www.anaconda.com/docs/getting-started/miniconda/install)
  - Miniconda is sufficient, but any anaconda installation will work

``` shell
# Clone the repository and necessary submodules
git clone --recurse-submodules 

# Install dependencies
conda env create -f environment.yml
conda activate rathindlimb

# Install the package
python -m pip install -e .
```

### Usage

Render with Quarto (narrative only):

``` shell
quarto render index.qmd
```

Run model edits in staged notebooks:

- `notebooks/pipeline/01_non_muscle_edits.ipynb`
- `notebooks/pipeline/02_muscle_edits.ipynb`
- `notebooks/pipeline/03_mirroring.ipynb`

Final published models are written to:

- `models/osim/rat_hindlimb_unilateral.osim`
- `models/osim/rat_hindlimb_unilateral_no_muscles.osim`
- `models/osim/rat_hindlimb_bilateral.osim`
- `models/osim/rat_hindlimb_bilateral_no_muscles.osim`

## Contributing

### Repo Structure

### TODO

- [ ] Separate out muscle specific edits
- [ ] Move computational things in index.qmd to isolated notebooks
  - This is now compatible with branch-aware artifact saving and Quarto caching
- [ ] Package install instructions and change src.\* to rathindlimb.\*
  - Create setup script
- [x] Add osimpy as submodule
  - Eventually this should be a dependency
- [ ] Formalize muscle analysis functions
- [ ] Create tests for model validation
- [x] Clean up intermediate model edits
- [ ] Clean up conda environment.yml
- [ ] Organize script usage into Makefile
- [ ] Switch to uv for dependency management
  - Currently waiting for opensim bindings to be easily available
  - Pyopensim doesn't quite work

## References and Acknowledgements

- Johnson 2008
- Johnson 2011
- Eng 2008
- Manal & Buchanan 2004
- Young 2017
- Open3D
- Hicks
- Dienes
- Delp? / OpenSim
- Charles 2016

## Citing

This model is associated with the publication ...

If you use this repository in your research, please cite:

```json


```
