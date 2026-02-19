# RatHindlimb

## Description

This repository contains code and data to generate a bilateral musculoskeletal model of the rat hindlimb in OpenSim, based on the original model from Johnson et al. (2008) updated to utilize attachment points from the work of Young et al. (2017), a more robust knee joint, muscle parameters from Johnson et al. (2011), and estimated tendon slack lengths based on the methods of Manal & Buchanan (2004) all mirrored to the contralateral limb. The model is intended for use in simulations of rat hindlimb biomechanics, including inverse kinematics, inverse dynamics, and computed muscle control. The repository includes scripts for model generation, muscle analysis, and visualization.

## Quickstart

- Clone the repository with submodules
- Run `make install` or `scripts/setup.sh`

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

Scripts are available in the `scripts/` folder, but can also be run from the `Makefile`.
Available targets include:

- `make render`
- `make preview`

## Contributing

### Repo Structure

### TODO

- [ ] Separate out muscle specific edits
- [ ] Move computational things in index.qmd to isolated notebooks
  - This might solve the kernel death problem
  - Will also allow easier inclusion in other works
- [ ] Package install instructions and change src.\* to rathindlimb.\*
  - Create setup script
- [x] Add osimpy as submodule
  - Eventually this should be a dependency
- [ ] Formalize muscle analysis functions
- [ ] Create tests for model validation
- [ ] Clean up intermediate model edits
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
