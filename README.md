# RatHindlimb

## Installation
To run the `mirror_hindlimb.py` script, you will need to install the OpenSim API. The following instructions are for installing OpenSim 4.4.1 on a Windows machine using the [Conda](https://docs.conda.io/en/latest/) environment and package manager. I recommend using [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) if you don't already have Conda installed.

```shell
conda create -n {env_name} -c opensim-org opensim=4.4.1 numpy 
```

The script may be run by activating the Conda environment and running the script with Python. 

```shell
conda activate {env_name}
python mirror_hindlimb.py
```

If you are unable to run it, try making the script executable by running the following command in the terminal from the directory containing the script:

```shell
chmod +x mirror_hindlimb.py
```
