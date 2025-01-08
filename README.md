# FJLT-Modal: Dynamics-preserving compression for modal flow analysis using Fast Johnson-Lindenstrauss Transforms
FJLT-Modal is a set of notebooks and python scripts that implement Fast Johnson-Lindenstrauss Transforms and apply them to modal analysis of complex datasets.

This repository contains all that is required to reproduce the figures and results in the paper: "Dynamics-preserving compression for modal flow analysis" by A. Glazkov and P. Schmid.
At present this includes applications to SVD and DMD decompostion methods.

## Table of contents
- [Installing](#installing)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installing
The code is presented as a set of python files and notebooks, in prefernece to a self-contained package, to enable the user to change and interact with the underlying code.

There are two main ways of setting up the repository to run the included code.
#### Option 1: Dev containers
- This is the simplest option, and works best if running on a linux machine, or server, and using VS Code.
- All the code dependencies and notebook environments are automatically set up in a sandboxed Docker container.
- **There may be some performance degredation due to less RAM on some machines.** All the examples will work, but loading in the data and performing some calculations may take a long time.
- If RAM is limited, run the notebooks with fewer data snapshots rather than the ~300-500 currently set in the notebooks.
- You may have to increase RAM and disk space quotas on the Docker Desktop application or through the Docker CLI.
#### Option 2: Using virtual environments
- You are responsible for setting up the coding enviornment and ensuring that the dependencies are correctly installed.
- A `requirements.txt` file is provided with the dependency versions that have been checked.
- Once the virtual environment is built, run the `./build_missing_directories.sh` script to build the directories required for storing the outputs from the notebooks.

This repository is verified using the packages and versions listed in `./requirements.txt`, however, the newest versions of these packages should work here too.

## Usage
### Organisation
The repository is organised in two main parts.
 - The `./utils` directory consists of the Python code used for reading in the data and processing it using the FJLT algorithms
 - The `./svd`, `./dmd` and `./performance-tests` directories contain Jypyter notebooks for generating the results.

### Running notebooks
1. Run the notebooks in `./notebooks/<specific notebook case>/scripts/` to generate the data for the plotting scripts.
2. Run the plotting notebooks in `./notebooks/<specific notebook case>/postproc/`.

### Downloading the data
The data used for this paper may be found at the following DOI address: https://doi.org/10.25781/KAUST-Y0F7D.

## Contributing
Bug reports are always welcome, raise these through issues in GitHub.

Ensure that the code satisfies style and formatting rules by installing pre-commit hooks defined in `.pre-commit-config.yaml`. To do this run
```
pre-commit install
```
to install the hooks and
```
pre-commit uninstall
```
to uninstall them.


A git filter, which uses [nbstripout](https://github.com/kynan/nbstripout) is used to ignore the outputs of the jupyter notebooks before the commit.
> [!WARNING]
>
> Changing branches with notebooks that have modified cells will result in the loss of metadata and outputs.

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
