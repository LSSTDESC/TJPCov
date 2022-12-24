[![Coverage Status](https://coveralls.io/repos/github/LSSTDESC/TJPCov/badge.svg?branch=master)](https://coveralls.io/github/LSSTDESC/TJPCov?branch=master)
[![Documentation Status](https://readthedocs.org/projects/tjpcov/badge/?version=latest)](https://tjpcov.readthedocs.io/en/latest/?badge=latest)

# TJPCov

TJPCov is a general covariance calculator interface to be used within LSST DESC.

## Installation

TJPCov is now pip installable for convenience, but for development
clone the git repository.

There are five different flavors of tjpcov at the moment:
 - `python -m pip install .` will install tjpcov and the minimal dependencies.
 - `python -m pip install .\[doc\]` will install tjpcov, the minimal
     dependencies and the dependencies needed to build the documentation.
 - `python -m pip install .\[nmt\]` will install tjpcov, the minimal
     dependencies and the dependencies needed to use NaMaster.
 - `python -m pip install .\[mpi4py\]` will install, the minimal
     dependencies and the mpi4py library to use MPI parallelization.
 - `python -m pip install .\[full\]` will install tjpcov and all dependencies

Note that due to a bug in the NaMaster installation, one needs to make sure
numpy is installed before trying to install NaMaster. If you are doing a fresh
install, run `python -m pip install .` first, and then `python -m pip install .\[nmt\]`

## Planning & development

Ask @felipeaoli or @carlosggarcia for access to the repository and join the #desc-mcp-cov channel on the LSST DESC slack to contribute.

See also [terms of reference](https://github.com/LSSTDESC/TJPCov/blob/master/doc/Terms_of_Reference.md).

## Environment for development
If you are working in conda (miniconda or anaconda) you can create a conda environment named **tjpcov** with 
```
conda env create --file environment.yml
```

To activate your new environment use:

```
conda activate tjpcov
```

## Dependencies and versioning
The latest version TJPCov needs pymaster >= 1.4 . Install it using (after `conda activate tjpcov`): 

```
python -m pip install pymaster>=1.4
```
The code requires ccl>=2.5.0
```
python -m pip install ccl>=2.5.0
```

