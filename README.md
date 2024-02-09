[![Conda Version](https://anaconda.org/conda-forge/tjpcov/badges/version.svg)](https://anaconda.org/conda-forge/tjpcov/)
[![PyPI version](https://badge.fury.io/py/tjpcov.svg)](https://badge.fury.io/py/tjpcov)
[![Coverage Status](https://coveralls.io/repos/github/LSSTDESC/TJPCov/badge.svg?branch=master)](https://coveralls.io/github/LSSTDESC/TJPCov?branch=master)
[![Documentation Status](https://readthedocs.org/projects/tjpcov/badge/?version=latest)](https://tjpcov.readthedocs.io/en/latest/?badge=latest)


# TJPCov

TJPCov is a general covariance calculator interface to be used within LSST DESC.

## Installation

### Quickstart
The easiest and recommended way to install TJPCov is to install it via conda:
```
conda install -c conda-forge tjpcov
```

Alternatively you may also install tjpcov via PyPi:
```
pip install tjpcov 
```
will install TJPCov with minimal dependencies, and
```
pip install 'tjpcov[full]'
```
will include all dependencies (for details, see Optional dependencies (PyPi only) section)

### Developer Installation
If you wish to contribute to TJPCov, follow the steps below to set up your development environment.

1. Clone the repository
2. Create the conda environment with `conda env create --file environment.yml`
3. Activate the environment with `conda activate tjpcov`
4. Run `pip install -e .`
5. Run `pytest -vv tests/`


### Optional dependencies (PyPi only)
Because TJPCov relies on some packages that may not be necessary for every user, we have added different installation options to accommodate different use cases. For example, if a user has no plans to use MPI with TJPCov, they do not need `mpi4py`. Below we list the different installation options available on PyPi.

 - `pip install tjpcov` will install tjpcov and the minimal dependencies.
 - `pip install tjpcov'[doc]'` will install tjpcov, the minimal
     dependencies and the dependencies needed to build the documentation.
 - `pip install 'tjpcov[nmt]'` will install tjpcov, the minimal
     dependencies and the dependencies needed to use NaMaster.
 - `pip install 'tjpcov[mpi4py]'` will install, the minimal
     dependencies and the mpi4py library to use MPI parallelization.
 - `pip install 'tjpcov[full]'` will install tjpcov and all dependencies


#### Developer installation (PyPi only)
If you are using PyPi to set up your development environment (we recommend using conda instead), due to a bug in the NaMaster installation, one needs to make sure numpy is installed before trying to install NaMaster. For a fresh install, run `python -m pip install .` first, and then `python -m pip install .\[nmt\]`


## Planning & development

Ask @felipeaoli or @mattkwiecien for access to the repository and join the #desc-mcp-cov channel on the LSST DESC slack to contribute.

We have adopted the following style convention (which are enforced in each PR):
 - [Google-style docstrings](https://google.github.io/styleguide/pyguide.html)
 - [Black code style](https://github.com/psf/black) (with 79 characters line-width)
 - PEP8 except for E203 (for better compatibility with black)

For a general idea of TJPCov's scientific scope, see also the [terms of reference](https://github.com/LSSTDESC/TJPCov/blob/master/doc/Terms_of_Reference.md).

## Contributing

We use `black` and `flake8` configuration files so that code follows a unified coding style and remains PEP8 compliant.

This means before submitting your PR you must run the following in the root directory:
```
black .
flake8 .
```
Furthermore, we are following GitHub's recommendation of using [Semantic Versioning](https://semver.org/) in our releases.


## Supported Python Versions
TJPCov currently runs on python 3.8, but python 3.9, 3.10 and 3.11 are supported.

TJPCov also has a few specific software versions hardcoded.  Please check the `pyproject.toml` file to see version requirements.
