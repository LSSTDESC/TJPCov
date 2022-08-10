# TJPCov

TJPCov is a general covariance calculator interface to be used within LSST DESC.

## Installation

TJPCov is now pip installable for convenience, but for development
clone the git repository.

## Planning & development

Ask @felipeaoli or @carlosggarcia for access to the repository and join the #desc-mcp-cov channel on the LSST DESC slack to contribute.

See also [terms of reference](https://github.com/LSSTDESC/TJPCov/blob/master/doc/Terms_of_Reference.md).

## Adding new versions to pip

When you want to push a new version to pip (the server is called PyPI) then:

1. increase the version number in setup.py
2. create an account at pypi.org if you don't have one already
3. run these commands:

```
# just the first time:
pip3 install twine

# remove the old distribution
rm -r dist

# make the distribution files
python3 setup.py sdist bdist_wheel

# upload
python3 -m twine upload  dist/*
```

## Environment for development
If you are working in conda (miniconda or anaconda) you can create a conda environment named **tjpcov** with 
```
conda env create --file environment.yml
```

To activate your new environment use:

```
source activate tjpcov
```

