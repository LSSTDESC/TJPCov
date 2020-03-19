# TJPCov

TJPCov is a general covariance calculator interface to be used within LSST DESC.

This is currently a placeholder with a handful of functions that may be useful.
See the notebooks in the examples directory.

## Installation

The placeholder TJPCov is now pip installable for convenience, but for development
clone the git repository.


## Planning

Join the #desc-tjpcov channel on the LSST DESC slack to contribute.

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
