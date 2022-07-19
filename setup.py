#!/usr/bin/env python
"""
Covariances for LSST DESC
Copyright (c) 2021 LSST DESC
http://opensource.org/licenses/MIT
"""
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='tjpcov',
    version='0.0.1',
    description='Covariances for LSST DESC',
    url='https://github.com/LSSTDESC/tjpcov',
    author='Sukhdeep Singh, Felipe Andrade Oliveira',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
    packages=['tjpcov'],
    install_requires=['scipy', 'numpy', 'Jinja2', 'pyyaml','pytest','pyccl','sacc'],
    python_requires='>=3.7',
)
