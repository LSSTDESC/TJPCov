#!/usr/bin/env python
"""
Covariances for LSST DESC
Copyright (c) 2018 LSST DESC
http://opensource.org/licenses/MIT
"""
from setuptools import setup

setup(
    name='tjocov',
    version='0.0.1',
    description='Covariances for LSST DESC',
    url='https://github.com/LSSTDESC/tjpcov',
    maintainer='Sukhdeep Singh',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    packages=['tjpcov'],
    install_requires=['scipy', 'numpy']
)
