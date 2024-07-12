.. _Terms of Reference:

Terms of Reference
------------------

Background
^^^^^^^^^^

- This document lays out terms of reference for TJPCov developers and contributors in the wider MCP and LSST DESC context.

Scope
^^^^^

- TJPCov provides a set of APIs for calculation of covariance (and cross-covariance) matrices (or other descriptions of measurement uncertainty as required) for all main canonical large scale structure cosmological probes in DESC:

  - weak lensing shear
  - galaxy clustering
  - galaxy clusters clustering
  - galaxy clusters number counts

- It provides flexibility in how its functionality is accessed:

  - it is called from TXPipe and APIs are designed to work with TXPipe
  - it can be called as a standalone library
  - it can be called from the command line through driver routines

- Interface support is as follows:

  - Public APIs are maintained in Python
  - Documentation should be maintained in the main README, readthedocs, and the benchmarks folder.
  - Command-line driver routines are maintained in the bin folder
  - Examples will be maintained in a separate repository. [AS?]

- In the basic model, covariance matrices are factorised as follows:

  - Disconnected part of the Gaussian covariance matrix
  - Connected part of the Gaussian covariance matrix arising from mode coupling
  - Additional covariance due to super-survey modes

- External libraries:

  - TJPCov aims to implement basic functionality internally but supports an external back-end for versatility and cross-checking
  - External back-ends can be linked as external libraries (on top of a thin wrapper inside TJPCov), but must be interfaced via pure python (i.e., the external back-end must implement a python wrapper as a precondition to be integrated into TJPCov)

- Boundaries of TJPCov:

  - TJPCov does not support the calculation of covariance matrices that are not covariant with the main large-scale structure probes (i.e., supernovae luminosity distance)
  - TJPCov performs theoretical computations of the covariance matrix given survey properties and assumed cosmology. It does not perform calculations on the data, e.g., various bootstrap techniques
  - In general, TJPCov does not support covariance arising from systematic effects although exceptions could be made on an as-needed basis (i.e., we provide a covariance matrix as observed by a perfect instrument with no galactic foregrounds, etc.)
