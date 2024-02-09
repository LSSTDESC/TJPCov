Installation
------------

Quickstart
^^^^^^^^^^
The easiest and recommended way to install TJPCov is to install it via conda::

    conda install -c conda-forge tjpcov

Alternatively, you may also install TJPCov via PyPi::

    pip install tjpcov

will install TJPCov with minimal dependencies, and::

    pip install 'tjpcov[full]'

will include all dependencies (for details, see Optional dependencies (PyPi only) section).

Supported Python Versions
^^^^^^^^^^^^^^^^^^^^^^^^^
TJPCov currently runs on python 3.8, but python 3.9, 3.10, and 3.11 are supported.

TJPCov also has a few specific software versions hardcoded. Please check the ``pyproject.toml`` file to see version requirements.

Optional dependencies (PyPi only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Because TJPCov relies on some packages that may not be necessary for every user, we have added different installation options to accommodate different use cases. For example, if a user has no plans to use MPI with TJPCov, they do not need ``mpi4py``. Below we list the different installation options available on PyPi.

- ``pip install tjpcov`` will install tjpcov and the minimal dependencies.
- ``pip install tjpcov'[doc]'`` will install tjpcov, the minimal dependencies, and the dependencies needed to build the documentation.
- ``pip install 'tjpcov[nmt]'`` will install tjpcov, the minimal dependencies, and the dependencies needed to use NaMaster.
- ``pip install 'tjpcov[mpi4py]'`` will install the minimal dependencies and the mpi4py library to use MPI parallelization.
- ``pip install 'tjpcov[full]'`` will install tjpcov and all dependencies.
