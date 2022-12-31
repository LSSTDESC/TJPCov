************
Installation
************
TJPCov can be installed with `pip`. There are five different flavors of TJPCov at the moment:
 - :code:`python -m pip install .` will install tjpcov and the minimal dependencies.
 - :code:`python -m pip install .\[doc\]` will install tjpcov, the minimal dependencies and the dependencies needed to build the documentation.
 - :code:`python -m pip install .\[nmt\]` will install tjpcov, the minimal dependencies and the dependencies needed to use NaMaster.
 - :code:`python -m pip install .\[mpi4py\]` will install, the minimal dependencies and the mpi4py library to use MPI parallelization.
 - :code:`python -m pip install .\[full\]` will install tjpcov and all dependencies

Note that due to a bug in the NaMaster installation, one needs to make sure
numpy is installed before trying to install NaMaster. If you are doing a fresh
install, run :code:`python -m pip install .` first, and then :code:`python -m pip install .\[nmt\]`
