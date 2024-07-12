Developer Installation
----------------------

Using conda (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^
If you wish to contribute to TJPCov, follow the steps below to set up your development environment.

1. Clone the repository
2. Create the conda environment with ``conda env create --file environment.yml``
3. Activate the environment with ``conda activate tjpcov``
4. Run ``pip install -e .``
5. Run ``pytest -vv tests/``

Using pip
^^^^^^^^^
1. Clone the repository
2. Run ``pip install -e .``
3. Run ``pytest -vv tests/``

.. warning::
    NaMaster installation (pip only).  
    
    If you are using PyPi to set up your development environment (we recommend using conda instead), due to a bug in the NaMaster installation, one needs to make sure numpy is installed before trying to install NaMaster. 
    
    For a fresh install, run ``pip install .`` first, and then ``pip install .[nmt]``
