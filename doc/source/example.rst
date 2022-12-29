Example: Basic covariance computation
=====================================

Users should use the :code:`tjpcov.covariance_calculator.CovarianceCalculator`
class with a configuration file or dictionary input.

An example configuration file could be the following. In this case, one will
compute the Gaussian covariance with NaMaster and the SSC covariance. The
configuration specific for each covariance is defined in their own sections.
Extra sections (e.g. :code:`GaussianFsky`) are ignored.

.. literalinclude:: ../../tests/data/conf_covariance_calculator.yml
   :language: yaml

In order to generate the covariance there are two options:

1. Using the :code:`run_tjpcov.py` script.
------------------------------------------

.. code-block:: bash

   $ python run_tjpcov.py config.yml

This will create a :code:`summary_statistics.sacc` file containing the final
covariance in the output directory defined in the configuration file (i.e.
:code:`tests/tmp/`). You can use the :code:`-o` or :code:`--output` arguments
to use a different file name (the path will be the same).

2. Interactively
----------------

If you just want to get the final covariance:

.. code-block:: python

   from tjpcov.covariance_calculator import CovarianceCalculator
   cc = CovarianceCalculator(config_yml)
   cov = cc.get_covariance()

If you just want to get the sacc file with the covariance (this will also save
a copy of the sacc file): 

.. code-block:: python

   from tjpcov.covariance_calculator import CovarianceCalculator
   cc = CovarianceCalculator(config_yml)
   s = cc.create_sacc_cov()
