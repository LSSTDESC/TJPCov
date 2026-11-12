Example: Basic covariance computation
-------------------------------------

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   $ python run_tjpcov.py config.yml

This will create a :code:`summary_statistics.sacc` file containing the final
covariance in the output directory defined in the configuration file (i.e.
:code:`tests/tmp/`). You can use the :code:`-o` or :code:`--output` arguments
to use a different file name (the path will be the same).

2. Interactively
^^^^^^^^^^^^^^^^

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

Example: C_ell x cluster covariance
-----------------------------------

The covariance between angular power spectra (cosmic shear, galaxy-galaxy
lensing and galaxy clustering) and cluster counts and mean log mass is
computed by :code:`ClusterClcNGHaloModel` (connected term, from the squeezed
halo model bispectrum) and :code:`ClusterClSSCHaloModel` (super-sample term).
They need a sacc file with both, C_ell and cluster data, and a CCL version
with :code:`pyccl.angular_cl_cov_counts`. Add them to :code:`cov_type`
together with the C_ell x C_ell and cluster x cluster covariances, e.g.

.. code-block:: yaml

   tjpcov:
     cov_type: [FourierGaussianFsky, FourierSSCHaloModelFsky,
                FouriercNGHaloModelFsky, ClusterCountsGaussian,
                ClusterCountsSSC, ClusterMass, ClusterClcNGHaloModel,
                ClusterClSSCHaloModel]

The clusters are described with the :code:`mor_parameters` and
:code:`photo-z` sections of the cluster counts covariance; lensing tracers
with an NFW profile and galaxies with the :code:`HOD` section (which sets
their bias). Galaxy clustering autocorrelations include the one-galaxy
(shot noise) term. The super-sample term projects the long mode with the
redshift bin windows of :code:`ClusterCountsSSC`. By default the footprint
is a spherical cap of the survey area. If :code:`tjpcov: mask_file` has a
mask for the survey tracer of the clusters, the cluster SSC terms use the
mask power spectrum instead, (2l+1) C_l / Omega^2 as the C_ell SSC with
masks, and the C_ell x cluster SSC uses its cross power spectrum with the
product of the masks of the C_ell tracers. If the C_ell in the sacc
file have bandpower windows, the covariance is that of the bandpowers,
C_b = sum_ell W_b(ell) C_ell, which includes the mode coupling of the mask;
otherwise it is evaluated at the effective ell. Optional settings:

.. code-block:: yaml

   ClusterCl:
     nz: 48    # redshift nodes per cluster bin for the bispectrum splines
     na: 800   # shells for the projection of the long mode (SSC)
     nell: 100 # multipoles interpolated to those of the bandpower windows
   ClusterCountsSSC:
     L: 20     # multipoles of the long mode (default 20 for a cap, 100 for a mask)
   cNG:
     nk: 256  # k nodes of the C_ell x C_ell trispectrum (default: P(k) spline)

The cluster selection of all cluster covariances can include the
completeness and purity models of crow (as crow, the counts are multiplied
by the completeness c(M, z) and divided by the purity p(richness, z), and the
false detections are assumed to be clustered as the true clusters).
Parameters not given take the crow defaults:

.. code-block:: yaml

   completeness:
     model: Aguena16          # crow CompletenessAguena16
     a_logm_piv: 13.31
   purity:
     model: Aguena16LnProxy   # crow PurityAguena16 or PurityAguena16LnProxy

See :code:`examples/shear_and_cluster_covariance.ipynb` for a full example.
