Contributing
------------

Planning & development
^^^^^^^^^^^^^^^^^^^^^^

Ask @felipeaoli or @mattkwiecien for access to the repository and join the #desc-mcp-cov channel on the LSST DESC slack to contribute.

We have adopted the following style convention (which are enforced in each PR):

- `Google-style docstrings <https://google.github.io/styleguide/pyguide.html>`_
- `Black code style <https://github.com/psf/black>`_ (with 79 characters line-width)
- PEP8 except for E203 (for better compatibility with black)

For a general idea of TJPCov's scientific scope, see also the :ref:`Terms of Reference`.

Code Quality Tools
^^^^^^^^^^^^^^^^^^

We use ``black`` and ``flake8`` configuration files so that code follows a unified coding style and remains PEP8 compliant.

This means before submitting your PR you must run the following in the root directory::

    black .
    flake8 .

Furthermore, we are following GitHub's recommendation of using Semantic Versioning <https://semver.org/> in our releases.


