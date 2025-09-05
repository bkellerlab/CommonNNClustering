Installation instructions
=========================

Requirements
------------

The :mod:`commonnn` package is developed and tested in Python >=3.6.
At runtime the package has a few mandatory third party dependencies.
We recommend to install the latest version of:

   * :mod:`loguru`
   * :mod:`numpy`

Optionally, additional functionality is available when the following
packages are installed as well:

   * :mod:`matplotlib`
   * :mod:`pandas`
   * :mod:`networkx`
   * :mod:`scipy`
   * :mod:`sklearn`

PyPi
----

The :mod:`commonnn` package is available on the Python package index.

.. code-block:: bash

   pip install commonnn-clustering

To install with optional dependencies use:

.. code-block:: bash

   pip install commonnn-clustering[optional]

Other extras can be specified with `[dev]`, `[docs]`, and `[test]`.

To make use of parallel implementations, enable OpenMP support prior to installation:

.. code-block:: bash

   $ export CFLAGS="-fopenmp $CFLAGS"
   $ export LDFLAGS="-fopenmp $LDFLAGS"

Development installation
------------------------

Clone the source repository `https://github.com/bkellerlab/CommonNNClustering
<https://github.com/bkellerlab/CommonNNClustering>`_ and use the package
:mod:`commonnn` as you prefer it, e.g. install it via `pip` in editable mode.

.. code-block:: bash

   $ git clone https://github.com/bkellerlab/CommonNNClustering
   $ cd CommonNNClustering
   $ pip install -e .

To recompile the Cython-extensions (requires :mod:`cython` installed) use:

.. code-block:: bash

   $ python setup.py build_ext --inplace

We provide a `env_dev.yml` file to create a conda environment with all development dependencies
before installing the package itself with `pip`:

.. code-block:: bash

   $ conda env create -f env_dev.yml
   $ conda activate commonnn-dev

Testing and documentation
-------------------------

Tests can be found under the `test/` directory and can be run using :mod:`pytest`.
Make sure you have all dependencies installed. To enable code coverage reports,
the package needs to be compiled with `TRACE_CYTHON=1`. Note that this slows
down the runtime of the package routines significantly and should be set
to `TRACE_CYTHON=0` in production. To run the tests, create a coverage report
and a corresponding badge, one can use the script `test.py`.

You can rebuild tests based on Cython functions with `python setup_tests.py build_ext --inplace`
(`PYTHONPATH` may need to include the `commonnn/src` directory).
