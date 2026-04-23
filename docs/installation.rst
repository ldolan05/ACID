.. _installation:

Installation and Setup
======================

Installing the package
------------------------

ACID has been tested in Python 3.13 and 3.14, running fastest on 3.14. It is recommended to install ACID in a new conda environment to avoid dependency conflicts.

In this example we create an environment named 'acid' and activate it using:

.. code-block:: bash

    conda create -n acid python=3.14
    conda activate acid

Once the environment has been activated ACID can be installed using pip_:

.. _pip: https://pip.pypa.io/en/stable/ 

.. code-block:: bash

    pip install ACID_code

.. _source:

This will install ACID into your environment with all of its dependencies.

.. _cloning:

Cloning the repository
------------------------
In order to use the example data (for the tutorials) or the test suite, ACID will need to be installed from the source (i.e. directly from the GitHub repository).
This can be done by cloning the source repository. All examples and tests attempt first to import from your pip installation.
If this fails they will attempt to import from the local source directory instead.

.. code-block:: bash

    git clone https://github.com/ldolan05/ACID.git
    cd ACID

.. _test:

Testing the installation
-------------------------

Test your installation by running our test file in the test directory. This may take a while (~1-2 min) but should run without any errors if the installation has gone smoothly.
The test file will attempt to run all of the methods and functions in ACID. If any of these fail and you believe that this is due to the source code and not your installation, please raise an issue on GitHub.

.. code-block:: bash
    
    conda install pytest
    python tests/tests.py
