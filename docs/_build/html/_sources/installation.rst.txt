.. _installation:

Installation
--------------

Installing the package
=======================

ACID is currently compatable with python=3.7 due to issues arising from the multiprocessing module in newer versions of python (hopefully to be remedied in upcoming releases!). 
It is therefore recommended to create a conda environment to avoid downgrading your local python installation. 

In this example we create an environment named 'acid' and activate it using:

.. code-block:: bash

    conda create -n acid python=3.7
    conda acivate acid

Once the environment has been activated ACID can be installed using pip_:

.. _pip: https://pip.pypa.io/en/stable/ 

.. code-block:: bash

    pip install ACID_code

.. _source:

Installing from the source
===========================
In order to use the example data (for the tutorials_) or the test suite ACID will need to be installed from the source (i.e. directly from the GitHub repository).
This can be done by cloning the source repository and installing from there.

.. _tutorials: file:///Users/lucydolan/Documents/GitHub/ACID/docs/_build/html/using_ACID.html

.. code-block:: bash

    git clone https://github.com/ldolan05/ACID.git
    cd ACID_code
    python -m pip install -e .

.. _test:

Testing the installation
==========================

Test your installation by running our test file in the root repository. This should run without any errors if the installation has gone smoothly.

.. code-block:: bash
    
    python easy_test.py


