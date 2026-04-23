.. _type_validation:

Type Validations
================

ACID uses the beartype package to simplify input validation, avoiding the need to code these errors myself.

Inputs that do not conform to those specified in the API and type hints will raise a BeartypeError. The error will describe the expected and received types. 
Please refer to the beartype_ documentation for more information on how to interpret these errors.

.. _beartype: https://beartype.readthedocs.io/en/stable/

We use aliases for some common input types, such as 1D arrays which can either be input as lists of numpy arrays. These can be found in :ref:`aliases`, where the meaning
of each alias is described in more detail.

It is also possible that a valid input may raise a BeartypeError, if you believe this is the case, then please consider raising an issue in the GitHub repository with the details.