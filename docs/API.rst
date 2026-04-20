.. _API:

ACID API
========

.. currentmodule:: ACID_code

Core Classes
-------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   Acid
   Result
   Data
   Config
   DataList
   LSD
   Profiles
   MCMC
   LineList
   MaskingLines

Utility Functions
------------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   calc_deltav
   utils

Legacy ACID Functions
----------------------

These functions are retained for backwards compatibility with ACID v1.

.. autosummary::
   :toctree: _api
   :nosignatures:

   ACID
   ACID_HARPS


Type Aliases
----------------

.. py:type:: FloatLike
   :canonical: float | numpy.floating

   Any pythonic or numpy floating-point scalar.

.. py:type:: IntLike
   :canonical: int | numpy.integer

   Any pythonic or numpy integer scalar.

.. py:type:: Scalar
   :canonical: FloatLike | IntLike | numpy.ndarray

   A scalar numeric value, including 0D NumPy arrays. The 0D condition is enforced.

.. py:type:: Array1D
   :canonical: numpy.ndarray | list[Scalar]

   A one-dimensional numeric array or a list of scalar numeric values. The 1D condition is enforced for all array types.

.. py:type:: Array2D
   :canonical: numpy.ndarray | list[list[Scalar]] | list[Array1D]

   A two-dimensional numeric array or nested numeric lists. The 2D condition is enforced for all array types.

.. py:type:: Array3D
   :canonical: numpy.ndarray | list[list[list[Scalar]]] | list[list[Array1D]] | list[Array2D]

   A three-dimensional numeric array or nested numeric lists. The 3D condition is enforced for all array types.
