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

   A floating-point scalar.

.. py:type:: IntLike
   :canonical: int | numpy.integer

   An integer scalar.

.. py:type:: Scalar
   :canonical: FloatLike | IntLike | numpy.ndarray

   A scalar numeric value, including 0D NumPy arrays.

.. py:type:: NumericArray
   :canonical: numpy.typing.NDArray[numpy.number]

   A NumPy numeric array of any dimension.

.. py:type:: Array1D
   :canonical: numpy.ndarray | list[Scalar]

   A one-dimensional numeric array or a list of scalar numeric values.

.. py:type:: Array2D
   :canonical: numpy.ndarray | list[list[Scalar]] | list[Array1D]

   A two-dimensional numeric array or nested numeric lists.

.. py:type:: ArrayAnyD
   :canonical: NumericArray | list

   Any-dimensional numeric array or list-like container.
