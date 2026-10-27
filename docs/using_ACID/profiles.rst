.. _profiles:

Analysing the Final Profiles
============================

The Profiles class fits absorption profiles normalized to a continuum of 1,
as returned by ACID.

There is simple class provided to fit the final Acid profiles, called :py:class:`ACID_code.Profiles`.
It can be initialised with a Data instance or directly with the profile and velocity values.
If a Data instance is provided, the combined profile at ``key="final"`` is used by default.

From a Result, use ``result.profiles.fit_gaussian()`` or
``result.profiles.plot_fit("gaussian")``. Each access to ``result.profiles`` creates
a fresh Profiles instance. To reuse fits, keep a local reference with
``profiles = result.profiles``. Raw ACID profile plots remain available through
``result.plot_profiles()``; on a standalone Plot instance use ``plots.profiles()``.
For a DataList, ``datalist.profiles`` similarly creates a fitter for the combined
profile across orders.

To save time we load a saved data instance, with the profiles already calculated previously

.. code-block:: python

   from astropy.io import fits
   import ACID_code as acid

    # Load a data instance
    data = acid.Data.load("data.pkl")

    # Start the Profiles instance
    profiles = acid.Profiles(data=data)

    # Alternatively directly input your values
    profiles = acid.Profiles(data.velocities, data.profiles[0,0], data.profiles[0,1])

    # Plot your fit using a voigt profile
    profiles.plot_fit("voigt")

    # Or compare with all possible fits
    profiles.plot_fit("all") # Plots all of the Gaussian, Lorentzian and Voigt fits

Multiple profile groups
-----------------------

See :ref:`profile_groups` to assign groups and names, or generate groups by depth
or species. Names stored on Data are used automatically in fit plots; direct
array inputs can pass ``profile_names=["Iron", "Titanium"]``.

For a multi-profile run, ``Profiles(data=data)`` automatically uses all profile
groups from the selected key. You can also supply flux with shape
``(n_profiles, n_velocities)`` directly. Velocities must be a shared 1D grid,
or a 2D array repeating that same grid. These rows represent profile groups
from the same data, rather than separate frames.

.. code-block:: python

    profiles = acid.Profiles(data=data)
    popt, pcov = profiles.fit_gaussian()
    first_parameters = popt[0]
    first_covariance = pcov[0]
    profiles.plot_fit("gaussian")

All three fitting methods fit each group independently. ``popt`` has shape
``(n_profiles, n_parameters)`` and ``pcov`` has shape
``(n_profiles, n_parameters, n_parameters)``. An initial guess ``p0`` can be
shared or supplied as one row per profile. Invalid pixels are removed separately
for each profile; ``on_exception="warn"`` or ``"pass"`` returns NaNs for failed rows.

Errors can be shared (1D) or supplied per profile (2D). Covariance matrices can
be shared, stacked per profile, or supplied as the flattened covariance returned
by ACID. For the latter, each fit uses its own diagonal block; cross-group
covariances are not used in these independent fits.

Multiple profiles are plotted together with their residuals, using one selected
model: ``"gaussian"``, ``"voigt"``, or ``"lorentzian"``. ``"all"`` is only available
for a single profile. ``None`` or ``"none"`` reuses existing fits, provided at most
one model has been fitted when plotting multiple profiles.

