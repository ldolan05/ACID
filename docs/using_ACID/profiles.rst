.. _profiles:

Analysing the Final Profiles
============================

Note that for ACID profiles are stored such that the continuum sits at 1 and decreases at the profile flux.
However, the Profile class fits profiles at a continuum of 0. If you decide to directly input the flux
values yourself, please ensure the continuum sits at 0 simply by subtracting 1. The profiles are stored this
way because they do not need to then be transformed if we wish to directly dot the linelist with the profiles
(in OD space!), even if the Result class plots them at a continuum of 0.

There is simple class provided to fit the final Acid profiles, called ACID_code.Profiles.
It can be initialised with a Data instance or directly withe the profile and velcoty values.
If a Data instance is provided, only the first frame will be used.

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

