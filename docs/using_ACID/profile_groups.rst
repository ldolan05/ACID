.. _profile_groups:

Multiple Profile Groups
=======================

ACID can solve a separate profile for each group of lines in the same spectrum.
These groups are independent of input frames: each frame can have multiple profiles.
Group settings can be passed to ``Acid``, ``Acid.ACID``, ``Config`` or ``DataList``.

Assigning groups and names
--------------------------

Pass one non-negative integer group ID per line through ``profile_groups``.
The array must match either the full stored line list, in wavelength order,
or the line list remaining after wavelength and S/N clipping. IDs need not be
consecutive. Output rows follow the sorted IDs of the surviving groups.

.. code-block:: python

    import ACID_code as acid

    runner = acid.Acid(velocities=velocities, linelist=linelist)
    # Example for a line list containing four lines
    groups = [0, 1, 0, 1]
    result = runner.ACID(
        wavelength, flux, errors, sn,
        profile_groups=groups,
        profile_names=["Weak lines", "Strong lines"],
    )
    result.plot_profiles()
    result.profiles.plot_fit("gaussian")

``profile_names`` accepts a list of non-empty strings, indexed by consecutive
group IDs starting at 0, or a dictionary mapping actual group IDs to strings.
For IDs 2 and 7, for example, use ``{2: "Iron", 7: "Titanium"}``, rather than
a two-element list. Names must cover exactly the input group IDs. Invalid types,
missing names, nonexistent IDs and mismatched list lengths raise an error.
If clipping removes an entire group, its name is removed from the resolved mapping.

The supplied settings remain on Config. The clipped groups and resolved names
are available as ``result.data.profile_groups`` and ``result.data.profile_names``.
Names appear in profile plots, analytic fit plots, grouped line-list legends in
forward-model plots, and DataList profile and RV plots. Without names, plots use
their default group labels. See :ref:`result` for indexing flux, errors and covariance.

Grouping by line depth
----------------------

``depth_group_rules`` generates groups after wavelength and S/N clipping, using
linear absorption depths from the line list:

.. code-block:: python

    result = runner.ACID(
        wavelength, flux, errors, sn,
        depth_group_rules={"n_groups": 3, "min_lines": 20, "0": 0.8},
    )

Group 0 takes the deepest lines down to depth 0.8, extending shallower if needed
to reach ``min_lines``. Unspecified groups split the remaining lines equally.
Explicit depth rules must start at group 0, be consecutive, and specify decreasing
minimum depths. Insufficient lines raise an error. With only ``n_groups`` and
``min_lines``, all groups have approximately equal line counts, deepest first.

Each group gets a name such as ``Depth 0.4–0.8``, describing the actual depth
range assigned to it. Supply ``profile_names`` to override those automatic names.

Grouping by species
-------------------

Use ``seperate_species`` (with this spelling) to give selected species their
own profiles:

.. code-block:: python

    result = runner.ACID(
        wavelength, flux, errors, sn,
        seperate_species=["Fe 1", "Ti 1"],
    )

Identifiers must exactly match the line list's ``spec_ion`` strings, including
ionization state and spaces. Enabling this option uses ``LineList(full=True)``,
reading both species and Lande factors from VALD files. In-memory line lists
must include both ``spec_ion`` and ``lande_factor`` columns; an existing LineList
must have retained that full metadata.

Group 0 is named ``Everything else``. Surviving species receive consecutive IDs
1, 2, ... in input order and are named with their species identifiers. Each group
requires at least 20 lines **after wavelength and S/N clipping**. A requested
species with fewer lines, including an unknown species with zero matches, emits
``ACIDProfileGroupsWarning`` and remains in group 0. Its lines are retained in the fit.
If the remainder has fewer than 20 lines, the last retained species is returned
to group 0 with a warning, repeating if necessary. If fewer than 20 lines exist
in total, only group 0 remains and a warning reports that limitation.

Promote this warning to an exception when the requested grouping is essential:

.. code-block:: python

    import warnings
    from ACID_code import ACIDProfileGroupsWarning

    warnings.filterwarnings("error", category=ACIDProfileGroupsWarning)

Explicit ``profile_groups`` take precedence over automatic grouping. Otherwise,
choose either ``depth_group_rules`` or ``seperate_species``; using both raises
an error. Custom names must match the surviving automatic groups. For joint
orders, shared lines must have the same group assignments and names in each
order; wavelength and S/N cuts still happen separately per order.

Fitting named profiles directly
--------------------------------

``result.profiles`` passes resolved names to the Profiles class automatically.
For direct arrays, pass names indexed by row:

.. code-block:: python

    fitter = acid.Profiles(
        velocities, profile_flux, profile_errors,
        profile_names=["Iron", "Titanium"],
    )
    popt, pcov = fitter.fit_gaussian()
    fitter.plot_fit("gaussian")

The fit output keeps its numeric row order. Names affect labels, not indexing.
See :ref:`profiles` for the returned shapes and multiple-profile plotting rules.
