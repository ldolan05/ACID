.. _masking_lines:

Using your own line masks
=========================

ACID has a default set of lines that it uses for masking. The default set targets wide hydrogen lines and strong metal lines which cause a poor fit near those wavelengths.
The default in ACID includes the following lines from testing on a few different star types and instruments:

- Wide lines (default width of 2000km/s):
    - The hydrogen lines from H-alpha to H-eta
- Medium lines (default width of 1000km/s):
    - Na D1 and D2
    - Ca II H and K,
- Narrow lines (default width of 200km/s):
    - Mg I b1, b2, b3
    - Fe I 3820.33
    - Fe I 5270.40
    - A few more telluric lines
    - A few more metal lines

You can use the default lines, or you can specify your own lines to mask. You can also specify the widths of the lines to mask, or just modify the default widths for each group. 
You can specify different sets of lines with different widths, and you can specify as many sets as you like. The key names for the dictionary (ie "wide, narrow") are arbitrary and are 
only used for plotting.

Below we detail the different types of formats for the inputs to be passed as "masking_lines" into :py:class:`ACID_code.Acid`.

Formats
-------

The format for the masking_lines input is a dictionary with keys as the names of the line groups (eg "wide", "narrow") and values as any of the following:

- A dictionary with keys "lines" (required), and "default_width" (optional), where "default_width" is the default width to use for all lines in that group. 
Any lines without widths will be filled with the default width for that group, and if widths are missing for some lines, an error will be raised. 
The "lines" key can be any of the below formats.
- A one-dimensional array or list of line wavelengths, which will be filled with the default width for that group.
- A two-dimensional array or list of line wavelengths and widths, where the first column is the wavelengths and the second column is the widths.
- A list of tuples, where each tuple contains a wavelength and optional width.

Examples
--------

.. code-block:: python

    # Example 1: Using the default lines and widths
    acid = ACID_code.Acid(masking_lines=None)

    # Example 2: Specifying custom lines with default widths
    custom_lines = {
        "wide": [3835.38, 3889.05, 4101.74, 4340.47, 4861.34, 6562.81],
        "medium": [5889.95, 5895.92, 3933.66, 3968.47],
        "narrow": [5172.68, 5183.62]
    }
    acid = ACID_code.Acid(masking_lines=custom_lines)

    # Example 3: Specifying custom lines with custom widths
    custom_lines_with_widths = {
        "wide": {
            "default_width": 2000,
            "lines": [
                (3835.38, 2500), # H eta
                (3889.05, 2500), # H zeta
                (4101.74, 2500), # H delta
                (4340.47, 2500), # H gamma
                (4861.34, 3000), # H beta
                (6562.81, 3000)  # H alpha
            ]
        },
        "medium": {
            "default_width": 1000,
            "lines": [
                (5889.95, 1200), # Na D1
                (5895.92, 1200), # Na D2
                (3933.66, 1500), # Ca II K
                (3968.47, 1500)  # Ca II H
            ]
        },
        "narrow": {
            "default_width": 200,
            "lines": [
                (5172.68, 300), # Mg I b1 triplet
                (5183.62, 300)  # Mg I b2 triplet
            ]
        }
    }
    acid = ACID_code.Acid(masking_lines=custom_lines_with_widths)

    # Example 4: Just modifying the defaults using pythonic dictionary/list manipulation
    defaults = ACID_code.Config.defaults["masking_lines"]
    defaults["wide"]["default_width"] = 2500 # modify default width
    defaults["medium"]["lines"] = [5889.95, 5895.92] # modify lines to mask in medium group
    defaults.pop("narrow") # remove narrow group entirely
    defaults["new_group"] = {
        "default_width": 500,
        "lines": [3300.0, 3500.0] # add a new group of lines to mask
    }
    defaults["wide"]["lines"].append(4102.0) # add a line to the wide group
    defaults["medium"]["lines"].pop(0) # remove the first line from the medium group
    acid = ACID_code.Acid(masking_lines=defaults)
