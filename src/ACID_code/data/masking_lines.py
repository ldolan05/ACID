from __future__ import annotations
import numpy as np
from ..utils import c_kms

class MaskingLines:
    """
    A simple class to expose the telluric lines when called in Config. This will help
    to store telluric lines as a dictionary. With a default itercall to list the line-wise elements,
    but a dictionary index to also store the width of the line, which can then allow for masking Hydrogen
    lines with much wider masks.
    """
    __slots__ = ("lines",) # the only thing stored in this class is this dictionary

    def __init__(self, lines:dict) -> None:
        """
        Sets the lines attribute after validating the input lines dictionary. The format is specified in :py:class:`Acid`.
        """
        self.lines = self.validate_lines(lines)

    def __getitem__(self, key):
        # should work for int or str keys
        return self.lines[key]

    def __iter__(self):
        return iter(self.lines.items())

    def get_masks(self, x, with_names=False) -> list | dict:
        """
        Generates masks for the given input array `x` based on the stored lines and widths.

        Parameters
        ----------
        x : array-like
            The input array for which to generate masks.
        with_names : bool, optional
            Whether to return a dictionary with line names as keys. Useful if plotting. Default is False.

        Returns
        -------
        list | dict
            A list of masks (ie list of 1D mask arrays) or a dictionary of masks keyed by line names.
        """
        mask = [] if not with_names else {}
        for name, line_data in self.lines.items():
            lines = np.asarray(line_data["lines"])
            widths = np.asarray(line_data["widths"])

            limits = 3 + (widths / c_kms) * lines
            conditions = np.abs(x[None, :] - lines[:, None]) <= limits[:, None]
            line_mask = np.any(conditions, axis=0)
            if with_names:
                mask[name] = line_mask
            else:
                mask.append(line_mask)
        return mask
    
    def get_1d_mask_on_grid(self, x:np.ndarray) -> np.ndarray:
        """
        Generates a single 1D mask for the given input array `x` based on the stored lines and widths.

        Parameters
        ----------
        x : array-like
            The input array for which to generate the mask.

        Returns
        -------
        np.ndarray
            A 1D boolean mask array where True indicates that the corresponding element in `x` is within the masking region of any line.
        """
        masks = self.get_masks(x)
        if len(masks) == 0:
            return np.zeros_like(x, dtype=bool)
        combined_mask = np.any(masks, axis=0)
        return combined_mask

    @staticmethod
    def validate_lines(input_lines:dict|MaskingLines) -> dict:
        """
        Standard method to validate linelist input, the format is quite flexible for convenience, but the output is always a standardised dictionary.
        See :ref:`masking_lines`
        """

        # Skip validation if MaskingLines object is input, as it would have already been validated
        if isinstance(input_lines, MaskingLines):
            return input_lines.lines

        # Set error messages for common errors to avoid repetition
        length_mismatch_error = f"The number of lines and inputted widths must be the same if inputting widths.\n" \
        f"If you only wish to input the widths of certain lines, use a list of tuples, see :ref:`masking_lines` for more details."
        default_width_error = "No default width was provided for the masking_lines of {}, see :ref:`masking_lines` for more details."

        # Set variables to be updated within the loop
        final_dict = {}

        for name, line_object in input_lines.items():
        
            default_width = None

            # Allow first dict inputs, convert them first to a array format to be validated like any other array input
            if isinstance(line_object, dict):
                if "default_width" in line_object:
                    default_width = line_object["default_width"]
                if "lines" not in line_object:
                    raise ValueError(f"If the value for {name} is a dictionary, it must contain a 'lines' key with the list/array of lines to mask")
                if "widths" in line_object:
                    line_input = [(l, w) for l, w in zip(line_object["lines"], line_object["widths"])]
                else:
                    line_input = line_object["lines"]
            else:
                line_input = line_object

            if isinstance(line_input, (np.ndarray, list)):
                # Reject empty lists or arrays, as this is likely a user error
                if len(line_input) == 0:
                    raise ValueError(f"The masking_lines for {name} cannot be an empty list or array, use None/remove the input to use the default lines.")

                # For lists of tuples, allow len 1 or 2 depending on if default_width was provided in the dictionary
                if isinstance(line_input[0], tuple):
                    lines = []
                    widths = []
                    for line in line_input:
                        if len(line) == 1:
                            lines.append(line[0])
                            if default_width is None:
                                raise ValueError(default_width_error.format(name))
                            widths.append(default_width)
                        elif len(line) == 2:
                            lines.append(line[0])
                            widths.append(line[1])
                        else:
                            raise ValueError(f"If the masking_lines for {name} is a list or array of tuples, each tuple must have length 1 " \
                            f"(line only) or 2 (line and width). \nGot tuple with length {len(line)}")          

                else:
                    # For arrays or lists, convert to numpy array and check dimensions
                    try:
                        lines = np.array(line_input)
                    except Exception as e:
                        raise ValueError(f"Could not convert the masking_lines for {name} to a numpy array. \n"
                                         f"It's possible the dimensions do not have the same shape. Please check the input format. \nError: {e}")
                    if lines.size == 0:
                        raise ValueError("lines cannot be an empty array or list, use None/remove the input to use the default lines.")                
                    if lines.ndim == 1:
                        if default_width is None:
                            raise ValueError(default_width_error.format(name))
                        widths = [default_width for _ in lines]
                    elif lines.ndim == 2:
                        widths = lines[1]
                        lines = lines[0]
                        if len(lines) != len(widths):
                            raise ValueError(length_mismatch_error + f"\nGot {len(lines)} lines and {len(widths)} widths.")
                    else:
                        raise ValueError("lines must be a one- or two-dimensional array or list")

            else:
                raise ValueError(f"The masking line for {name} does not conform to the accepted formats, see :ref:`masking_lines`"
                                 f" for more details. Got type {type(line_input)}.")

            if len(lines) != len(widths):
                raise ValueError(f"lines and widths should be of same length, got: {len(lines)}, {len(widths)}")
            final_dict[name] = {"lines": np.array(lines), "widths": np.array(widths)}
        return final_dict
