"""
This script contains functions to load data from different fits files and instruments to be used with Acid.
Each function will load a data object that can be directly input into the ACID initialisation.
"""
from __future__ import annotations
from beartype import beartype
from .data import Data
from . import utils
from .utils import Array2D
from astropy.io import fits

@beartype
class Load:

    def __init__(
            self,
            instrument    : str,
            file          : str,
            blaze_file    : str|None     = None,
            blaze_profile : Array2D|None = None
            ):
        """
        Parameters
        ----------
        instrument : str
            The name of the instrument the data was taken with. This will determine how the data is loaded and configured.
            Current options are "HARPS", "HARPS-N", "GEMINI-GHOST", "GEMINI-GRACES"
        file : str
            The path to the fits file containing the data to be loaded.
        blaze_file : str, optional
            The path to the fits file containing the blaze profile for the data to be loaded.
            If not provided, no blaze correction is applied. By default, None
        blaze_profile : Array2D, optional
            Instead load the blaze profile yourself and input it here. This will override the blaze_file input if 
            both are provided. By default, None
        
        
        """

        self.file = file
        self.blaze_file = blaze_file

        instruments = {
            "harps": {
                "function": self.HARPS,
                "accepted_names": ["harps", "harps-s", "harps-south"]
            },
            "harps-n": {
                "function": self.HARPS_N,
                "accepted_names": ["harps-n", "harps-north", "harpsn", "harps_north", "harps_n"]
            },
            "gemini_ghost": {
                "function": self.GEMINI_GHOST,
                "accepted_names": ["gemini_ghost", "ghost", "gemini-ghost"]
            },
            "gemini_graces": {
                "function": self.GEMINI_GRACES,
                "accepted_names": ["gemini_graces", "graces", "gemini-graces"]
            },
        }

        for k, v in instruments.items():
            if instrument.lower() in v["accepted_names"]:
                self.loader = v["function"]
                self.instrument = k
                break
        if not hasattr(self, "loader"):
            raise ValueError(f"Instrument '{instrument}' not recognised. Accepted instruments are: {', '.join(instruments.keys())}.")
        self.instrument = instrument # instrument is now an accepted name

        # Save exceptions now
        self.load_exception = f"Error loading data for the '{self.instrument}' instrument. \n" \
        f"Please check the file format and ensure it follows standard routines for extracting with fits formats. \n" \
        f"If you believe this is a standard format and are still getting this error, \n " \
        f"please open an issue on the ACID GitHub repository with details of the file format and the error message you received." \
        f"Exception: \n"

        self.data = Data()
        # Load data into class
        self.hdul = fits.open(file) # standard astropy error catching will catch if this fails
        self.loader(self.hdul)

        # If blaze profile is provided, load it
        if blaze_profile is not None:
            self.data.blaze = blaze_profile
        elif blaze_file is not None:
            pass # TODO: load blaze file and save to self.data.blaze or tailor blaze files to each instrument

    def HARPS(self):
        # La Silla HARPS spectra
        try:
            hdr = self.hdul[0].header
            data = self.hdul[0].data
        except Exception as e:
            raise ValueError(self.load_exception + str(e))
            
        pass

    def HARPS_N(self):
        # La Palma HARPS-N spectra
        pass

    def GEMINI_GHOST(self):
        # Gemini GHOST spectra
        pass

    def GEMINI_GRACES(self):
        # Gemini GRACES spectra
        pass

    def CARMENES(self):
        # Calar Alto CARMENES spectra
        pass

    def VLT_UVES(self):
        # La Silla UVES spectra
        pass

    def HARPS3(self):
        # La Palma HARPS3 spectra
        pass

    def ESPRESSO(self):
        # La Silla ESPRESSO spectra
        pass

