from __future__ import annotations
from beartype import beartype
from ..utils import Array1D, Array2D, Array3D, IntLike
from .config import Config
from .data import Data
from .line_list import LineList
import os, pickle
import traceback as tb
import numpy as np
import matplotlib.pyplot as plt
from ..errors import *
from .. import utils
from tqdm import tqdm
import matplotlib as mpl

@beartype
class DataList:
    """
    A class that stores Data instances in a list indexed by order. The DataList is a useful class for running ACID over multiple orders with parallelization. 
    Fundamentally this class holds Data instances (which ACID updates with the results per order) as a list and can map the true order number
    from the instrument (stored in the config) to the index of the list. It handles missing/incomplete orders, and the ability to append new orders.
    For more information and a full example on how to use the DataList, see :ref:`datalist' in the documentation. Note that the DataList is not 
    strictly necessary to run ACID over multiple orders, you can handle the multiple instances yourself.

    The DataList class works with a required root directory specified by the user to access to the same data across parallel processes, 
    and also to save intermediate results and figures per order.
    """

    def __init__(
        self,
        wavelengths      : Array3D|Array2D|None           = None,
        flux             : Array3D|Array2D|None           = None,
        errors           : Array3D|Array2D|None           = None,
        sn               : Array2D|Array1D|None           = None,
        velocities       : Array1D|None                   = None,
        linelist         : Array2D|None|str|LineList|dict = None,
        order_range      : Array1D|None                   = None,
        config           : Config|list[Config]|None       = None,
        save_dir         : str|None                       = None,
        overwrite        : bool                           = False,
        verbose          : IntLike|bool|str|None          = None,
        _load                                             = None,
        _data_list       : list[Data]|None                = None,
        **config_kwargs,
        ) -> None:
        """
        Initializes the DataList object. The DataList can be initialized in two ways: either by providing the wavelengths, flux, errors, and sn arrays directly in
        the class initialization (here), or using the :py:classmethod:`DataList.from_datalist` method with a list of Data objects. 
        The former is useful for quickly initializing a DataList from raw data, while the latter is useful 
        for loading a saved DataList or for more fine-grained control over the initialization of each Data object.

        Parameters
        ----------
        wavelengths : :py:type:`Array3D` | :py:type:`Array2D` | None, optional
            A 2D or 3D array of wavelengths for the input spectra.
            If a 2D array is provided, it is assumed to have shape (n_orders, n_pixels).
            If a 3D array is provided, it is assumed to have shape (n_orders, n_frames, n_pixels). Default is None.
            The format for the last 1 or 2 dimensions follows that of the "wavelengths" input in the :py:function:`Acid.ACID` method.
            Sometimes, fits files store their frames in shape (n_frames, n_orders, n_pixels), you can swap the axes with np.swapaxes(wavelengths, 0, 1) 
            to get them in the correct shape. It is also possible to input orders with different numbers of pixels, in which case the wavelengths should be a list
            of 2D arrays/lists.
        flux : :py:type:`Array3D` | :py:type:`Array2D` | None, optional
            A 2D or 3D array of fluxes for the input spectra. Same shape assumptions as wavelengths. Default is None.
        errors : :py:type:`Array3D` | :py:type:`Array2D` | None, optional
            A 2D or 3D array of errors for the input spectra. Same shape assumptions as wavelengths. Default is None.
        sn : :py:type:`Array2D` | :py:type:`Array1D` | None, optional
            A 1D or 2D array of signal-to-noise ratios for the input spectra. If a 1D array is provided, it is assumed to have shape (n_orders,). 
            If a 2D array is provided, it is assumed to have shape (n_orders, n_frames). Default is None. Follows the same logic as the "sn" input in 
            the :py:function:`Acid.ACID` method, for approximating the errors (or vice versa) if one is not provided.
        velocities : :py:type:`Array1D` | None, optional
            The velocity grid to be used for all the orders. This should be a 1D array of velocity values in km/s. Follows the same format as the "velocities" 
            input in the :py:function:`Acid.ACID` method. Default is None.
        linelist : :py:type:`Array1D` | str | :py:class:`LineList` | dict | None, optional
            The linelist to be used for all the orders. This can be provided in the same formats as the "linelist" input in the :py:function:`Acid.ACID` method.
        order_range : :py:type:`Array1D` | None, optional
            A 1D array of order labels corresponding to the orders in the input data.
            The index of this array should match to the order of the index of the first dimension of the wavelengths, flux, errors, and sn arrays.
            For example, if your input data has 3 orders and they correspond to orders 100, 101, and 102 in the instrument, then you should input order_range = [100, 101, 102].
            If not provided, it is assumed to be a pythonic 0-indexed range of the same length as the number of orders in the input data. Default is None.
        config : :py:class:`Config` | list[:py:class:`Config`] | None, optional
            A template Config object for all orders or a list of Config objects per order containing the configuration for the ACID run.
            If inputting a list, the index and length of the list must match the first dimension of the input data arrays and the order_range.
            These take higher priority than any config_kwargs passed in the initialization.
            Setting 'order' will not have any effect as they will be overwritten by the order numbers in the order_range.
            If not provided, default Config values will be used. Default is None.
        save_dir : str | None, optional
            The default directory to save results and figures for each order. A trailing separator is optional.
            If it does not exist, only this final directory is created and its parent must already exist.
            By default the DataList will save data.pkl and sampler.h5 to the directory (named by the order number) to in this directory.
            If the Configs or kwargs passed contain their own save_path or sampler_path (see :py:class:`Acid`), those instead are used.
            If None, no saving will be done, this is however, not recommended. Default is None.
        overwrite : bool, optional
            Whether to overwrite existing with new Data instances when using run_ACID, or to load and use existing Data instance if they exist.
            If True, if a Data instance already exists for an order, it will be overwritten with the new Data instance generated from the ACID run for that order.
            Note, that the saving of this new Data instance only applies when run_ACID is run, otherwise it is just held in memory.
            If False, if a Data instance already exists for an order, it will be loaded and used instead of generating a new Data instance from the ACID run for that order.
            Default is False.
        verbose : int | bool | str | None, optional
            The verbosity level for printing information during the initialization. 
            Follows the same format as the "verbose" input in the :py:class:`Config` class. 
            Default is None.
        _load : Any, optional
            Not yet implemented, do not use. The idea is that you can input a Load object which has its own tools to pull s2d data from common instruments 
            such as ESPRESSO, HARPS, etc. If you want to use this feature, please open an issue or contribute a pull request with the implementation.
        _data_list : list[:py:class:`Data`] | None, optional
            This is an internal argument used for initializing the DataList from a list of Data objects in the :py:classmethod:`DataList.from_datalist` method.
        **config_kwargs : 
            Additional keyword arguments to be passed with low priority to all of the generated Config objects.
            These kwargs will join with but NOT overwrite any existing keys in the input Config object(s).
            Setting 'order' will not have any effect as they will be overwritten by the order numbers in the order_range.
            Inputting kwargs not part of the defaults in the Config class will cause an error.
            If not provided, default Config values will be used.
        """

        # Raise if load was used
        if _load is not None:
            raise NotImplementedError(f"The 'load' argument is not yet implemented. \n"
                                      f"The idea is that you can input a Load object which has its own tools to pull s2d data from common "\
                                      f"instruments such as ESPRESSO, HARPS, etc. \nIf you want to use this feature, please open an issue or "\
                                      f"contribute a pull request with the implementation.")

        # Configure verbosity
        self.verbose = Config(verbose=verbose).verbose

        # All orders should have the same velocity grid and line list
        self.velocities = velocities

        # Configure order_range, creates one if not input from the shape of wavelengths
        self.order_range = order_range # if None, will be set later, otherwise self.from_datalist handles the range from configs     

        # Set class attributes
        self._save_dir = None
        self._data_list = None
        self._combined_profile = None
        self._results = None
        self.overwrite = overwrite
        self.excluded_orders = []
        self.save_dir = save_dir

        if _data_list is not None:
            self.data_list = _data_list # datalist property handles the rest
            return

        # From here, the array inputs must be provided
        if order_range is None:
            order_range = np.arange(len(wavelengths), dtype=np.int32)
        else:
            order_range = np.asarray(order_range, dtype=np.int32)
            if not np.all(np.isfinite(order_range)):
                raise ValueError("order_range must only contain finite values.")
            if len(order_range) != len(wavelengths):
                raise ValueError("The length of the order_range must match the number of frames in the input data.")
            order_range = np.round(order_range).astype(np.int32)
        self.order_range = order_range
        
        # Convert config to dict(s) to be reinitialized in each Data instance
        if isinstance(config, list):
            if len(config) != len(order_range):
                raise ValueError("If inputting a list of Config objects, the length of the list must match the length of the order_range and input arrays.\n" \
                f"len(order_range): {len(order_range)}, len(wavelengths): {len(wavelengths)}, len(config): {len(config)}.")
            config_dict = [cfg.to_dict() for cfg in config]
        else:
            config_dict = config.to_dict() if config is not None else {}

        # --- Create datalist of Data instances for each order ---
        datalist = []
        for idx, order in enumerate(self.order_range):
            data = Data() # create data instance

            # If config_dict is a list, take the dict at the current index, otherwise use the same dict for all orders
            config_dict_input = config_dict[idx] if isinstance(config_dict, list) else config_dict
            data.config = Config(**config_dict_input) # create and set config instance with default config dict

            # Update the config with any kwargs passed in the initialization of the DataList, and with the order number for this Data instance
            data.config.update_lowpri(**config_kwargs)
            data.config.update_hipri(order=order) # order must be overwritten and set last by us to ensure it matches with the order of the input data for this index

            # Set the inputs for this Data instance, taking the idx'th element of the input arrays for this order.
            # If errors or sn are not provided, they will be set to None and approximated in the Data class.
            input_errors = errors[idx] if errors is not None else None
            input_sn = sn[idx] if sn is not None else None
            data.set_inputs(wavelengths[idx], flux[idx], input_errors, input_sn)

            # Set linelist and velocities, which always should be same for all orders
            data.linelist = linelist # sets the linelist for this Data instance, which is shared across all orders in the DataList
            data.velocities = velocities

            if self.save_dir is not None:
                config_dir = os.path.join(self.save_dir, f"order_{order}")
                data.config.update_hipri(dir=config_dir) # set default save path for this order which can be overwritten by user

                # Check if file already exists
                if os.path.exists(data.config.save_path):
                    if self.overwrite:
                        if self.verbose >= 2:
                            print(f"File {data.config.save_path} already exists, but will be overwritten (when using run_ACID) due to setting.")
                    else:
                        if self.verbose >= 1:
                            print(f"File {data.config.save_path} already exists. The data for this order will be loaded from this file.")
                        data = Data.load(data.config.save_path) # load the existing data from the file instead of using the newly initialized data
                else:
                    data.save() # save the newly initialized, but mostly empty data instance to the file for future reference and use

            datalist.append(data) # finally append to the datalist

        self.data_list = datalist # datalist property handles the rest

    @classmethod
    def from_datalist(cls, data_list:list[Data]|Data, save_dir:str|None=None, verbose:IntLike|bool|str|None=None) -> DataList:
        """
        Load a DataList from a list of Data objects. This is useful for loading a saved DataList or for more fine-grained 
        control over the initialization of each Data object. All Data objects should be already properly initialised with linelists, velocities,
        configs and inputs, and the DataList will check for consistency across the list (e.g. all orders should have the same velocity grid, etc.).

        Parameters
        ----------
        data_list : list[:py:class:`Data`] | :py:class:`Data`
            A list of Data objects to initialize the DataList from. If a single Data object is provided, it will be converted to a list with one element.
        save_dir : str | None, optional
            The directory to save intermediate results and figures for each order. If None, no saving will be done. Default is None.
        verbose : int | bool | str | None, optional
            The verbosity level for printing information during the initialization. Follows the same format as the "verbose" input in the 
            :py:class:`Config` class. Default is None.

        Returns
        -------
        :py:class:`DataList`
            A DataList object initialized from the provided list of Data objects.
        """
        if isinstance(data_list, Data):
            data_list = [data_list]

        # Configure verbosity, if None, use highest verbosity in list
        if verbose is None:
            verbose = np.max([data.config.verbose for data in data_list])

        # All configs should have the same order_range so that they are internally aware. We just take the first one to 
        # generate the mapping of order to index in the list. The Load class will configure the correct order range based
        # off extracted fits header info (if provided), otherwise the default is a pythonic 0-indexed order range.
        order_range = data_list[0].config.order_range
        if len(data_list) > 1 and verbose >= 1:
            if not all(np.array_equal(data.config.order_range, order_range) for data in data_list):
                print("Warning: Not all Data instances have the same order_range. Taking the longest order range.")

        # Take the order range with the greatest length, 
        max_order_range_idx = np.argmax([len(data.config.order_range) for data in data_list])
        order_range = data_list[max_order_range_idx].config.order_range

        # Check all velocity grids match, store velocities
        v0 = data_list[0].velocities
        for data in data_list:
            if not np.array_equal(data.velocities, v0):
                raise ValueError("All Data instances must have the same velocity grid.")
        velocities = v0

        return cls(
            _data_list  = data_list, # skips initialisation of the empty datalist in __init__
            save_dir    = save_dir,
            verbose     = verbose,
            order_range = order_range,
            velocities  = velocities,
        )

    def __iter__(self):
        yield from self.data_list

    def __getitem__(self, k):
        """
        Allows for indexing the DataList with the order number, e.g. datalist[order_number] 
        will return the Data instance with that order number. Uses the internal order to index mapping to find 
        the correct index in the data list.

        Parameters
        ----------
        k : int
            The order number to index the DataList with.
        
        Returns
        -------
        :py:class:`Data`
            The Data instance with the specified order number.
        """
        return self.data_list[self.o2i[k]]

    def __str__(self) -> str|None:
        return f"DataList with {len(self.data_list)} Data instances, storing the orders: {self.orders} out of a total order range: {self.order_range}"

    def __call__(self, *args, **kwargs):
        """Runs and returns the results of the :py:function:`DataList.run_ACID` method, which runs ACID on the Data instances in the list for the specified orders.
        
        Parameters
        ----------
        See :py:function:`DataList.run_ACID` for the accepted parameters and their descriptions.
        """
        return self.run_ACID(*args, **kwargs)

    def __len__(self):
        return len(self.data_list)

    def __iter__(self):
        yield from self.data_list

    def append(self, data:Data, overwrite:bool=False, extend:bool=False, force_order:IntLike|None=None) -> None:
        """
        Appends a Data instance to the data list. Note that the order range of the class is kept, 
        if you want to set a new order range, use the set_order_range() method first to change it.

        Parameters
        ----------
        data : :py:class:`Data`
            The Data instance to append to the data list. The order of the Data instance is taken from its config, 
            but can be overridden with the force_order argument.
        overwrite : bool, optional
            If True, will overwrite an existing Data instance with the same order number. Default is False.
        extend : bool, optional
            If True, will extend the order range to include the new order if it is not already present. Default is False.
        force_order : int, optional
            If provided, will set the order of the Data instance to this value, overriding its config. Default is None.
        """
        if force_order is not None:
            data.config.order = force_order
        order = data.config.order
        if order in self.orders and overwrite is False:
            raise ValueError(f"A Data instance with order {order} already exists in the list. " \
            "If you want to overwrite it, set overwrite=True in the append method.")
        if order not in self.order_range:
            if not extend:
                raise ValueError(f"The order of the appended data class does not match the rest of the list. \n" \
                                 f"If you want to extend the order_range to append the new order, set extend=True.")
            else:
                self.order_range = np.append(self.order_range, order).astype(np.int32)
        
        if overwrite and order in self.orders:
            self.data_list[self.o2i[order]] = data
        else:
            self.data_list.append(data)

        self.sort_by_order() # re-sorts the list and updates the o2i mapping

    def set_order_range(self, order_range:Array1D) -> None:
        """Sets the order range for the DataList. The new range must be a superset of the already saved orders in the list, 
        otherwise a ValueError is raised.
        
        Parameters
        ----------
        order_range : :py:type:`Array1D`
            The new order range to set for the DataList. This should be a 1D array of order numbers. 
        """
        if np.any([o not in order_range for o in self.orders]):
            raise ValueError("The already saved orders must be a subset of the inputted order_range.")
        self.order_range = np.array(order_range, dtype=np.int32)
        self.sort_by_order() # re-sorts the list and updates the o2i mapping, and injects the new order_range into each config

    def sort_by_order(self) -> None:
        """
        Sorts the data list by order number, and updates the o2i mapping accordingly. Internally called whenever self.data_list is updated.
        """
        self.data_list.sort(key=lambda data: data.config.order)
        self.o2i = {data.config.order: i for i, data in enumerate(self.data_list)}
        self.i2o = {i: data.config.order for i, data in enumerate(self.data_list)}
        self.orders = np.array([data.config.order for data in self.data_list], dtype=np.int32)
        for data in self.data_list:
            data.config.order_range = self.order_range # inject the order range into each config for internal awareness

        if len(np.unique(self.orders)) != len(self.orders):
            raise ValueError("All Data instances within the inputted list must have unique order numbers.")

    def run_ACID(
        self,
        orders            : Array1D|int|str|None = None,
        use_index_mapping : bool                 = True,
        worker            : IntLike|None         = None,
        nworkers          : IntLike|None         = None,
        overwrite         : bool|None            = None,
        overwrite_kwargs  : bool                 = False,
        pack              : bool                 = False,
        **kwargs,
        ) -> None:
        """
        Runs ACID on the Data instances in the data list for the specified orders. The results are saved in the save_dir if it is not None, 
        with one pickle file per order containing the Result object. The idea is that you can run ACID on any orders you choose

        Parameters
        ----------
        orders : :py:type:`Array1D` | int | str | None, optional
            The orders to run ACID on. This can be provided as a single integer for one order, a list of integers for multiple specific orders, 
            the string "all" to run on all orders, or None to run on all orders. Default is None, which will run all orders.
        use_index_mapping : bool, optional
            If False, will not use the order to index mapping, instead orders are indexed directly. Default is True. Only applies for int or array inputs for orders.
        worker : :py:type:`IntLike` | None, optional
            Used in conjunction with nworkers. If an integer is provided, it specifies the worker number for this process. 
            When both worker and nworkers are provided, all the orders specified in "orders" will be split evenly across the nworkers. 
            For example, if there are 100 orders, and nworkers is 4, then worker 0 will run orders 0-24, worker 1 will run orders 25-49, etc. 
            The workers are 0-indexed. Default is None, which means no splitting and all specified orders will be run in this process.
        nworkers : :py:type:`IntLike` | None, optional
            The total number of workers to use to split the orders. See the "worker" parameter for more details. Default is None.
        store_sampler : bool, optional
            If True, the sampler object from the ACID run will be stored in the same folder as the resulting data pickles.
            This will take up more disk space, but allow for use of the :py:class:`Result` methods requiring the sampler attribute.
            We recommend leaving this on True if using deterministic sampling, otherwise set to False. Default is True.
        size_limit : Scalar | None, optional
            A hard size limit to the sampler in GB.
            If the sampler exceeds this size, it will not be stored regardless of the store_sampler flag.
            This is to avoid accidentally storing very large samplers. If None, no limit is set. Default is 1GB.
            A warning will be printed if this size_limit forces the store_sampler to be False if store_sampler was set to True.
        overwrite : bool, optional
            If True, will allow overwriting existing data and sampler pickles in the save_dir. Default is None, which will use the class
            default behaviour set in initialization (which is False). If False, this will skip running ACID on orders 
            that already have result pickles in the save_dir.
        overwrite_kwargs : bool, optional
            If True, any keys in the kwargs that are also in the config for the Data instance will be overwritten by the kwargs values.
            Use with caution, by default False.
        pack : bool, optional
            If True, a copy of all the DataList instances are packed into a single pickle for faster loading. Only applies if this task is not being split
            over multiple workers, otherwise, reverts back to False. By default, False.
        **kwargs :
            Additional keyword arguments to be passed to the ACID method for each order. These will not overwrite any existing keys unless
            overwrite_kwargs is set to True, in which case they will overwrite existing keys in the config for the Data instance for that order.
            The kwargs passed also allow you to add/overwrite the linelist and velocities in the Data instance with the same overwrite logic.
        """
        from ..acid import Acid # local import to avoid circular imports, since Acid imports Data

        # Configure overwrite from class default if not input in the method call
        if overwrite is None:
            overwrite = self.overwrite

        # Validate worker and nworkers inputs for splitting orders across workers, and set defaults if not provided for easier logic below.
        if worker is not None or nworkers is not None:
            if worker is None or nworkers is None:
                raise ValueError("Both worker and nworkers must be provided together to use the worker splitting functionality.")
            if worker < 0 or worker >= nworkers:
                raise ValueError(f"worker must be between 0 and nworkers-1. Got: worker={worker}, nworkers={nworkers}")
        else:
            nworkers = 1 # if no worker splitting, just set nworkers to 1 for the logic below to work

        # Handle formats for orders input
        if isinstance(orders, int):
            orders = orders if use_index_mapping else self.i2o[orders]
            orders = np.array([orders], dtype=np.int32)
        elif isinstance(orders, str):
            if orders.lower() == "all":
                orders = self.orders
            else:
                raise ValueError(f"If orders is a string, it must be 'all' to run ACID on all orders. Got: {orders!r}")
        elif orders is None:
            orders = self.orders
        elif isinstance(orders, (list, np.ndarray)):
            if not all(isinstance(o, (int, np.integer)) for o in orders):
                raise ValueError(f"If orders is a list, all elements must be integers. Got: {orders!r}")
            if use_index_mapping:
                if not all(o in self.orders for o in orders):
                    raise ValueError(f"All orders in the input list must be in the DataList. Got: {orders!r}, but available orders are: {self.orders!r}")
            else:
                if not all(o in self.i2o for o in orders):
                    raise ValueError(f"All orders in the input list must be in the DataList. Got: {orders!r}, but available orders are: {self.orders!r}")
                orders = [self.i2o[o] for o in orders] # converts from order to index if use_index_mapping is False, otherwise assumes orders are indexed directly
            orders = np.array(orders, dtype=np.int32)
        else:
            raise ValueError(f"orders must be an int, a list of ints, 'all', or None. Got: {orders!r}")

        # Now we split the orders across workers and select the orders for this worker
        if nworkers > 1:
            orders = np.array_split(orders, nworkers)[worker]

        # Check if linelist or velocities were in kwargs and remove them now once if so
        ll = kwargs.pop("linelist", None)
        vel = kwargs.pop("velocities", None)

        iterable = tqdm(orders, "Running ACID on orders", unit="order") if self.verbose >= 2 else orders
        for order in iterable:

            data = self.data_list[self.o2i[order]]

            # Check if ACID already ran for this order
            if os.path.exists(data.config.save_path) and overwrite is False:
                if data.complete:
                    if self.verbose >= 2:
                        print(f"An ACID completed result for order {order} already exists. \n"
                                f"Skipping this order. To overwrite existing results, set overwrite=True.")
                    continue
                elif data.exception is not None:
                    if self.verbose >= 2:
                        print(f"An ACID run for order {order} previously encountered an exception. \n"
                                f"Skipping this order. To retry and overwrite existing results, set overwrite=True.")
                    continue

            # Handling if any kwargs were input
            # Only overwrite if overwrite_kwargs is True, otherwise keep the existing linelist/velocities in the Data instance
            if ll is not None:
                data.linelist = ll if overwrite_kwargs else data.linelist
            if vel is not None:
                data.velocities = vel if overwrite_kwargs else data.velocities
            if overwrite_kwargs:
                data.config.update_hipri(**kwargs)
            else:
                data.config.update_lowpri(**kwargs)

            # The following try-except loops came from just testing ACID on a lot of different orders, stars, and instruments
            failed_msg = f"Order {order} (list index {self.o2i[order]}) failed with error:"
            exception_raised = False
            try:
                _result = Acid(data=data).ACID() # All params are stored in Data and Config (in Data)
            except LineListRangeError:
                print(f"{failed_msg} line list range error. Your linelist is likely out of "\
                      f"range of the wavelengths. Skipping this order.", flush=True)
                exception_raised = True
            except ContinuumError:
                print(f"{failed_msg} continuum fitting error. The fitted continuum likely "\
                      f"had negative values. Skipping this order.", flush=True)
                exception_raised = True
            except SNCutError:
                print(f"{failed_msg} S/N cut error. The S/N of the spectrum is likely too "\
                      f"low, and no lines survived the cut. Skipping this order.", flush=True)
                exception_raised = True
            # If no known exception arose, just print the last 3 calls in traceback for debugging and skip the order.
            except Exception as e:
                print(f"{failed_msg} unknown error, see traceback. Skipping this order. Traceback:\n", flush=True)
                tb.print_exc(limit=-3)
                exception_raised = True
                data.exception = str(e)
            
            if exception_raised:
                try:
                    data.traceback = tb.format_stack() # include the new exception in the data instance for future reference
                    data.save() # save the data instance with the exception for future reference
                except:
                    print(f"Failed to save the Data instance for order {order} after an exception was raised.\n" \
                          f"This is likely due to a corrupted Data instance.", flush=True)

        # Once all the orders have been done, we can repack the all the data instances (if asked) into one to speedup loading time
        # The data instances are very light as they do not store the sampler, so we can usually afford to pack and store duplicates
        if pack and np.array_equal(orders, self.orders):
            self.save()

    def save(self, save_dir:str|None=None) -> None:
        """
        Packs all of the DataList instances into a single DataList pickle, and save the state of the datalist to this pickle.
        Otherwise, the data instances can always be reloaded separately from the inidividual resulting pickle files in the directory for their order.
        Or just wherever the Config has them stored.
        The pickle file contains a dictionary with the list of Data objects (converted to dictionaries) and the save_dir.
        The filename is always "datalist.pkl", as save_dir must be a directory.
        If save_dir is not provided, self.save_dir is used. If that is also None, a ValueError is raised.
        All the orders should be in the memory to run this function, you can ensure they are all loaded with the load() method 
        and pointing to the directory with all the data pickles.

        Parameters
        ----------
        save_dir : str | None, optional
            The directory to save the DataList pickle file. If None, self.save_dir is used. Default is None.
        """
        if save_dir is not None:
            self.save_dir = save_dir
        if self.save_dir is None:
            raise ValueError("No save directory provided and save_dir was not set.")
        save_loc = os.path.join(self.save_dir, "datalist.pkl")
        d = {
            "verbose": self.verbose,
            "data_list": [data.to_dict() for data in self.data_list],
        }
        with open(save_loc, "wb") as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path:str, verbose:int|str|bool|None=None) -> DataList:
        """
        Loads a DataList from a pickle file. The pickle file should contain a dictionary with the list of Data objects (converted to dictionaries) and the save_dir.
        Will attempt to load from datalist.pkl in the provided path if it is a directory, otherwise will attempt to load from the provided path directly. 
        If neither of those work, it will attempt to load from result pickles in a results directory within the provided path.

        Parameters
        ----------
        path : str
            The directory containing the datalist.pkl file, or the datalist.pkl itself. Note that the directories containing the results should also be in here.
        verbose : int | str | bool | None, optional
            The verbosity level to use when loading the DataList. If None, the verbosity level from the pickle file is used. Default is None. The verbosity only
            affects this function and will not overwrite the verbosity level of the DataList once it is loaded.

        Returns
        -------
        DataList
            The loaded DataList object.
        """
        abspath = os.path.abspath
        join = os.path.join
        isdir = os.path.isdir
        exists = os.path.exists

        path = abspath(path)
        if path.endswith("datalist.pkl"):
            if not exists(path):
                raise ValueError(f"No pickle file found at {path} to load the DataList from.")
            else:
                path = os.path.dirname(path)
        elif not isdir(path):
            raise ValueError(f"The provided path {path} is not a directory, or a datalist pickle file.\n"
                             f"You should provide a path to a directory containing the folders with the data pickles and sampler files.")

        d = {}
        if exists(join(path, "datalist.pkl")):
            with open(join(path, "datalist.pkl"), "rb") as f:
                d = pickle.load(f)

        if verbose is None:
            verbose = d.get("verbose", None)
        verbose = Config(verbose=verbose).verbose

        # If the datalist was repacked, load directly from there
        if "data_list" in d:
            data_list = [Data().from_dict(payload) for payload in d["data_list"]]

            folder_moved_flag = False
            for data in data_list:
                folder_moved_flag |= cls._set_paths_for_data(data, path)

            datalist = cls.from_datalist(data_list, save_dir=path, verbose=verbose)
            datalist.save() # repack with new save locations

            if folder_moved_flag and verbose >= 1:
                print("Warning: At least one Data instance did not match the current location and has been updated.")

            return datalist

        dir_list = os.listdir(path)        
        data_list = []
        folder_moved_flag = False
        dir_list = dir_list if verbose < 2 else tqdm(dir_list, "Loading Data instances from directory", unit="folder")
        for folder in dir_list:
            if isdir(join(path, folder)) and folder.startswith("order_"):
                save_path = abspath(join(path, folder, "data.pkl"))
                sampler_path = abspath(join(path, folder, "sampler.h5"))
                if exists(save_path):
                    data = Data.load(save_path)
                    folder_moved_flag |= cls._set_paths_for_data(data, path)
                    data_list.append(data)

        if folder_moved_flag and verbose is not None and verbose >= 1:
            print(f"Warning: At least one of the Data instances found in the directory does not match the current location, it has been updated.")

        obj = cls.from_datalist(data_list, save_dir=path, verbose=verbose)
        return obj

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, dir):
        if dir is not None:
            dir = utils.ensure_directory(dir, "DataList save directory")
        elif self._save_dir is None:
            if self.verbose >= 1:
                print("Warning: save_dir is set to None. No results will be saved. This is not recommended.")
        self._save_dir = dir
        return

    @property
    def combined_profile(self) -> tuple|None:
        """
        Returns the combined profile and its errors. If the combined profile has not been calculated yet, 
        it will attempt to combine the profiles without any exclusions.

        Returns:
            tuple[Array1D, Array1D]|None: The combined profile and its errors, or None if not available.
        """
        if self._combined_profile is None:
            try:
                self.combine_profiles()
            except Exception as e:
                raise ValueError(f"An attempt was made to combine profiles, as they did not already exist, but there was an exception:\n{e}")
        return self._combined_profile

    @property
    def data_list(self):
        return self._data_list

    @property
    def results(self):
        """
        Returns a list of Result objects for each Data instance in the DataList.
        If a Data instance does not have a result, None is returned for that order.
        This property is useful for not reaccessing ther result each time a plot is made.
        
        Returns:
            list[Result|None]: A list of Result objects or None for each order in the DataList.
        """
        if self._results is None:
            if self.verbose >= 1:
                print("Accessing results, the output below comes from initialising the Result object" \
                " and will only be shown once for this DataList instance.")
            self._results = [data.result for data in self.data_list]

        return self._results

    @data_list.setter
    def data_list(self, data_list):
        """
        Sets the data list and ensures that it is a list of Data instances. 
        Also sorts the list by order and updates the order to index mapping.
        """
        if not isinstance(data_list, list):
            raise ValueError("data_list must be a list of Data instances.")
        if not all(isinstance(data, Data) for data in data_list):
            raise ValueError("All elements in data_list must be instances of the Data class.")
        self._data_list = data_list
        self.sort_by_order() # ensures that the list is sorted and the order to index mapping is updated when setting a new list

    def combine_profiles(self, exclude:int|Array1D|None=None, must_have_converged:bool=False, od:bool=True) -> None:
        """
        Calculates the combined profile and its errors across all orders, excluding any orders specified in the exclude argument.
        
        Parameters
        ----------
        exclude : int | list[int] | None, optional
            Orders to exclude from the combined profile calculation.
        must_have_converged : bool, optional
            If True, only includes orders that have converged in the combined profile calculation. Default is False, which 
            includes all orders regardless of convergence status.
        od : bool, optional
            If True, the combination is done in optical depth. The returned profile is always in flux. By default, True.
        """
        if isinstance(exclude, int):
            exclude = [exclude]
        elif exclude is None:
            exclude = []

        if not all(o in self.orders for o in exclude):
            raise ValueError(f"All orders in the exclude list must be in the DataList. \nGot: {exclude!r}, but available orders are: {self.orders!r}")

        profiles = []
        errors = []
        covariances = []
        for data in self.data_list:
            if data.config.order in exclude:
                continue
            try:
                if (must_have_converged and not data.result.converged):
                    continue
            except:
                continue
            if "final" not in data.profile:
                continue

            # We actually want to combine in optical depth - more stable
            p, e, c = utils.flux_to_od(data.profile["final"][0], data.profile["final"][1], cov_matrix=data.profile["final"][2], od=od)

            profiles.append(p)
            errors.append(e)
            covariances.append(c)

        self._combined_profile = utils.combine_profiles(profiles, errors, covariances)

        self._combined_profile = utils.od_to_flux(self._combined_profile[0], self._combined_profile[1], cov_matrix=self._combined_profile[2], od=od)

        self.excluded_orders = exclude
        return

    def plot_combined_profile(self, return_fig:bool=False) -> None|tuple[plt.Figure, plt.Axes]:
        """
        Plots the combined profile across all orders

        Parameters
        ----------
        return_fig : bool
            If True, returns the figure and axis objects instead of displaying the plot.
        
        Returns
        -------
        tuple[plt.Figure, plt.Axes] | None
            The figure and axis objects if return_fig is True, otherwise None.

        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        for data in self.data_list:
            if "final" not in data.profile or data.config.order in self.excluded_orders:
                continue # failed or excluded orders
            ax.errorbar(self.velocities, data.profile["final"][0], alpha=0.2, color="C0",
                        label=f"All profiles" if data.config.order == self.orders[0] else None)

        ax.errorbar(self.velocities, self.combined_profile[0], self.combined_profile[1], color="black", fmt=".-", ecolor="red", label="Combined profile")
        ax.legend()
        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Relative Flux")
        ax.set_title("Combined ACID profiles")
        ax.grid(True)
        if return_fig:
            return fig, ax
        plt.show()

    def fit_profile(self, **kwargs) -> None|tuple:
        """
        Fits the combined profile across all orders.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to pass to the :py:function:`Profiles.plot_fit` method.
        
        Returns
        -------
        tuple[plt.Figure, plt.Axes] | None
            The figure and axis objects from the profile fit plot if return_fig is True, otherwise None.
        """
        from ..profiles import Profiles
        profiles = Profiles(self.velocities, *self.combined_profile)
        return profiles.plot_fit(**kwargs)

    def plot_all_profiles(self, od:bool=False, return_fig:bool=False) -> None|tuple[plt.Figure, plt.Axes]:
        """
        Plots all the profiles for each order in the DataList. The combined profile is also shown.

        Parameters
        ----------
        od : bool, optional
            If True, shows the profiles in optical depth.
        return_fig : bool, optional
            If True, returns the figure and axis objects instead of displaying the plot.

        Returns
        -------
        tuple[plt.Figure, plt.Axes] | None
            The figure and axis objects if return_fig is True, otherwise None.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        norm = mpl.colors.Normalize(vmin=self.order_range[0], vmax=self.order_range[-1])
        cmap = mpl.colormaps.get_cmap("viridis")#, len(self.order_range))

        peak_vel_idx = np.nanargmin(self.combined_profile[0])
        min_prof = 1
        for data in self.data_list:
            order = data.config.order
            color = cmap(norm(order))
            if "final" not in data.profile:
                continue # failed orders
            ax.plot(self.velocities, utils.flux_to_od(data.profile["final"][0], od=od), alpha=0.2, color=color)

            if data.profile["final"][0][peak_vel_idx] < min_prof:
                min_prof = data.profile["final"][0][peak_vel_idx]
        
        ax.errorbar(self.velocities, # self.combined_profile[0], self.combined_profile[1],
                    *utils.flux_to_od(self.combined_profile[0], self.combined_profile[1], od=od),
                    color="black", fmt=".-", ecolor="red", label="Combined profile", zorder=10)

        ax.axhline(1, color="black", linestyle="--", alpha=0.5)

        # Show colour map
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation="vertical", label="Order")

        # Set sensible limits
        if not od:
            ax.set_ylim(max(min_prof-0.1, 0), 1.2)
        # if od, limits are usually sensible due to log scaling
        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Relative Flux")
        ax.set_title("All ACID profiles")
        ax.grid(True)
        ax.legend()
        if return_fig:
            return fig, ax
        plt.show()

    def plot_mean_profile_errors(self, return_fig:bool=False) -> None|tuple[plt.Figure, plt.Axes]:
        """
        Plots the errors of all the profiles for each order in the DataList.

        Parameters
        ----------
        return_fig : bool
            If True, returns the figure and axis objects instead of displaying the plot.

        Returns
        -------
        tuple[plt.Figure, plt.Axes] | None
            The figure and axis objects if return_fig is True, otherwise None.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        errors = []
        for data in self.data_list:
            if "final" not in data.profile:
                errors.append(np.nan)
                continue
            errors.append(np.mean(data.profile["final"][1]))

        ax.plot(self.orders, errors, marker='o', linestyle='-', color='blue')
        ax.set_xlabel("Order")
        ax.set_ylabel("Mean Profile Error")
        ax.set_title("Mean Profile Errors for each order")
        ax.set_yscale("log")
        ax.grid(True)

        if return_fig:
            return fig, ax
        plt.show()

    def plot_chi2(self, return_fig:bool=False) -> None|tuple[plt.Figure, plt.Axes]:
        """
        Plots the chi-squared values against order in the DataList.
        This helps diagnose which orders have a bad fit and may need to be excluded from the combined profile.

        Parameters
        ----------
        return_fig : bool
            If True, returns the figure and axis objects instead of displaying the plot.

        Returns
        -------
        tuple[plt.Figure, plt.Axes] | None
            The figure and axis objects if return_fig is True, otherwise None.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        orders = []
        chi2_values = []
        for data in self.data_list:
            if "final" in data.profile:
                orders.append(data.config.order)
                try:
                    flux = np.asarray(data.flux["final"])
                    model = np.asarray(data.forward_y["final"])
                    err = np.asarray(data.errors["final"])
                    chi2 = np.sum(((flux-model)/err) ** 2)
                    chi2_values.append(chi2)
                except Exception as e:
                    print(f"Warning: Could not calculate chi-squared for order {data.config.order}. :\n{e}")
                    chi2_values.append(np.nan)
        
        ax.plot(orders, chi2_values, marker='o', linestyle='-', color='blue')
        ax.set_xlabel("Order")
        ax.set_ylabel("Chi-squared")
        ax.set_yscale("log")
        ax.set_title("Chi-squared values for each order")
        ax.grid(True)
        
        if return_fig:
            return fig, ax
        plt.show()

    @staticmethod
    def _set_paths_for_data(data: Data, save_dir: str) -> bool:
        """Update and save a Data instance only if its directory paths changed."""
        order = data.config.order

        save_path = os.path.abspath(
            os.path.join(save_dir, f"order_{order}", "data.pkl")
        )
        sampler_path = os.path.abspath(
            os.path.join(save_dir, f"order_{order}", "sampler.h5")
        )

        stored_save_path = data.config.save_path
        stored_sampler_path = data.config.sampler_path
        # TODO: on kelvin this was printing and moving unexpectedly, print the locations and find out why
        changed = (
            stored_save_path is None
            or os.path.abspath(stored_save_path) != save_path
            or stored_sampler_path is None
            or os.path.abspath(stored_sampler_path) != sampler_path
        )

        if changed:
            data.config.save_path = save_path
            data.config.sampler_path = sampler_path

            if os.path.exists(sampler_path):
                data.sampler = sampler_path

            data.save()

        return changed
