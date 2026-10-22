from __future__ import annotations
import contextlib, functools, inspect, os, warnings
import corner
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from beartype import beartype
from emcee import EnsembleSampler
from . import utils
from .data import Data
from .diagnostics.errors import *
from .diagnostics.warnings import *
from .diagnostics.logging import get_logger
from .utils import IntLike
try:
    from dynesty.sampler import Sampler
    from dynesty import plotting as dyplot
except ImportError:
    Sampler = None
    dyplot = None

logger = get_logger(__name__)

def _require_sampler(method):
    # Make sure sampler object is available before calling method
    sig = inspect.signature(method)
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind_partial(self, *args, **kwargs)
        inputted_sampler = bound.arguments.get("sampler", None)
        self.initiate_sampler(inputted_sampler, _method_name=method.__qualname__)
        return method(self, *args, **kwargs)
    return wrapper


@beartype
class Plot:
    """Plot the available stages of a shared Data instance without processing results.

    Accepts Data, Acid or Result. Profiles and forward models can be selected with ``key``;
    sampler diagnostics require a stored sampler or one supplied to the method.
    """

    def __init__(self, data: Data|object) -> None:
        from .acid import Acid
        from .result import Result
        if isinstance(data, (Acid, Result)):
            self.data = data.data
        elif isinstance(data, Data):
            self.data = data
        else:
            raise ACIDInputError(f"First argument must be either an Acid, Result or Data object. Got {type(data)} instead.")
        self.config = self.data.config
        self.sampler_initialized = False

    def linelist(self, idx:np.ndarray|list|None=None, bounds:tuple|list|None=None, fig_ax:tuple|None=None, return_fig:bool=False) -> None|tuple:
        """
        Plots the linelist points with their corresponding depths as delta-function lines.

        Parameters
        ----------
        idx : np.ndarray or list, optional
            The indices of the linelist points to plot. If None, plots all points.
        bounds : tuple or list, optional
            The wavelength bounds (min, max) to clip the linelist for plotting. If None, plots all points.
        fig_ax : tuple, optional
            A tuple of (figure, axis) objects to plot on. If None, a new figure and axis are created.
        return_fig : bool, optional
            If True, returns the figure and axis objects instead of displaying the plot.

        Returns
        -------
        tuple or None
            If return_fig is True, returns a tuple of (figure, axis) objects. Otherwise, returns None.
        """
        if self.data.linelist is None:
            raise ACIDStateError("No linelist found. Please set a linelist before trying to plot it.")
        wl = self.data.linelist["wavelengths"]
        depths = self.data.linelist["depths"]

        if idx is not None:
            wl = wl[idx]
            depths = depths[idx]

        # Clip the linelist to the specified bounds if provided, and to the min_depth
        if bounds is not None:
            idx_in_bounds = (wl >= bounds[0]) & (wl <= bounds[1])
            wl = wl[idx_in_bounds]
            depths = depths[idx_in_bounds]

        # Plot linelist
        if fig_ax is None:
            fig, ax = plt.subplots(figsize=(15, 9))
        else:
            fig, ax = fig_ax

        ax.vlines(wl, 1, 1-depths, color='C0', alpha=0.7, label='Line List')
        ax.set_title('Line List')
        ax.set_xlabel('Wavelength (Angstroms)')
        ax.set_ylabel('Relative Line Depth')
        ax.legend()
        if return_fig:
            return fig, ax
        utils.show_or_save(plt, self.config.figure_dir, "linelist.png", output_dir=self.config.dir)

    def masking_lines(self, return_fig:bool=False) -> None|tuple:
        """Forward to :py:meth:`Config.plot_masking_lines`."""
        return self.config.plot_masking_lines(return_fig=return_fig)

    def continuum_fit(self, key:str="masked", return_fig:bool=False, save_fig:str|None=None) -> None|tuple:
        """
        Plots the result of the continuum fitting step, showing the original spectrum, the fitted continuum, and the clipped points used for the continuum fit.

        Parameters
        ----------
        key : str, optional
            The type of continuum fit to plot and pull the data from, either "initial" for the initial continuum fit or
            "masked" for the continuum fit after residual masking. Default is "masked".
        return_fig : bool, optional
            Whether to return the figure and axis objects instead of showing the plot, by default False.
        save_fig : str or None, optional
            If provided, the path to save the figure. If None, the figure will not be saved. Default is None.
        """
        # Check we have all inputs needed for plot
        if key not in ["initial", "masked"]:
            raise ACIDInputError("key must be either 'initial' or 'masked' (these are the only two steps that produce a scipy continuum fit).")
        if key not in self.data.plotting_variables:
            raise ACIDStateError(f"No plotting variables found for key={key!r}. " \
                             "Please ensure that the continuum fit has been performed for this key.")
        if not all(
            attr in self.data.plotting_variables[key] for attr in [
                "clipped_waves", "clipped_flux", "good"]
            ):
            raise ACIDStateError("To plot the continuum fit, please first run the continuum fitting in ACID step for the specified plot_type.")

        # Unpack variables
        good                     = self.data.plotting_variables[key]["good"]
        clipped_waves            = self.data.plotting_variables[key]["clipped_waves"]
        clipped_flux             = self.data.plotting_variables[key]["clipped_flux"]

        # Normalise wavelengths and plot flux and fit
        fig, ax = plt.subplots(figsize=(15, 9))
       
        ax.plot(self.data.wavelengths[key], self.data.flux[key], label='Original Spectrum', color="C0", alpha=0.7)
        ax.plot(self.data.wavelengths[key], self.data.continuum[key], label='Fitted Continuum', color='red')
        a, b = utils.get_normalisation_coeffs(self.data.wavelengths[key])
        ax.plot((clipped_waves[good]-b)/a, clipped_flux[good], 'o', label='Continuum Normalized Spectrum', color='green')

        # Plot the linelist points, with a color corresponding to their depth in the linelist within the range
        # Only plot the 20 strongest lines to avoid overcrowding.
        ll_wl = self.data.linelist["wavelengths"]
        ll_depths = self.data.linelist["depths"]
        from .lsd import LSD
        ll_wl, ll_depths = LSD.clip_wavelengths(self.data.wavelengths[key], ll_wl, ll_depths)
        idx = np.argsort(ll_depths)
        ll_wl = ll_wl[idx]
        ll_depths = ll_depths[idx]
        ll_wl = ll_wl[-20:]
        ll_depths = ll_depths[-20:]

        # Try colouring them, but often the linelist points will be outside the wavelength range so just skip if 
        # there's an error to avoid breaking the plot
        try:
            cmap = plt.cm.viridis_r
            norm = mpl.colors.Normalize(vmin=np.nanmin(ll_depths), vmax=np.nanmax(ll_depths))
            for i, (wl, depth) in enumerate(zip(ll_wl, ll_depths)):
                ax.axvline(
                    wl,
                    color=cmap(norm(depth)),
                    linestyle="--",
                    alpha=1,
                    label="Line List (20 strongest lines in region)" if i == 0 else None,
                )
            # Create colorbar for depth
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])  # needed for some matplotlib versions
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Line depth")
        except:
            warnings.warn("There was an error plotting the linelist points, most likely your linelist range is outside your wavelength range.",
                          ACIDRuntimeWarning, stacklevel=2)
            pass

        # Plot the line masks with their names
        x = self.data.wavelengths[key]
        line_mask = self.config.masking_lines.get_masks(x, with_names=True)
        for i, (name, masks) in enumerate(line_mask.items()):
            padded = np.concatenate(([False], masks, [False]))
            starts = np.flatnonzero(~padded[:-1] & padded[1:])
            ends   = np.flatnonzero(padded[:-1] & ~padded[1:])
            for j, (start, end) in enumerate(zip(starts, ends)):
                ax.axvspan((x[start]), (x[end-1]), color=f'C{i+2}', alpha=0.3,
                        label=f"{name} Line masks" if j == 0 else None)
        
        # Plot the other two masking regions if in the masked plot type, these masking are only done after the initial fit
        if key == "masked":
            masked = self.data.pix_mask | self.data.sigma_mask
            padded = np.concatenate(([False], masked, [False]))
            starts = np.flatnonzero(~padded[:-1] & padded[1:])
            ends   = np.flatnonzero(padded[:-1] & ~padded[1:])
            for i, (start, end) in enumerate(zip(starts, ends)):
                ax.axvspan((x[start]), (x[end-1]),
                            color='red', alpha=0.15, label="Sigma masking and pixel deviation regions" if i == 0 else None)

        # Add labels and legend, and save or show figure
        plot_title = "Initial Continuum Fit" if key == "initial" else "Continuum Fit after Residual Masking"
        ax.set_title(plot_title)
        ax.legend()
        ax.set_ylim(np.min(self.data.flux[key])*0.9, np.max(self.data.flux[key])*1.1)
        if save_fig is not None:
            plt.savefig(save_fig)
        if return_fig:
            return fig, ax
        title = "continuum_fit_initial.png" if key == "initial" else "continuum_fit_masked.png"
        utils.show_or_save(plt, self.config.figure_dir, title, output_dir=self.config.dir)

    def residual_masking(self) -> None:
        """
        Creates 3 plots to show the result of the residual masking step, showing the residuals with the sigma clipping thresholds, 
        the masked regions, and the initial profile after masking.
        """
        # Check we have all inputs needed for plot
        if "masked" not in self.data.plotting_variables:
            raise ACIDStateError("No plotting variables found for masking. Residual masking likely has not been performed in Acid.")
        if "masked" not in self.data.wavelengths and "masked" not in self.data.flux:
            raise ACIDStateError("No masked wavelengths or fluxes found. Please ensure that the residual masking step has been performed")

        # Unpack variables
        x = self.data.wavelengths["combined"]
        y = self.data.flux["combined"]
        residuals = self.data.residuals["masked"]
        upper_clip = self.data.plotting_variables["masked"]["upper_clip"]
        lower_clip = self.data.plotting_variables["masked"]["lower_clip"]
        uc_finite = np.isfinite(upper_clip) # useful for plotting
        lc_finite = np.isfinite(lower_clip)
        pix_mask = self.data.pix_mask
        line_mask = self.data.line_mask
        full_mask = self.data.full_mask

        nremoved = np.sum(full_mask)
        logger.info(f"{nremoved}/{len(residuals)} pixels were removed after residual masking.")

        # Create plot and add residuals with sigma clipping thresholds and masked regions
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.axhline(0, color='black', linestyle='--', linewidth=1)

        utils.plot_masked_line(ax, x, residuals, ~full_mask, colors=["blue", "red"], label=["Residuals", "Masked Residuals"])

        # Show sigma clipping
        if uc_finite:
            ax.axhline(upper_clip, color='C0', linestyle='--', label='Sigma Clip Thresholds', linewidth=2)
        if lc_finite:
            lc_label = 'Sigma Clip Thresholds' if not uc_finite else None
            ax.axhline(lower_clip, color='C0', linestyle='--', linewidth=2, label=lc_label)

        # Show line masking regions
        line_mask = self.config.masking_lines.get_masks(x, with_names=True)
        for i, (name, masks) in enumerate(line_mask.items()):
            padded = np.concatenate(([False], masks, [False]))
            starts = np.flatnonzero(~padded[:-1] & padded[1:])
            ends   = np.flatnonzero(padded[:-1] & ~padded[1:])
            for j, (start, end) in enumerate(zip(starts, ends)):
                ax.axvspan((x[start]), (x[end-1]), color=f'C{i+2}', alpha=0.3,
                           label=f"{name.capitalize()} line masks" if j == 0 else None)

        # Show pix_chunk masked points:
        masked = pix_mask
        padded = np.concatenate(([False], masked, [False]))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends   = np.flatnonzero(padded[:-1] & ~padded[1:])
        for i, (start, end) in enumerate(zip(starts, ends)):
            ax.axvspan((x[start]), (x[end-1]),
                        color='red', alpha=0.15, label="Applied chunk deviation masking" if i == 0 else None)
        
        # And show chunk deviation range
        dev = self.config.dev_perc / 100
        ax.hlines([-dev, dev], xmin=np.min(x), xmax=np.max(x), color='C1', linestyle='--', linewidth=2, label="Chunk deviation masking range")

        # Set a good ylim off everything but the masked points
        ymax = np.nanmax([dev, upper_clip if uc_finite else np.nan, np.max(residuals[~full_mask])])
        ymin = np.nanmin([-dev, lower_clip if lc_finite else np.nan, np.min(residuals[~full_mask])])
        diff = (ymax - ymin) * 0.1 # Set an even 10% buffer on either side of max/min
        ax.set_ylim(ymin - diff, ymax + diff)

        ax.set_xlim(np.min(x), np.max(x))
        ax.grid(True)
        ax.set_title('Residuals with Sigma Clipping Thresholds')
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Residuals')
        ax.legend()
        utils.show_or_save(plt, self.config.figure_dir, "masking_residuals.png", output_dir=self.config.dir)

        # Plot the LSD profile
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            ax.plot(self.data.velocities, self.data.profile["masked"][0], label='LSD Profile after Masking and before sampling', color='red')
        except:
            ax.plot(self.data.velocities, np.median(self.data.profile["masked"][0], axis=0), label='MEDIAN LSD Profile after Masking and before sampling', color='red')
        ax.set_title('LSD Profile after Residual Masking')
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('LSD Profile')
        ax.axhline(1, color='black', linestyle='--')
        ax.legend()
        ax.grid(True)
        utils.show_or_save(plt, self.config.figure_dir, "masking_profile.png", output_dir=self.config.dir)

        # Finally plot the forward model
        x = self.data.wavelengths["combined"]
        y = self.data.flux["combined"]
        forward = self.data.forward_y["masked"]
        continuum = self.data.continuum["masked"]
        residuals = (y - forward) / forward
        fig, ax = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax[0].plot(x, y, label='Original data', color='black', linewidth=1)
        ax[0].plot(x, continuum, color='C1', linewidth=1, label='Fitted Continuum', linestyle='--')
        utils.plot_masked_line(ax[0], x, forward, ~full_mask, label=["Forward model", "Masked Forward model"])
        ax[0].set_title('Masked Forward Model')
        ax[0].set_ylabel('Flux')
        ax[0].grid(True)
        ax[0].legend()
        
        utils.plot_masked_line(ax[1], x, residuals, ~full_mask, label=["Residuals", "Masked Residuals"])
        ax[1].axhline(0, color='black', linestyle='--', linewidth=1)

        ymax = np.max(residuals[~full_mask])
        ymin = np.min(residuals[~full_mask])
        diff = (ymax - ymin) * 0.1 # Set an even 10% buffer on either side of max/min
        ax[1].set_ylim(ymin - diff, ymax + diff)

        ax[1].grid(True)
        # ax[1].set_title('Residuals of forward model with masked residuals')
        ax[1].set_xlabel('Wavelength')
        ax[1].set_ylabel('Residuals')
        ax[1].legend()
        plt.tight_layout()

        utils.show_or_save(plt, self.config.figure_dir, "initial_forward_model.png", output_dir=self.config.dir)

    @_require_sampler
    def walkers(
        self,
        sampler    : EnsembleSampler|None = None,
        burnin     : IntLike|None         = None,
        thin       : IntLike|None         = None,
        plot_kwargs  : dict|None         = None,
        subplot_kwargs : dict|None         = None,
        return_fig : bool                 = False,
        **kwargs
        ) -> None | tuple:
        """Plots, at maximum, the last 10 MCMC walkers for the LSD profile and continuum 
        polynomial coefficients.

        Parameters
        ----------
        sampler : :py:class:`emcee.EnsembleSampler`, optional
            Optionally provide a different sampler to plot from, otherwise,
            takes the sampler from the Result object, by default None
        burnin : :py:type:`IntLike`, optional
            Optionally define the number of burnin steps, by default uses the burnin calculated when the sampler was initiated.
        thin : :py:type:`IntLike`, optional
            Optionally define the number of thinning steps, by default uses the thinning calculated when the sampler was initiated.
        return_fig : bool, optional
            Whether to return the figure and axis objects instead of showing the plot, by default False

        Returns
        ----------
        If return_fig is True, returns a tuple (fig, ax) of the figure and axes objects containing, else None
        """
        # Set burnin and thin to defaults if not provided
        burnin = burnin if burnin is not None else self.burnin
        thin = thin if thin is not None else self.thin
        samples = self.sampler.get_chain(thin=int(thin))
        steps = np.arange(samples.shape[0]) * thin

        # Use dict defaults
        plot_defaults = {"fmt": "k", "alpha": 0.3}
        plot_kwargs = utils.set_dict_defaults(plot_kwargs, plot_defaults)

        subplot_defaults = {"figsize": (10, 20), "sharex": True}
        subplot_kwargs = utils.set_dict_defaults(subplot_kwargs, subplot_defaults)

        # Setup plot and plot the walkers for the default parameters
        naxes = len(self.default_params)
        fig, ax = plt.subplots(naxes, 1, **subplot_kwargs)
        fmt = plot_kwargs.pop("fmt", "k")
        for i in range(naxes):
            ax[i].plot(steps, samples[:, :, self.default_params[i]], fmt, **plot_kwargs)
            ax[i].axvspan(0, burnin, color="red", alpha=0.1, label="burn-in")
            ax[i].set_ylabel(self.default_param_labels[i])
        ax[-1].legend()
        ax[-1].set_xlabel("Step number")
        ax[-1].set_xlim(0, self.data.nsteps)
        ax[0].set_title('MCMC Walkers')
        plt.subplots_adjust(hspace=0.05)
        if return_fig:
            return fig, ax
        utils.show_or_save(plt, self.config.figure_dir, "walkers.png", output_dir=self.config.dir)

    @_require_sampler
    def traceplot(self, return_fig:bool=False, **kwargs) -> None | tuple:
        if not self.dynesty:
            raise ACIDInputError("Traceplot is only available for dynesty samplers, as emcee traceplots are already plotted in plot_walkers.")
        fig, ax = dyplot.traceplot(self.sampler.results, labels=self.default_param_labels, **kwargs)
        plt.suptitle('Dynesty Traceplot')
        if return_fig:
            return fig, ax
        utils.show_or_save(plt, self.config.figure_dir, "traceplot.png", output_dir=self.config.dir)

    @_require_sampler
    def corner(
        self,
        sampler    :EnsembleSampler|None = None,
        return_fig :bool                 = False,
        **kwargs,
        ) -> None | plt.Figure | tuple:
        """Creates a corner plot for at maximum the last 8 LSD profile and continuum polynomial coefficients.

        Parameters
        ----------
        sampler : emcee.EnsembleSampler | None, optional
            Optionally provide a different sampler to plot from, otherwise,
            takes the sampler from the Result object, by default None
        return_fig : bool, optional
            Whether to return the figure object instead of showing the plot, by default False
        **kwargs:
            Additional keyword arguments to pass to corner.corner() or dynesty.cornerplot().
        
        Returns
        ----------
        If return_fig is True, returns the figure object containing the corner plot, else None
        """
        if self.dynesty:
            fig, axes = dyplot.cornerplot(self.sampler.results, labels=self.default_param_labels, show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 16}, **kwargs)
            plt.suptitle('Dynesty Corner Plot')
            if return_fig:
                return fig, axes
            utils.show_or_save(plt, self.config.figure_dir, "corner.png", output_dir=self.config.dir)
            return

        # Get samples and thin and burnin from the class variables
        samples = self.sampler.get_chain()
        samples = self.sampler.get_chain(discard=self.burnin, flat=True, thin=self.thin)[:, self.default_params]

        # Set defaults:
        default_kwargs = {"show_title": True, "title_fmt": ".3f", "title_kwargs": {"fontsize": 16}}
        kwargs = utils.set_dict_defaults(kwargs, default_kwargs)

        # Use corner.corner to handle corner plot
        fig = corner.corner(samples, labels=self.default_param_labels, **kwargs)
        plt.suptitle('MCMC Corner Plot')
        if return_fig:
            return fig
        utils.show_or_save(plt, self.config.figure_dir, "corner.png", output_dir=self.config.dir)

    def profiles(
        self,
        grid            :bool      = True,
        labels          :dict|None = None,
        return_fig      :bool      = False,
        subplot_kwargs  :dict|None = None,
        errorbar_kwargs :dict|None = None,
        fig_ax                     = None,
        key             :str       = "final",
        ) -> None | tuple:
        """Plots the LSD profiles stored for the selected stage.

        Parameters
        ----------
        grid : bool, optional
            Show or hide grid, by default True
        labels : dict | None, optional
            Keys: 'xlabel', 'ylabel', and 'title'. Allows label overrides., by default None
        return_fig : bool, optional
            Whether to return the figure and axis objects instead of showing the plot, by default False
        subplot_kwargs : dict | None, optional
            Keyword arguments to be passed to plt.subplots(), by default None
        errorbar_kwargs : dict | None, optional
            Keyword arguments to be passed to ax.errorbar(), by default None
        fig_ax : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes] | None, optional
            Optionally provide an existing fig/axis tuple to plot on, by default None
        key : str, optional
            Stage to plot, such as "initial", "masked", or "final". The default
            "final" plots all input frames when per-frame results are available.
        
        Returns
        ----------
        If return_fig is True, returns a tuple (fig, ax) of the figure and axes objects containing the plot.
        Otherwise, displays the plot and returns None.
        """
        if key == "final" and self.data.profiles is not None:
            profiles = self.data.profiles
        elif key in self.data.profile:
            profiles = [self.data.profile[key]]
        else:
            raise ACIDStateError(f"No profiles found for key={key!r}. Please run the corresponding LSD step first.")

        # Set default errorbar kwargs
        errorbar_defaults = {
            "fmt"      : ".-",
            "ecolor"   : "red",
            "linewidth": 1,
        }
        errorbar_kwargs = utils.set_dict_defaults(errorbar_kwargs, errorbar_defaults)

        # Set default subplot kwargs
        subplot_kwargs = utils.set_dict_defaults(subplot_kwargs, {"figsize": (10, 6)})

        # Set default labels
        default_labels = {
            "title" : "Acid Profile",
            "xlabel": "Velocity (km/s)",
            "ylabel": "Normalised Flux"
        }
        labels = utils.set_dict_defaults(labels, default_labels)

        # Set useful variables
        nframes = len(profiles)
        if fig_ax is None:
            fig, ax = plt.subplots(**subplot_kwargs)
        else:
            fig, ax = fig_ax

        # Iterate through and plot frames
        for f, frame in enumerate(profiles):
            x, y, yerr = self.data.velocities, frame[0], frame[1]

            if x.shape == y.shape:
                errorbar_kwargs["label"] = f"Frame {f+1}" if nframes > 1 else None
                ax.errorbar(x, y, yerr=yerr, **errorbar_kwargs)

            else: # multi-profile case
                groups = np.unique(self.data.profile_groups)
                for prof, err, g in zip(y, yerr, groups):
                    errorbar_kwargs["label"] = f"Group {g}"
                    errorbar_kwargs["ecolor"] = None
                    ax.errorbar(x, prof, yerr=err, **errorbar_kwargs)               

        # Set sensible limits - implemented after multi-profile results showing some very out of range profiles
        ymin = max(np.min(profiles[0][0]), -0.1)
        ymax = min(np.max(profiles[0][0]), 1.2)
        ax.set_ylim(ymin-0.05, ymax+0.05)

        # Add labels and titles
        ax.set_title(labels["title"])
        ax.set_xlabel(labels["xlabel"])
        ax.set_ylabel(labels["ylabel"])
        ax.axhline(1, color='black', linestyle='--', linewidth=1)
        if ax.get_legend_handles_labels()[1]:
            ax.legend()
        ax.grid(grid)
        if return_fig:
            return fig, ax
        utils.show_or_save(plt, self.config.figure_dir, f"{key}_profiles.png", output_dir=self.config.dir)

    def forward_model(
        self,
        key              : str        = "final",
        normalized       : bool       = False,
        divide_by_median : bool       = False,
        show_masking     : bool       = True,
        show_continuum   : bool       = True,
        show_linelist    : bool       = False,
        fig_ax           : tuple|None = None,
        grid             : bool       = True,
        labels           : dict|None  = None,
        return_fig       : bool       = False,
        subplot_kwargs   : dict|None  = None,
        order            : IntLike|None = None,
        separate_orders  : bool       = False,
        ) -> None | tuple:
        """
        Plots the stored forward model against the spectrum for the selected stage.

        Parameters
        ----------
        key : str, optional
            Stage to plot, such as "initial", "masked", or "final", by default "final".
        normalized : bool, optional
            Whether to plot the normalized spectrum and forward model (divided by the continuum), 
            or the raw flux, by default False.
        divide_by_median : bool, optional
            Divides flux units by the median continuum value so that the y-axis is in units of median continuum,
            by default False. Only applies if normalized is False.
        show_masking : bool, optional
            Whether to show the masked regions of the spectrum, by default True
        show_continuum : bool, optional
            Whether to show the fitted continuum, by default True
        fig_ax: tuple | None
            Optionally provide an existing fig/axis tuple to plot on, by default None and
            creates a new figure and axis. The axis must be a 2 element array of axes, 
            where the first axis is for the spectrum and forward model, 
            and the second axis is for the residuals.
            If provided, the grid, labels, and titles should be set by you.
        grid : bool, optional
            Show or hide grid, by default True
        labels : dict | None, optional
            Keys: 'xlabel', 'ylabel', 'title', and 'residuals_ylabel'. Allows label overrides, by default None
        return_fig : bool, optional
            Whether to return the figure and axis objects instead of showing the plot, by default False
        subplot_kwargs : dict | None, optional
            Keyword arguments to be passed to plt.subplots(). Allows label overrides, by default None
        
        order : IntLike | None, optional
            Plot only this instrument order. None plots all orders.
        separate_orders : bool, optional
            Stack a spectrum/residual pair for each order in one figure, with
            independent axes between orders. Default is False. Returned axes
            have shape (n_orders, 2) when True.

        Returns
        ----------
        If return_fig is True, returns a tuple (fig, ax) of the figure and axes objects containing the plot.
        Otherwise, displays the plot and returns None.
        """

        if not all(key in values for values in (
            self.data.forward_x, self.data.forward_y, self.data.flux, self.data.continuum
        )):
            raise ACIDStateError(f"No forward model found for key={key!r}. Please run the corresponding LSD step first.")

        # Set default labels
        default_labels = {
            "title"           : "Forward Model Fit to Observed Spectrum",
            "xlabel"          : "Wavelength (Angstroms)",
            "ylabel"          : "Normalised Flux",
            "residuals_ylabel": "Residuals",
        }
        labels = utils.set_dict_defaults(labels, default_labels)

        slices = self.data.continuum_slices.get(key)
        if slices is None:
            slices = np.array([[0, len(self.data.flux[key]), 0, 0]])
        numbers = np.atleast_1d(self.config.order).tolist()
        indices = list(range(len(slices)))
        if order is not None:
            if order not in numbers:
                raise ACIDInputError(f"Order {order} is unavailable; available orders are {list(numbers)}.")
            indices = [numbers.index(order)]

        if separate_orders:
            if fig_ax is None:
                options = dict(subplot_kwargs or {})
                options.setdefault("figsize", (12, 8 * len(indices)))
                options.setdefault("gridspec_kw", {"height_ratios": [3, 1] * len(indices)})
                options.update(sharex=False, sharey=False, squeeze=False)
                fig, axes = plt.subplots(2 * len(indices), 1, **options)
            else:
                fig, axes = fig_ax
            axes = np.asarray(axes).reshape(len(indices), 2)
            for index, pair in zip(indices, axes):
                self.forward_model(key=key, normalized=normalized, divide_by_median=divide_by_median,
                    show_masking=show_masking, show_continuum=show_continuum, show_linelist=show_linelist,
                    fig_ax=(fig, pair), grid=grid, labels=labels, return_fig=True, order=numbers[index])
                if fig_ax is None:
                    pair[0].set_title(f"{labels['title']} — Order {numbers[index]}")
                    pair[0].set_ylabel(labels["ylabel"])
                    pair[1].set_ylabel(labels["residuals_ylabel"])
                    pair[1].set_xlabel(labels["xlabel"])
                    for axis in pair:
                        axis.grid(grid)
            if return_fig:
                return fig, axes
            utils.show_or_save(plt, self.config.figure_dir, "forward_model.png", output_dir=self.config.dir)
            return

        # Set default subplot kwargs
        default_kwargs = {
            "figsize": (12, 8),
            "sharex": True,
            "gridspec_kw": {'height_ratios': [3, 1]}
        }
        subplot_kwargs = utils.set_dict_defaults(subplot_kwargs, default_kwargs)

        # Plotting
        if fig_ax is not None:
            fig, ax = fig_ax
        else:
            fig, ax = plt.subplots(2, 1, **subplot_kwargs)
            ax[0].set_title(labels["title"])
            ax[1].set_xlabel(labels["xlabel"])
            ax[0].set_ylabel(labels["ylabel"])
            ax[1].set_ylabel(labels["residuals_ylabel"])
            ax[0].grid(grid)
            ax[1].grid(grid)
            plt.subplots_adjust(hspace=0.05)

        mask = self.data.full_mask
        if mask is None or len(mask) != len(self.data.flux[key]):
            mask = np.zeros(len(self.data.flux[key]), dtype=bool)
            # Recover masks for joint results saved before full_mask was stored.
            mcmc_slices = self.data.continuum_slices.get("mcmc")
            if key != "mcmc" and mcmc_slices is not None:
                for (i, j, _, _), (a, b, _, _) in zip(slices, mcmc_slices):
                    mask[i:j] = ~np.isin(self.data.forward_x[key][i:j], self.data.wavelengths["mcmc"][a:b])

        flux_limits, residual_limits = [], []
        median_continuum = 1.
        if divide_by_median and not normalized:
            median_continuum = np.median(np.concatenate([
                utils.drop_edges(self.data.continuum[key][slices[index, 0]:slices[index, 1]])
                for index in indices]))
        ax[1].axhline(0, color='black', linestyle='--', linewidth=1)
        for index in indices:
            i, j, _, _ = slices[index]
            # Trim and plot each order independently, even on the combined axes.
            wavelengths = utils.drop_edges(self.data.forward_x[key][i:j])
            flux = utils.drop_edges(self.data.flux[key][i:j]).copy()
            forward = utils.drop_edges(self.data.forward_y[key][i:j]).copy()
            continuum_model = utils.drop_edges(self.data.continuum[key][i:j]).copy()
            full_mask = utils.drop_edges(~mask[i:j])
            ll_mask = (self.data.order_ll_masks[index] if self.data.order_ll_masks
                       else self.data.ll_mask)
            ll_wl = self.data.linelist["wavelengths"][ll_mask]
            in_order = (ll_wl >= wavelengths.min()) & (ll_wl <= wavelengths.max())
            ll_wl = ll_wl[in_order]
            ll_depths = 1 - self.data.linelist["depths"][ll_mask][in_order]
            ll_tops = np.ones_like(ll_wl)

            if normalized:
                flux /= continuum_model
                forward /= continuum_model
                continuum_model[:] = 1
            else:
                # Interpolation never crosses an order boundary or overlap.
                continuum_at_ll = np.interp(ll_wl, wavelengths, continuum_model)
                ll_depths *= continuum_at_ll
                ll_tops *= continuum_at_ll

            if divide_by_median:
                flux /= median_continuum
                forward /= median_continuum
                continuum_model /= median_continuum
                ll_depths /= median_continuum
                ll_tops /= median_continuum

            residuals = flux - forward
            ax[0].plot(wavelengths, flux, color='black', linewidth=1, label='Observed Spectrum')
            if show_continuum:
                ax[0].plot(wavelengths, continuum_model, color='C1', linewidth=1,
                           label='Fitted Continuum', linestyle='--')
            if show_masking:
                utils.plot_masked_line(ax[0], wavelengths, forward, full_mask,
                                       label=["Forward model", "Masked Forward model"])
                utils.plot_masked_line(ax[1], wavelengths, residuals, full_mask,
                                       label=["Residuals", "Masked Residuals"])
                good = full_mask if np.any(full_mask) else np.ones_like(full_mask)
                flux_limits.extend([flux.min(), flux.max(), forward[good].min(), forward[good].max()])
                residual_limits.extend([residuals[good].min(), residuals[good].max()])
            else:
                ax[0].plot(wavelengths, forward, color='C0', linewidth=1, label='Forward Model Fit')
                ax[1].plot(wavelengths, residuals, color='C0', linewidth=1, label='Residuals')
            if show_linelist:
                colors = "green"
                if self.data.profile_groups is not None:
                    groups = self.data.profile_groups[np.searchsorted(self.data.ll_mask, ll_mask)]
                    colors = [f"C{group}" for group in groups[in_order]]
                ax[0].vlines(ll_wl, ll_tops, ymax=ll_depths, color=colors,
                             linewidth=1, label='Line List', alpha=0.5)

        if show_masking:
            for axis, limits in zip(ax, (flux_limits, residual_limits)):
                ymin, ymax = min(limits), max(limits)
                padding = (ymax - ymin) * 0.1 or 0.01
                axis.set_ylim(ymin - padding, ymax + padding)
        handles, names = ax[0].get_legend_handles_labels()
        legend = dict(zip(names, handles))
        ax[0].legend(legend.values(), legend.keys())

        if return_fig:
            return fig, ax
        utils.show_or_save(plt, self.config.figure_dir, "forward_model.png", output_dir=self.config.dir)

    @_require_sampler
    def autocorrelation(
        self,
        sampler        : EnsembleSampler|None = None,
        n_grid         : IntLike              = 12,
        c              : float                = 5.0,
        return_fig     : bool                 = False,
        subplot_kwargs : dict|None            = None,
        min_steps      : IntLike              = 100
        ) -> None | tuple:
        """
        Plot estimated integrated autocorrelation time as a function of chain length.

        From the emcee docs:
        - For several prefixes of the chain, estimate tau with Sokal windowing.
        - Plot tau(N) and the reference line tau = N/50.

        Parameters
        ----------
        sampler : :py:class:`emcee.EnsembleSampler` | None, optional
            Optionally provide a different sampler to plot from, otherwise,
            takes the sampler from the Result object, by default None
        n_grid : :py:type:`IntLike`, optional
            Number of N values (prefix lengths) to evaluate, by default 12.
        c : float, optional
            Sokal window constant, by default 5.0.
        return_fig : bool, optional
            Whether to return the figure and axes objects, by default False
        subplot_kwargs : dict | None, optional
            Keyword arguments to be passed to plt.subplots(). Allows label overrides, by default None
        min_steps : :py:type:`IntLike`, optional
            Minimum number of post-burnin samples required to attempt autocorrelation estimation, by default 100
            If you decrease this, you may get unreliable estimates or errors from the autocorrelation time estimation.

        Returns
        ----------
        If return_fig is True, returns a tuple (fig, ax) of the figure and axes objects containing 
        the plot. Otherwise, displays the plot and returns None.
        """

        chain = self.sampler.get_chain()  # (nsteps, nwalkers, ndim)
        nsteps, nwalkers, ndim = chain.shape

        if nsteps < min_steps:
            raise ACIDInputError("Not enough post-burnin samples to estimate autocorrelation from the min_steps. " \
            "Increase the number of steps or decrease min_steps.")
        
        Ns = np.unique(np.exp(np.linspace(np.log(min_steps), np.log(nsteps), n_grid)).astype(int))
        Ns = Ns[Ns >= min_steps]  # Ensure we only consider N >= min_steps

        tau_estimates = {p: np.full(len(Ns), np.nan, dtype=float) for p in self.default_params}

        # Estimate taus
        for i, n in enumerate(Ns):
            for p in self.default_params:
                y = chain[:n, :, p].T
                tau_estimates[p][i] = utils.autocorr_new(y, c=c)

        subplot_kwargs = utils.set_dict_defaults(subplot_kwargs, {"figsize": (10, 6)})

        fig, ax = plt.subplots(**subplot_kwargs)

        for i, (label, p) in enumerate(zip(self.default_param_labels, self.default_params)):
            ax.loglog(Ns, tau_estimates[p], "o-", label=f"{label}")

        # Reference line tau = N/50
        ax.loglog(Ns, Ns / 50.0, "--", label=r"$\tau = N/50$")

        ax.set_xlabel("Number of post-burnin samples per walker (N)")
        ax.set_ylabel(r"Estimated integrated autocorrelation time $\tau$")
        ax.set_title("Autocorrelation time estimates vs chain length")
        if i < 10: # too many labels make the plot ugly
            ax.legend(loc="upper left")
        ax.grid(True, which="both")

        if return_fig:
            return fig, ax
        utils.show_or_save(plt, self.config.figure_dir, "autocorrelation_time.png", output_dir=self.config.dir)

        return

    @_require_sampler
    def acf(
        self,
        sampler        : EnsembleSampler|None = None,
        max_lag        : IntLike|None         = None,
        return_fig     : bool                 = False,
        subplot_kwargs : dict|None            = None,
        ) -> None | tuple:
        """
        Plot the autocorrelation function (ACF) for each parameter, averaged across walkers.
        
        Parameters
        ----------
        sampler : :py:class:`emcee.EnsembleSampler`, optional
            Optionally provide a different sampler to plot from, otherwise,
            takes the sampler from the Result object, by default None
        max_lag : :py:type:`IntLike`, optional
            Maximum lag to plot, by default None (plots up to min(5000, nsteps-1))
        return_fig : bool, optional
            Whether to return the figure and axes objects, by default False
        subplot_kwargs : dict, optional
            Keyword arguments to be passed to plt.subplots(). Allows label overrides, by default None

        Returns
        -------
        If return_fig is True, returns a tuple (fig, ax) of the figure and axes objects containing 
        the plot. Otherwise, displays the plot and returns None.
        """
        chain = self.sampler.get_chain() 
        nsteps, nwalkers, ndim = chain.shape

        subplot_kwargs = {} if subplot_kwargs is None else dict(subplot_kwargs)
        subplot_kwargs = utils.set_dict_defaults(subplot_kwargs, {"figsize": (10, 5)})

        fig, ax = plt.subplots(**subplot_kwargs)

        for i, (param, label) in enumerate(zip(self.default_params, self.default_param_labels)):

            y = chain[:, :, param].T  # (nwalkers, nsteps)

            # Mean ACF across walkers
            f = np.zeros(nsteps)
            for w in range(nwalkers):
                f += utils.autocorr_func_1d(y[w], norm=True)
            f /= nwalkers

            if max_lag is None:
                max_lag = min(5_000, nsteps - 1)
            max_lag = int(max_lag)

            ax.plot(np.arange(max_lag + 1), f[: max_lag + 1], label=f"{label}")
        
        ax.set_xlabel("Lag (steps)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"Mean ACF across walkers")
        ax.set_xscale("log")
        ax.grid(True)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        if i < 10: # too many labels make the plot ugly
            ax.legend()

        if return_fig:
            return fig, ax
        utils.show_or_save(plt, self.config.figure_dir, "autocorrelation_function.png", output_dir=self.config.dir)

    def initiate_sampler(self, sampler:EnsembleSampler|Sampler|None, _method_name=None) -> None: # type:ignore
        """
        Initiates the sampler attribute from an external sampler.

        Parameters
        ----------
        sampler : :py:class:`emcee.EnsembleSampler` or object, optional
            An emcee EnsembleSampler object or a compatible sampler object to set as the sampler attribute.
        _method_name : str, optional
            Internal parameter used to track which method is calling initiate_sampler, for error messages. 
            Not intended for user input, by default None.
        """
        if self.sampler_initialized and sampler is None and self.sampler is not None:
            return # reuse initialization only while the sampler is still available
        self.sampler = sampler if sampler is not None else self.sampler
        if self.sampler is None:
            if _method_name is not None:
                error_msg = f"Cannot run {_method_name} without a sampler, please pass in a sampler to the method or during initialisation."
            else:
                error_msg = "Cannot initiate sampler without a sampler stored in the instance or passed as a parameter, please pass in a sampler "
            raise ACIDStateError(error_msg)

        self.dynesty = Sampler is not None and isinstance(self.sampler, Sampler)
        if self.dynesty:
            a=ord('a')
            alph=[chr(i) for i in range(a,a+26)]
            poly_labels = [alph[i] if i < len(alph) else f"c{i}" for i in range(self.data.ndim)]
            self.default_param_labels = poly_labels
            self.default_params = None
            self.sampler_initialized = True
            return

        # Calculate autocorr time, burnin, thin
        # Suppress output from get_autocorr_time call
        with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull), \
            contextlib.redirect_stderr(devnull):
            self.tau = self.sampler.get_autocorr_time(quiet=True)

        self.converged = True
        if self.data.nsteps < 50 * np.max(self.tau):
            self.converged = False
            warnings.warn(
                "The number of MCMC steps is less than 50 times the maximum autocorrelation " \
                "time.\n The sampler may not have converged. Consider running more steps or checking " \
                f"the walker plots.\n The max autocorrelation time is {np.max(self.tau):.2f}, therefore " \
                f"the minimum number of steps should be roughly {int(50 * np.max(self.tau))}.\n Disabling burnin " \
                f"from autocorrelation time, instead using burnin=steps-1000",
                ACIDConvergenceWarning, stacklevel=2
            )

        try:
            self.thin = int(np.min(self.tau)/5)
            if self.converged:
                self.burnin = int(3 * np.max(self.tau))
            else:
                self.burnin = self.data.nsteps - 1000 # just the last 1000 steps
        except:
            warnings.warn(
                "Warning: Could not compute autocorrelation time for burnin and thinning.\n This is likely" \
                " due to all posterior samples being rejected (possibly by prior constraints).\n The resulting profile is likely" \
                " wrong. Try Result.plot_walkers() to see the issue.\nSetting defaults: burnin=nsteps-1000, and thin=1.",
                ACIDConvergenceWarning, stacklevel=2
            )
            self.burnin = self.data.nsteps - 1000 # just the last 1000 steps
            self.thin = 1
        
        self.burnin = int(np.clip(self.burnin, 0, self.data.nsteps-1)) # ensure burnin is at least 0 and less than total steps
        self.thin = int(np.clip(self.thin, 1, self.data.nsteps-1)) # ensure thin is at least 1, and not clipping to nsteps

        # Below is used for the parameters for the walker and corner plots
        n_poly_params = len(self.data.poly_coeffs["masked"])
        poly_params = np.arange(-1, -n_poly_params-1, -1).tolist()
        
        # Generates labels for the polynomial coefficients, starting from 'a' for the highest order term, and going backwards in the alphabet.
        a=ord('a')
        alph=[chr(i) for i in range(a,a+26)]
        poly_labels = [alph[i] if i < len(alph) else f"c{i}" for i in range(n_poly_params)]
        
        samples = self.sampler.get_chain(thin=self.thin, discard=self.burnin)
        if not self.config.deterministic_profile:
            max_profile_idx = np.argmax(samples[:,:,:-n_poly_params].mean(axis=(0,1)))
            poly_params.extend([-5, max_profile_idx, 1])
            poly_labels.extend(["$Z_{-1}$", "$Z_{max}$", "$Z_0$"])

        self.default_params = poly_params
        self.default_param_labels = poly_labels

        self.sampler_initialized = True

    @property
    def sampler(self) -> EnsembleSampler|Sampler|None: # type:ignore
        """Returns the sampler attribute, by default is None if not saved."""
        return self.data.sampler

    @sampler.setter
    def sampler(self, value: EnsembleSampler|Sampler|None) -> None: # type:ignore
        """Sets the sampler in the data class."""
        self.data.sampler = value

