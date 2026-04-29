import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def simple_local_maxima(x, y, vicinity=7, include_edges=False):
    x = np.asarray(x)
    y = np.asarray(y)

    w = 2 * vicinity + 1

    if len(y) < w:
        idx = np.array([], dtype=int)
    else:
        windows = sliding_window_view(y, w)
        idx = (
            np.flatnonzero(
                y[vicinity:len(y) - vicinity] == windows.max(axis=1)
            )
            + vicinity
        )

    if include_edges:
        idx = np.unique(np.r_[0, idx, len(y) - 1])

    return x[idx], y[idx], idx

def rolling_pin_anchors(wave, flux, R):
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]

    if np.isscalar(R):
        radius = np.full(len(wave), float(R))
    else:
        radius = np.asarray(R, dtype=float)[order]

    keep = [0]
    j = 0
    n = len(wave)

    while n - j > 3:
        par_R = float(radius[j])

        while True:
            stop = np.searchsorted(
                wave,
                wave[j] + 2.0 * par_R,
                side="left"
            )

            cand = np.arange(j + 1, stop)

            if cand.size:
                dx = wave[cand] - wave[j]
                dy = flux[cand] - flux[j]
                c = np.hypot(dx, dy)

                mask = (dx > 0.0) & (c < 2.0 * par_R)

                if np.any(mask):
                    cand = cand[mask]
                    dx = dx[mask]
                    dy = dy[mask]
                    c = c[mask]
                    break

            par_R *= 1.5

        h = np.sqrt(par_R * par_R - 0.25 * c * c)

        cx = wave[j] + 0.5 * dx - h / c * dy
        cy = flux[j] + 0.5 * dy + h / c * dx

        theta = np.where(
            cy >= flux[j],
            -np.arccos((cx - wave[j]) / par_R) + np.pi,
            -np.arcsin((cy - flux[j]) / par_R) + np.pi,
        )

        j = cand[np.argmin(theta)]
        keep.append(j)

    keep = np.asarray(keep, dtype=int)

    return wave[keep], flux[keep], keep

def model(grid, spectrum, R, vicinity=7, y_scale=None):
    grid = np.asarray(grid, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)

    order = np.argsort(grid)
    grid_sorted = grid[order]
    spectrum_sorted = spectrum[order]

    if y_scale is None:
        y_scale = np.ptp(spectrum_sorted) / np.ptp(grid_sorted)

    scaled_flux = spectrum_sorted / y_scale

    max_wave, max_flux, _ = simple_local_maxima(
        grid_sorted,
        scaled_flux,
        vicinity=vicinity,
        include_edges=False
    )

    anchor_wave, anchor_flux_scaled, _ = rolling_pin_anchors(
        max_wave,
        max_flux,
        R=R
    )

    continuum_scaled = np.interp(
        grid_sorted,
        anchor_wave,
        anchor_flux_scaled
    )

    continuum_scaled[grid_sorted < anchor_wave[0]] = anchor_flux_scaled[0]
    continuum_scaled[grid_sorted > anchor_wave[-1]] = anchor_flux_scaled[-1]

    continuum_sorted = continuum_scaled * y_scale
    anchor_flux = anchor_flux_scaled * y_scale

    continuum = np.empty_like(continuum_sorted)
    continuum[order] = continuum_sorted

    return continuum, anchor_wave, anchor_flux
