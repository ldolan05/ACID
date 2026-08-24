#%%
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ACID_code import Data, LSD
from ACID_code.errors import LineListRangeError, SNCutError


def test_sparse_and_dense_alpha_agree(synthetic_spectrum):
    # The sparse path is the production default; the dense path is its legacy calculation.
    wavelengths, _, _, _, velocities, linelist = synthetic_spectrum
    sparse = LSD.calc_alpha(wavelengths, linelist["wavelengths"], linelist["depths"], velocities)
    dense = LSD.calc_alpha(wavelengths, linelist["wavelengths"], linelist["depths"], velocities, sparse=False)

    np.testing.assert_allclose(sparse, dense, atol=1e-12)


def test_lsd_recovers_known_profile(synthetic_spectrum):
    # A noiseless constructed spectrum gives an exact, small LSD regression case.
    wavelengths, flux, errors, sn, velocities, linelist = synthetic_spectrum
    lsd = LSD()
    lsd.run_LSD(wavelengths, flux, errors, sn, linelist=linelist, velocities=velocities)

    # LSD should recover both a finite profile and its original forward model.
    assert lsd.profile.shape == velocities.shape
    assert np.all(np.isfinite(lsd.profile_errors))
    np.testing.assert_allclose(lsd.forward_model, flux, atol=3e-3)


@pytest.mark.parametrize("od", [True, False])
def test_lsd_accepts_the_harps_order_in_both_flux_spaces(harps_order_40, od):
    """Cover the optical-depth and legacy linear-flux LSD paths on real data."""
    wavelengths, flux, errors, sn, velocities, linelist = harps_order_40
    # The same observed order supports both modern optical-depth and legacy flux modes.
    lsd = LSD(od=od)
    lsd.run_LSD(wavelengths, flux, errors, sn, linelist=linelist, velocities=velocities)

    assert lsd.profile.shape == velocities.shape
    assert np.all(np.isfinite(lsd.profile_errors))


def test_multi_profile_convolution_matches_flattened_alpha(synthetic_spectrum):
    # Grouped alpha blocks should be algebraically equivalent to their flattened form.
    wavelengths, _, _, _, velocities, linelist = synthetic_spectrum
    groups = np.array([0, 1])
    alpha, labels = LSD.calc_mp_alpha(wavelengths, velocities, linelist["wavelengths"],
                                      linelist["depths"], groups)
    profiles = np.vstack((np.linspace(0, 0.1, len(velocities)),
                          np.linspace(0.1, 0, len(velocities))))

    assert labels.tolist() == [0, 1]
    np.testing.assert_allclose(LSD.dot_alpha_and_profile(alpha, profiles),
                               LSD.flatten_alpha(alpha) @ profiles.ravel())


def test_profile_groups_follow_wavelength_and_sn_clipping_through_run_lsd():
    # Construct a spectrum from the one line that should survive both clipping stages.
    wavelengths = np.linspace(5000, 5010, 1001)
    velocities = np.array([-5.0, 0.0, 5.0])
    profile = np.array([0.02, 0.1, 0.02])
    alpha = LSD.calc_alpha(wavelengths, np.array([5003.0]), np.array([0.2]), velocities)
    flux = np.exp(-(alpha @ profile))
    errors = np.full_like(flux, 0.01)

    # Lines 0 and 3 lie outside the padded spectrum; line 2 fails the S/N depth cut.
    linelist = {"wavelengths": np.array([4990.0, 5003.0, 5007.0, 5020.0]),
                "depths": np.array([0.5, 0.2, 0.001, 0.3])}
    profile_groups = np.array([9, 0, 1, 9])
    lsd = LSD()
    lsd.run_LSD(wavelengths, flux, errors, 100.0, linelist=linelist,
                velocities=velocities, profile_groups=profile_groups)

    # The group array and stored line-list mask must identify the same surviving line.
    np.testing.assert_array_equal(lsd.profile_groups, [0])
    np.testing.assert_array_equal(lsd.data.profile_groups, [0])
    np.testing.assert_array_equal(lsd.ll_mask, [1])
    assert lsd.alpha.shape == (1, len(wavelengths), len(velocities))


def test_wavelength_and_sn_clippers_apply_identical_masks_to_groups():
    # Use distinctive group labels so accidental reordering or unmasked groups are obvious.
    line_wavelengths = np.array([4990.0, 5002.0, 5004.0, 5006.0, 5020.0])
    line_depths = np.array([0.9, 0.001, 0.02, 0.005, 0.8])
    groups = np.array([10, 11, 12, 13, 14])
    clipped = LSD.clip_wavelengths(np.array([5000.0, 5010.0]), line_wavelengths,
                                   line_depths, groups, pad=0)

    # Wavelength clipping retains the middle three entries and their matching labels.
    np.testing.assert_array_equal(clipped[0], [5002.0, 5004.0, 5006.0])
    np.testing.assert_array_equal(clipped[2], [11, 12, 13])

    # At S/N=100 the 0.001-depth line is removed from every parallel array.
    sn_clipped = LSD().sn_clip(*clipped[:2], 100.0, clipped[2])
    np.testing.assert_array_equal(sn_clipped[0], [5004.0, 5006.0])
    np.testing.assert_array_equal(sn_clipped[1], [0.02, 0.005])
    np.testing.assert_array_equal(sn_clipped[2], [12, 13])


def test_convolve_profile_uses_supplied_or_calculated_alpha(synthetic_spectrum):
    # Calculate one reference alpha, then verify the fast precomputed-alpha route.
    wavelengths, _, _, _, velocities, linelist = synthetic_spectrum
    alpha = LSD.calc_alpha(wavelengths, linelist["wavelengths"], linelist["depths"], velocities)
    profile = np.array([0.02, 0.1, 0.02])
    expected = alpha @ profile
    from_alpha = LSD.convolve_profile(profile, alpha=alpha)

    # The convenience route should calculate and optionally return that same alpha.
    calculated, returned_alpha = LSD.convolve_profile(
        profile, velocities=velocities, wavelengths=wavelengths,
        linelist_wavelengths=linelist["wavelengths"],
        linelist_depths=linelist["depths"], return_alpha=True,
    )
    np.testing.assert_allclose(from_alpha, expected)
    np.testing.assert_allclose(calculated, expected)
    np.testing.assert_allclose(returned_alpha, alpha)


def test_convolve_profile_calculates_grouped_alpha(synthetic_spectrum):
    # Two profiles and two line groups should produce one alpha block per group.
    wavelengths, _, _, _, velocities, linelist = synthetic_spectrum
    groups = np.array([0, 1])
    profiles = np.array([[0.02, 0.1, 0.02], [0.01, 0.05, 0.01]])
    expected_alpha, _ = LSD.calc_mp_alpha(wavelengths, velocities,
                                          linelist["wavelengths"], linelist["depths"], groups)
    expected = LSD.dot_alpha_and_profile(expected_alpha, profiles)

    # This public convenience call must retain, rather than overwrite, the grouped alpha.
    convolved, alpha = LSD.convolve_profile(
        profiles, profile_groups=groups, velocities=velocities, wavelengths=wavelengths,
        linelist_wavelengths=linelist["wavelengths"],
        linelist_depths=linelist["depths"], return_alpha=True,
    )
    np.testing.assert_allclose(alpha, expected_alpha)
    np.testing.assert_allclose(convolved, expected)


def test_convolve_profile_validates_inputs_and_dimensions(synthetic_spectrum):
    # Calculating alpha requires all four coordinate and line-list arrays.
    with pytest.raises(ValueError, match="If alpha is not input"):
        LSD.convolve_profile(np.ones(3))

    # Dot products reject unsupported alpha dimensions and mismatched profiles.
    with pytest.raises(ValueError, match="either 2D or 3D"):
        LSD.dot_alpha_and_profile(np.ones(3), np.ones(3))
    with pytest.raises(ValueError, match="requires profile shape"):
        LSD.dot_alpha_and_profile(np.ones((2, 4, 3)), np.ones((2, 2)))
    # The public annotation rejects a matrix before the method's own check runs.
    with pytest.raises(BeartypeCallHintParamViolation):
        LSD.flatten_alpha(np.ones((4, 3)))


def test_solve_z_supports_profile_error_and_covariance_return_modes(synthetic_spectrum):
    # Reuse the exact synthetic system so each return mode solves the same equations.
    wavelengths, flux, errors, _, velocities, linelist = synthetic_spectrum
    alpha = LSD.calc_alpha(wavelengths, linelist["wavelengths"], linelist["depths"], velocities)
    factor = LSD.calc_cholesky(alpha, errors)
    optical_depth = -np.log(flux)

    profile_only = LSD.solve_z(alpha, optical_depth, errors, factor, return_error=False)
    profile_error = LSD.solve_z(alpha, optical_depth, errors, factor)
    profile_covariance = LSD.solve_z(alpha, optical_depth, errors, factor, return_cov=True)

    np.testing.assert_allclose(profile_only, profile_error[0])
    np.testing.assert_allclose(profile_only, profile_covariance[0])
    np.testing.assert_allclose(profile_covariance[1] ** 2,
                               np.diag(profile_covariance[2]))


def test_depth_group_rules_assign_every_line_and_validate_constraints():
    # Explicit rules take the deepest lines first; remaining groups split shallower lines.
    depths = np.array([0.05, 0.8, 0.1, 0.6, 0.2, 0.4])
    rules = {"n_groups": 3, "min_lines": 2, "0": 0.5}
    groups = LSD.group_profs_by_depth(depths, rules)

    assert set(groups) == {0, 1, 2}
    assert np.all(groups[np.argsort(depths)[-2:]] == 0)
    assert np.all(groups >= 0)

    # Impossible group sizes, invalid labels, and increasing thresholds are separate errors.
    with pytest.raises(ValueError, match="Cannot make"):
        LSD.group_profs_by_depth(depths[:3], {"n_groups": 2, "min_lines": 2})
    with pytest.raises(ValueError, match="outside"):
        LSD.group_profs_by_depth(depths, {"n_groups": 2, "min_lines": 2, "2": 0.5})
    with pytest.raises(ValueError, match="decrease"):
        LSD.group_profs_by_depth(depths, {"n_groups": 2, "min_lines": 2,
                                          "0": 0.3, "1": 0.5})


def test_runlsd_and_store_populates_each_data_product(synthetic_spectrum):
    # Prepare the Data keys normally created immediately before an ACID LSD stage.
    wavelengths, flux, errors, sn, velocities, linelist = synthetic_spectrum
    data = Data()
    data.set_inputs(wavelengths, flux, errors, sn)
    data.linelist = linelist
    data.velocities = velocities
    data.wavelengths["test"] = wavelengths
    data.flux["test"] = flux
    data.errors["test"] = errors
    data.fitted_flux["test"] = flux
    data.fitted_errors["test"] = errors
    data.sn["test"] = sn
    data.continuum["test"] = np.ones_like(flux)

    # The class helper should store every downstream product under the requested key.
    lsd = LSD.runlsd_and_store(data, "test", return_cls=True)
    for mapping in (data.alpha, data.c_factor, data.forward_x, data.forward_y,
                    data.profile, data.residuals, data.ll_mask):
        assert "test" in mapping
    np.testing.assert_allclose(data.forward_y["test"], lsd.forward_model)


def test_alpha_rejects_invalid_velocity_grids(synthetic_spectrum):
    # A profile grid needs at least two evenly spaced velocity bins.
    wavelengths, _, _, _, _, linelist = synthetic_spectrum

    velocities = [np.array([0.0, 1.0]), np.array([0.0, 1.0, 3.0])]

    with pytest.raises(BeartypeCallHintParamViolation):
        LSD.calc_alpha(wavelengths, linelist["wavelengths"], linelist["depths"], velocities)


def test_lsd_reports_invalid_inputs(synthetic_spectrum):
    # Each public validation branch should raise its corresponding domain exception.
    wavelengths, flux, errors, sn, velocities, linelist = synthetic_spectrum

    with pytest.raises(ValueError, match="normalised"):
        LSD().run_LSD(wavelengths, flux * 2, errors, sn, linelist=linelist, velocities=velocities)
    with pytest.raises(ValueError, match="same shape"):
        LSD().run_LSD(wavelengths, flux[:-1], errors, sn, linelist=linelist, velocities=velocities)
    with pytest.raises(LineListRangeError):
        LSD().run_LSD(wavelengths, flux, errors, sn, linelist=[[6000], [0.2]], velocities=velocities)
    with pytest.raises(SNCutError):
        LSD().run_LSD(wavelengths, flux, errors, sn, linelist=[[5003], [0.001]], velocities=velocities)


def test_cholesky_solver_and_convolution_validation(synthetic_spectrum):
    # The Cholesky solver exposes profile uncertainties through the covariance diagonal.
    wavelengths, flux, errors, _, velocities, linelist = synthetic_spectrum
    alpha = LSD.calc_alpha(wavelengths, linelist["wavelengths"], linelist["depths"], velocities)
    c_factor = LSD.calc_cholesky(alpha, errors)
    profile, profile_error, covariance = LSD.solve_z(alpha, -np.log(flux), errors, c_factor, return_cov=True)

    assert covariance.shape == (len(velocities), len(velocities))
    np.testing.assert_allclose(profile_error ** 2, np.diag(covariance))
    with pytest.raises(ValueError, match="incompatible"):
        LSD.dot_alpha_and_profile(alpha, np.ones(2))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
