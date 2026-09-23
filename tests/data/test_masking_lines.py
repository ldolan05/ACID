#%%
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ACID_code import Config, MaskingLines


def test_masking_lines_accepts_compact_inputs_and_masks_grid():
    # A compact dictionary should expand its default width for each line.
    lines = MaskingLines({"telluric": {"default_width": 100, "lines": [5000.0]}})
    grid = np.array([4990.0, 5000.0, 5010.0])

    # Both the combined and named-mask interfaces should describe the same region.
    assert lines.get_1d_mask_on_grid(grid).tolist() == [False, True, False]
    masks = lines.get_masks(grid)
    assert isinstance(masks, list) and len(masks) == 1
    assert list(lines.get_masks(grid, with_names=True)) == ["telluric"]
    with pytest.raises(ValueError):
        MaskingLines({"bad": {"lines": [5000.0]}})


def test_masking_line_plot_has_one_labelled_series_per_group():
    # Give each group a distinct line and width so plot components are unambiguous.
    config = Config(masking_lines={
        "narrow": {"default_width": 100, "lines": [5000.0]},
        "wide": {"default_width": 500, "lines": [5100.0]},
    })
    fig, ax = config.plot_masking_lines(return_fig=True)

    assert len(ax.lines) == 2
    assert [line.get_label() for line in ax.lines] == ["Narrow line", "Wide line"]
    assert len(ax.patches) == 2
    plt.close(fig)


@pytest.mark.parametrize("line_input, expected_widths", [
    ([(5000.0, 100), (5010.0, 200)], [100, 200]),
    (np.array([[5000.0, 5010.0], [100, 200]]), [100, 200]),
    ([(5000.0,), (5010.0, 200)], [150, 200]),
])
def test_masking_lines_normalises_supported_width_formats(line_input, expected_widths):
    # Tuple, two-row array, and default-width forms all describe the same stored model.
    lines = MaskingLines({"test": {"default_width": 150, "lines": line_input}})

    np.testing.assert_array_equal(lines["test"]["widths"], expected_widths)
    named_masks = lines.get_masks(np.array([4990.0, 5000.0, 5010.0]), with_names=True)
    assert set(named_masks) == {"test"}


def test_masking_lines_rejects_empty_and_mismatched_definitions():
    # Invalid definitions should fail during construction, before they can mask spectra.
    with pytest.raises(ValueError, match="empty"):
        MaskingLines({"bad": {"default_width": 100, "lines": []}})
    with pytest.raises(ValueError, match="same"):
        MaskingLines({"bad": {"lines": [[5000.0, 5010.0], [100]]}})



if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--tb=auto"]))
