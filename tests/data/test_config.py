#%%
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ACID_code.diagnostics.errors import ACIDInputError
from ACID_code import Config


def test_config_priorities_properties_and_environment(monkeypatch):
    # Low-priority values should fill gaps, while high-priority values win conflicts.
    config = Config(verbose="off", order=1)
    config.update_lowpri(order=2, poly_ord=4)
    config.update_hipri(order=3)

    assert config.verbose == 0
    assert config.order == 3
    assert config.poly_ord == 4
    verbose_config = Config()
    verbose_config.update_lowpri(verbose="low")
    verbose_config.update_hipri(verbose="high")
    verbose_config.update_lowpri(verbose="off")
    assert verbose_config.verbose == 3
    # Invalid configuration and misplaced Data attributes should fail clearly.
    with pytest.raises(ACIDInputError):
        config.update_hipri(not_a_setting=True)
    with pytest.raises(ACIDInputError):
        config.linelist = []
    with pytest.raises(ACIDInputError):
        config.not_a_setting = True
    config.order = 4
    config.order = None
    assert config.order == 4

    # Environment overrides are intentionally evaluated on attribute access.
    monkeypatch.setenv("_ACID_CONFIG", '{"order": 99}')
    assert config.order == 99


def test_config_dictionary_views_repr_and_verbose_validation(capsys):
    # The compact dictionary contains explicit values; the full view also resolves defaults.
    config = Config(order=7, verbose="medium")
    compact = config.to_dict()
    complete = config.to_full_dict()

    assert compact["order"] == complete["order"] == 7
    assert complete["poly_ord"] == Config.defaults["poly_ord"]
    assert config.pix_chunk == complete["pix_chunk"] == 50
    assert "order: 7" in repr(config)

    # Defaults are printable for interactive inspection, and invalid verbose values fail early.
    Config.print_defaults(use_logger=False)
    assert "poly_ord" in capsys.readouterr().out
    for invalid_verbose in (-1, 5):
        with pytest.raises(ValueError, match="between 0 and 4"):
            config.verbose = invalid_verbose
    with pytest.raises(ValueError, match="not recognised"):
        config.verbose = "loud"
    config.verbose = True
    assert config.verbose == 2
    config.verbose = "low"
    config.verbose = None
    assert config.verbose == 1


def test_config_sampler_path_is_non_destructive_and_sampler_type_is_validated(tmp_path):
    sampler_path = tmp_path / "sampler.h5"
    sampler_path.write_bytes(b"existing sampler")

    config = Config(sampler_path=str(sampler_path))

    assert sampler_path.read_bytes() == b"existing sampler"
    assert config.sampler_path == str(sampler_path)
    with pytest.raises(ValueError, match="sampler_type"):
        Config(sampler_type="nested")


@pytest.mark.parametrize("path_name", ["save_path", "sampler_path", "figure_dir"])
@pytest.mark.parametrize("disabled", ["None", "False", "OFF", "no", "n", "0"])
def test_config_path_disabling_strings_survive_round_trip(tmp_path, monkeypatch, path_name, disabled):
    monkeypatch.chdir(tmp_path)
    config = Config(**{path_name: disabled}, verbose=0)
    assert getattr(config, path_name) is None
    assert not (tmp_path / disabled).exists()

    config.dir = str(tmp_path / "output")
    assert getattr(config, path_name) is None
    restored = Config(**config.to_dict())
    assert getattr(restored, path_name) is None
    # Python None still means "leave unchanged", rather than a disabling string.
    setattr(restored, path_name, None)
    assert getattr(restored, path_name) is None
    restored.dir = str(tmp_path / "moved")
    assert getattr(restored, path_name) is None
    if path_name == "figure_dir":
        assert not (tmp_path / "output" / "figures").exists()
        assert not (tmp_path / "moved" / "figures").exists()



if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--tb=auto"]))
