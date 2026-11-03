"""Check startup and deferred return types without an editable installation."""
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import textwrap


def test_package_import_and_return_types_without_editable_install(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    dependencies = sysconfig.get_paths()
    script = textwrap.dedent("""
        import sys
        from pathlib import Path

        # -I -S skips site initialization and editable-install .pth files.
        sys.path[:0] = sys.argv[1:]
        import ACID_code
        import numpy as np
        from ACID_code import Data, DataList, Plot, Profiles, Result

        assert Path(ACID_code.__file__).resolve().is_relative_to(Path(sys.argv[1]))
        data = Data()
        data.config.order = 0
        data.velocities = np.linspace(-5., 5., 21)
        flux = Profiles.gaussian_func(data.velocities, -.3, 0., 1.)
        data.profile["final"] = (flux, np.full_like(flux, .01), np.eye(len(flux)) * .0001)
        data.profiles = [data.profile["final"]]
        data.complete = True

        # Exercise the circular return hints after package initialization has finished.
        assert isinstance(data.plot, Plot)
        result = data.result
        assert isinstance(result, Result)
        assert isinstance(result.profiles, Profiles)
        orders = DataList.from_datalist(data, verbose=0)
        assert isinstance(orders.profiles, Profiles)
        assert isinstance(orders.results[0], Result)
    """)
    paths = list(dict.fromkeys([str(source), dependencies["purelib"], dependencies["platlib"]]))
    environment = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(tmp_path / "mpl"))
    result = subprocess.run([sys.executable, "-I", "-S", "-c", script, *paths],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
