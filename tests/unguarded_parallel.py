from pathlib import Path
import sys

from astropy.io import fits
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

counter_path = Path(sys.argv[1])
run_count = int(counter_path.read_text()) + 1 if counter_path.exists() else 1
counter_path.write_text(str(run_count))
if run_count > 1:
    raise RuntimeError("The main script was imported by a worker process.")

from ACID_code import Acid

with fits.open(PROJECT_ROOT / "data" / "sample_spec_1.fits") as spectrum:
    wavelengths = spectrum[0].data
    flux = spectrum[1].data
    errors = spectrum[2].data
    sn = spectrum[3].data

acid = Acid(velocities=np.arange(-25, 25, 5.0),
            linelist=str(PROJECT_ROOT / "data" / "linelist.txt"), seed=1)
acid.ACID(wavelengths, flux, errors, sn, run_mcmc=False,
          parallel=True, cores=2, verbose=0)
acid.run_mcmc(1, state=acid.data.initial_state)
