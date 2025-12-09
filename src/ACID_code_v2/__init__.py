from .acid import ACID, ACID_HARPS, Acid
from .lsd import LSD
from .result import Result
from .utils import scale_spectra, calc_deltav, guess_SNR
__all__ = ['ACID', 'ACID_HARPS', 'Acid', 'calc_deltav', 'LSD',
           'Result', 'scale_spectra', 'guess_SNR']

def _reload_all():
    # Reloads all submodules of the ACID_code_v2 package. 
    # Only useful for development purposes when using ipython.
    import importlib, sys
    pkg = __name__  # 'ACID_code_v2'

    # reload submodules
    for name, module in list(sys.modules.items()):
        if name.startswith(pkg + ".") and module is not None:
            importlib.reload(module)

    # reload top-level package last
    importlib.reload(__import__(pkg))