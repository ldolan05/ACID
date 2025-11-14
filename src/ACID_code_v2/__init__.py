from .ACID import ACID, run_ACID, run_ACID_HARPS
from .result import Result
from .utils import scale_spectra, calc_deltav
__all__ = ['ACID', 'run_ACID', 'run_ACID_HARPS', 'calc_deltav', 'Result', 'scale_spectra']

def _reload_all():
    # Reloads all submodules of the ACID_code_v2 package. Only useful for development purposes when using ipython.
    import importlib, sys
    pkg = __name__  # 'ACID_code_v2'

    # reload submodules
    for name, module in list(sys.modules.items()):
        if name.startswith(pkg + ".") and module is not None:
            importlib.reload(module)

    # reload top-level package last
    importlib.reload(__import__(pkg))