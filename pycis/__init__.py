import traceback
from . import paths

try:
    from .demod import *
except ImportError as e:
    print('--pycis ImportError: could not import pycis.demod:')
    print(e)

try:
    from .model import *
except ImportError as e:
    print('--pycis ImportError: could not import pycis.model:')
    print(traceback.format_exc())
    print(e)

try:
    from . import data
except ImportError as e:
    print('--pycis ImportError: could not import pycis.raw_data:')
    print(e)

try:
    from . import calib
except ImportError as e:
    print('--pycis ImportError: could not import pycis.calib:')
    print(e)

try:
    from . import tools
except ImportError as e:
    print('--pycis ImportError: could not import pycis.tools:')
    print(e)




