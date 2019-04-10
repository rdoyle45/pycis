import numpy as np

import pycis


class Spectrum:
    """
    spectrum of scene observed by the instrument

    """
    def __init__(self, wl, spec):
        self.wl = wl
        self.spec = spec

