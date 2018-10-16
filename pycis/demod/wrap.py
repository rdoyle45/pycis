import numpy as np
from copy import deepcopy

units_period = {'rad': 2 * np.pi, 'waves': 1, 'fringes': 1}

def wrap(phase, units='rad'):
    """ Wrap phase between [-pi, pi] radians.

    :param phase: Input phase in radians, 1-D or 2-D.
    :type phase: array_like
    :param units: Units of phase (accepted: 'rad', 'waves', 'fringes').
    :type units: str
    
    :return: Wrapped phase (radians).
    """

    assert units in units_period
    period = units_period[units]

    phase = deepcopy(phase)
    return (phase + period / 2) % period - (period / 2)