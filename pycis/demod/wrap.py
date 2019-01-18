import numpy as np

units_period = {'rad': 2 * np.pi, 'waves': 1, 'fringes': 1}


def wrap(vals, units='rad'):
    """ Wrap values within (- pi, pi] radian interval.

    :param vals: input phase
    :type vals: array_like
    
    :param units: Units of phase (accepted: 'rad', 'waves', 'fringes').
    :type units: str
    
    :return: wrapped values (radians).
    """

    assert units in units_period
    period = units_period[units]

    return (vals + period / 2) % period - (period / 2)