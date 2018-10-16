import numpy as np


def find_nearest(array, value):
    """ Find the nearest LARGER array entry than input value. 
    
    Returns index of array of FIRST instance of nearest entry. Returns empty if no larger array entry than value exists.

    :param array: 
    :param value: 
    :return: 
    """
    resid = array-value
    if np.all(resid[:] <= 0): #There are no array entries above value
        idx = []
    else:
        resid[resid < 0] = np.max(array) + 1 # Disregard array entries that lie below value
        idx = np.argmin(resid)
    return idx
