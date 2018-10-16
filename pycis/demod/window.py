import numpy as np
import scipy.signal

# dictionary containing the window functions available, add your own.
fns = {'hanning': scipy.signal.hanning,
      'blackmanharris': scipy.signal.blackmanharris,
      'tukey': scipy.signal.tukey}


def window(rfft_length, nfringes, fn='tukey', width_factor=1.):
    """ Generate a filter window for isolating the carrier (fringe) frequency. 
    
    :param rfft_length: length of the real FFT to be filtered
    :type rfft_length: int.
    :param nfringes: The carrier (fringe) frequency to be demodulated, in units of cycles per sequence -- approximately the number of fringes present in the image.
    :type nfringes: int.
    :param fn: Specifies the window function to use, see the 'fns' dict at the top of the script for the 
    :type fn: str.
    :param width_factor: a multiplicative factor determining the width of the filter, multiplies nfringes, which is found to work reasonably well.
    :type width_factor: float
    
    :return: generated window as an array.
    """

    assert fn in fns
    fn = fns[fn]

    window_width = int(nfringes * width_factor)

    pre_zeros = [0] * int(nfringes - window_width / 2)

    if fn == 'tukey':
        window_fn = fn(window_width, alpha=0.7)
    else:
        window_fn = fn(window_width)

    post_zeros = [0] * (rfft_length - window_width - int(nfringes - window_width / 2 - 1))

    return np.concatenate((pre_zeros, window_fn, post_zeros))


