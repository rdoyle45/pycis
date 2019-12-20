import numpy as np
import scipy.signal

# dictionary containing the window functions available, add your own.
fns = {'hanning': scipy.signal.hanning,
      'blackmanharris': scipy.signal.blackmanharris,
      'tukey': scipy.signal.windows.tukey}


def window(rfft_length, nfringes, window_width=None, fn='tukey', width_factor=1., alpha=0.5):
    """ Generate a filters window for isolating the carrier (fringe) frequency. 
    
    :param rfft_length: length of the real FFT to be filtered
    :type rfft_length: int.
    :param nfringes: The carrier (fringe) frequency to be demodulated, in units of cycles per sequence -- approximately the number of fringes present in the image.
    :type nfringes: int.
    :param fn: Specifies the window function to use, see the 'fns' dict at the top of the script for the 
    :type fn: str.
    :param width_factor: a multiplicative factor determining the width of the filters, multiplies nfringes, which is found to work reasonably well.
    :type width_factor: float
    
    :return: generated window as an array.
    """

    assert fn in fns

    if window_width == None:
        window_width = int(nfringes * width_factor)

    pre_zeros = [0] * int(nfringes - window_width / 2)

    if fn == 'tukey':
        fn = fns[fn]

        window_fn = fn(window_width, alpha=alpha)
    else:
        fn = fns[fn]
        window_fn = fn(window_width)

    post_zeros = [0] * (rfft_length - window_width - int(nfringes - window_width / 2 - 1))

    return np.concatenate((pre_zeros, window_fn, post_zeros))[:rfft_length]


if __name__ == '__main__':

    alphas = [0, 0.25, 0.5, 0.75, 1.0]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for a in alphas:
        win = window(1000, 100, fn='tukey', width_factor=1, alpha=a)
        print(win)
        ax.plot(win)

    plt.show(block=True)


