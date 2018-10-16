import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import pycis as ps

indexing = 'xy'


def log_trapz(log_y, dx, axis=0):
    """Integrate along the given axis using the composite trapezoidal rule.
    
    Done in logspace to avoid numerical underflow. ASSUMES REGULAR INTERVALS.
    
    TODO: Generalise to higher dimensions?

    Integrate `y` (`x`) along given axis.

    Parameters
    ----------
    log_y : array_like
        Log input array to integrate.
    dx : scalar
        The spacing between sample points when `x` is None.
    axis: 0 or 1.
        
    Returns: log of the integral. """

    ndim = np.ndim(log_y)
    shape = np.shape(log_y)

    if ndim == 1:
        ak = np.ones(len(log_y))
        ak[0] = 0.5
        log_int = np.log(dx) + scipy.misc.logsumexp(np.log(ak) + log_y)
    elif ndim == 2:
        if axis == 0:
            N = shape[1]
            K = shape[axis]

            ak = np.ones(N)
            ak[0] = 0.5

            log_int = np.zeros(K)

            for k in range(K):
                log_int[k] = np.log(dx) + scipy.misc.logsumexp(np.log(ak) + log_y[k, :])

        elif axis == 1:
            N = shape[0]
            K = shape[axis]

            ak = np.ones(N)
            ak[0] = 0.5

            log_int = np.zeros(K)

            for k in range(K):
                log_int[k] = np.log(dx) + scipy.misc.logsumexp(np.log(ak) + log_y[:, k])
        else:
            raise Exception('axis must be either 0 or 1 (for now!)')
    else:
        raise Exception('log_y must be a 1D or 2D array (for now!)')

    return log_int


def check_1d():

    a = 5
    b = 42
    c = 2

    N = 1000
    x = np.linspace(0, 100, N)
    dx = x[1] - x[0]
    ak = np.ones(N)
    ak[0] = 0.5

    f = gaussian(x, a, b, c)
    logf = np.log(f)

    # integrate using trapz

    integral_trapz = np.trapz(f, x)

    # integrate using lse_int


    integral_lse_int_1 = np.exp(np.log(dx) + scipy.misc.logsumexp(np.log(ak) + logf))
    integral_lse_int_2 = log_trapz(logf, dx)

    print(integral_trapz, integral_lse_int_1, integral_lse_int_2)

    return


def check_2d():

    NX = 100
    NY = 80
    x = np.linspace(0, 70, NX)
    y = np.linspace(0, 100, NY)

    xx, yy = np.meshgrid(x, y, indexing=indexing)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    f = gaussian2d(x, y, 4, 30, 23, 40, 5)
    logf = np.log(f)



    ### TRAPZ

    # integrate using trapz, axis = 0:
    integral_trapz_0 = np.zeros(NY)

    for i in range(NY):
        integral_trapz_0[i] = np.trapz(f[i, :], dx=dx)

    # integrate using trapz, axis = 1:
    integral_trapz_1 = np.zeros(NX)

    for i in range(NX):
        integral_trapz_1[i] = np.trapz(f[:, i], dx=dy)

    ### LSE INT

    integral_lse_0 = np.exp(log_trapz(logf, dx=dx, axis=0))
    integral_lse_1 = np.exp(log_trapz(logf, dx=dy, axis=1))

    print(sum(integral_trapz_0), sum(integral_trapz_1), sum(integral_lse_0), sum(integral_lse_1))

    plt.figure()
    plt.pcolor(xx, yy, f)
    plt.show()

    return


def gaussian(x, a, b, c):
    """ Gaussian"""

    return a * np.exp(-(((x - b) ** 2) / (2 * c ** 2)))

def gaussian2d(x, y, amp, a0, a_std, b0, b_std):

    xx, yy = np.meshgrid(x, y, indexing=indexing)
    return amp * np.exp(- 0.5 * (((xx - a0) / a_std) ** 2 + ((yy - b0) / b_std) ** 2))


if __name__ == '__main__':
    check_2d()