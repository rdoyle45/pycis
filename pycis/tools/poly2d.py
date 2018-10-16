# Fit 2D, 3rd order polynomial to the phase calibration frame using least squares.
# Inputs are synth_data and boolean mask, of same dimensions. Pixels failing to meet the noise criteria specified by the mask
# are weighted with large fitting uncertainties and so are effectvely taken out of consideration.

# jsallcock 11/16

import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt


def poly2d(data_in=None, mask=None, disp=False):
    if data_in is None:
        # load default calib frame
        data_in = np.load('/Users/jsallcock/Documents/physics/phd/code/pycis/synth_data/phi_calib/29657/frame10_phi.npy')
    if mask is None:
        # create default test mask (simple central square)
        mask = np.zeros_like(data_in, dtype=bool)
        mask[412:612, 412:612] = 1

    y_pixels = np.size(data_in, 0)
    x_pixels = np.size(data_in, 1)
    y = np.linspace(0, y_pixels - 1, y_pixels)
    x = np.linspace(0, x_pixels - 1, x_pixels)
    x, y = np.meshgrid(x, y, copy=False)

    initial_guess = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    sigma = np.ones_like(data_in)
    sigma[~mask] = 100000000  # arbitrary large number

    # Flatten curve_fit() inputs:
    sigma_input = sigma.ravel()
    data_in_input = data_in.ravel()

    popt, pcov = scipy.optimize.curve_fit(poly3, (x, y), data_in_input, sigma = sigma_input, p0=initial_guess)

    data_fitted = poly3((x, y), *popt)
    data_fitted = data_fitted.reshape(y_pixels, x_pixels)

    if disp:
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(data_in, 'plasma', interpolation='none')
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.imshow(data_fitted, 'plasma', interpolation='none')
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.imshow(abs(data_in - data_fitted), 'plasma', interpolation='none')
        plt.colorbar()

    return data_fitted


def poly3(x_y, coeff1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7, coeff8, coeff9):
    x, y = x_y
    poly3_out = coeff1*(x*0+1) +\
        coeff2*x +\
        coeff3*y +\
        coeff4*(x**2) +\
        coeff5*(x**2)*y +\
        coeff6*(x**2)*(y**2) +\
        coeff7*y**2 +\
        coeff8*x*(y**2) +\
        coeff9*x*y

    poly3_out = poly3_out.ravel()
    return poly3_out


