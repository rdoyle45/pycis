# CIS
# fit_functions

# store for commonly used fit functions
# jsallcock
# created: 28/09/2017

import matplotlib as mpl
import numpy as np
import scipy as sp
import scipy.io as sio
import os
import glob

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import pycis



def phase_shift_wl_poly(wl_shift, theta):
    """ nth order polynomial describing relationship between wavelength * phase shift and wavelength shift.
    
    :param wl_shift: units (m)
    :param theta: polynomial coefficients
    
    :return: phase_shift, units (fringes) 
    """

    group_delay_0 = theta[0]
    phase_shift_wl = - group_delay_0 * wl_shift

    poly_order = len(theta)

    if poly_order > 1:
        for i in range(2, poly_order+1):
            phase_shift_wl += theta[i - 1] * wl_shift ** (i)

    return phase_shift_wl

def poly1(x, a):
    return a * x

def poly2(x, a, b):
    return a * x + b * x ** 2

def poly3(x, a, b, c):
    return a * x + b * x ** 2 + c * x ** 3


def phase_shift_wl_poly_std(wl_shift, theta_std):
    """ from the uncertainty in the fit parameters theta at a wl_shift, return the uncertainty in the phase shift * 
    wavelength.

    :param wl_shift: units (m)
    :param theta_std: polynomial coefficient uncertainties.

    :return: phase_shift, units (fringes) 
    """

    group_delay_0_std = theta_std[0]

    phase_shift_wl_var = (group_delay_0_std * wl_shift) ** 2

    poly_order = len(theta_std)

    if poly_order > 1:
        for i in range(1, poly_order):
            phase_shift_wl_var += (theta_std[i] * wl_shift ** (i + 1)) ** 2

    phase_shift_wl_std = np.sqrt(phase_shift_wl_var)

    return phase_shift_wl_std


# phase fitting:

def cauchy_2(x, a, b):
    return a + b * x ** -2


def cauchy_3(x, a, b, c):
    return a + b * x ** -2 + c * x ** -4


def cauchy_4(x, a, b, c, d):
    return a + b * x ** -2 + c * x ** -4 + d * x ** -6


def cauchy_phase_2(x, a, b):
    """ 2 term cauchy equation approximation to sellmeier dispersion.

    fits phase change [waves] divided by crystal thickness [m] """
    B = cauchy_2(x, a, b)
    return B / x - B[0] / x[0]


def cauchy_phase_3(x, a, b, c):
    """ 3 term cauchy equation approximation to sellmeier dispersion.
    
    fits phase change [waves] divided by crystal thickness [m] """

    B = cauchy_3(x, a, b, c)
    return B / x - B[0] / x[0]


def cauchy_phase_4(x, a, b, c, d):
    """ 3 term cauchy equation approximation to sellmeier dispersion.

    fits phase change [waves] divided by crystal thickness [m] """
    B = a + b * x ** -2 + c * x ** -4 + d * x ** -6
    return B / x - B[0] / x[0]

def linear_phase(x, a, b):
    B = linear(x, a, b)
    return B / x - B[0] / x[0]


def quadratic_phase(x, a, b, c):
    B = quadratic(x, a, b, c)
    return B / x - B[0] / x[0]


def cubic_phase(x, a, b, c, d):
    B = cubic(x, a, b, c, d)
    return B / x - B[0] / x[0]

def quartic_phase(x, a, b, c, d, e):
    B = quartic(x, a, b, c, d, e)
    return B / x - B[0] / x[0]

def group_delay_c2(wavelength, a, b):
    return a / wavelength + 3 * b / wavelength ** 3

def group_delay_c3(wavelength, a, b, c):
    return a / wavelength + 3 * b / wavelength ** 3 + 5 * c / wavelength ** 5
