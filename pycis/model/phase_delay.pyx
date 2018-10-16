from __future__ import division
import cython
cimport cython
from pycis.model.bbo import bbo

import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, sqrt

import pycis


def uniaxial_crystal_3d(wl_axis, thickness, theta, omega, alpha=0):
    """
    
    phase delay due to uniaxial crystal across 2D grid of incidence and azimuthal angles, for a range of wavelengths
    
    There is a lot of repeated code in these functions that can be gotten rid of.
    
    :param wl_axis: 
    :param thickness: 
    :param theta: 
    :param omega: 
    :param alpha: 
    :return: 
    """

    img_dim = theta.shape
    wl_axis_length = len(wl_axis)

    reps_img = [wl_axis_length, 1, 1]
    reps_axis = [img_dim[0], img_dim[1], 1]

    biref_axis, n_e_axis, n_o_axis = pycis.model.bbo_array(wl_axis, 0) ### TODO: CHANGE THIS TO THE CONFIG DEFAULT

    wl_axis_cube = np.tile(wl_axis, reps_axis)
    n_e_axis_cube = np.tile(wl_axis, reps_axis)
    n_o_axis_cube = np.tile(wl_axis, reps_axis)

    theta_cube = np.moveaxis(np.tile(theta, reps_img), 0, -1)
    omega_cube = np.moveaxis(np.tile(omega, reps_img), 0, -1)

    term_1 = np.sqrt(n_o_axis_cube ** 2 - np.sin(theta_cube) ** 2)
    term_2 = (n_o_axis_cube ** 2 - n_e_axis_cube ** 2) * \
             (np.sin(alpha) * np.cos(alpha) * np.cos(omega_cube) * np.sin(theta_cube)) / \
             (n_e_axis_cube ** 2 * sin(alpha) ** 2 + n_o_axis_cube ** 2 * cos(alpha) ** 2)
    term_3 = - n_o_axis_cube * np.sqrt((n_e_axis_cube ** 2 * (n_e_axis_cube ** 2 * np.sin(alpha) ** 2 + n_o_axis_cube ** 2 * np.cos(alpha) ** 2)) -
                          ((n_e_axis_cube ** 2 - (n_e_axis_cube ** 2 - n_o_axis_cube ** 2) * np.cos(alpha) ** 2 * np.sin(omega_cube) ** 2) * np.sin(theta_cube) ** 2)) / \
             (n_e_axis_cube ** 2 * np.sin(alpha) ** 2 + n_o_axis_cube ** 2 * np.cos(alpha) ** 2)

    return (thickness / wl_axis_cube) * (term_1 + term_2 + term_3)


def savart_plate_3d(wl_axis, thickness, theta, omega):

    img_dim = theta.shape
    wl_axis_length = len(wl_axis)

    reps_img = [wl_axis_length, 1, 1]
    reps_axis = [img_dim[0], img_dim[1], 1]

    biref_axis, n_e_axis, n_o_axis = pycis.model.bbo_array(wl_axis, 0) ### TODO: CHANGE THIS TO THE CONFIG DEFAULT

    wl_axis_cube = np.tile(wl_axis, reps_axis)
    n_e_axis_cube = np.tile(wl_axis, reps_axis)
    n_o_axis_cube = np.tile(wl_axis, reps_axis)

    theta_cube = np.moveaxis(np.tile(theta, reps_img), 0, -1)
    omega_cube = np.moveaxis(np.tile(omega, reps_img), 0, -1)


    a_axis_cube = 1 / n_e_axis_cube
    b_axis_cube = 1 / n_o_axis_cube

    term_1 = ((a_axis_cube ** 2 - b_axis_cube ** 2 ) / (a_axis_cube ** 2 + b_axis_cube ** 2)) * (np.cos(omega_cube) + np.sin(omega_cube)) * np.sin(theta_cube)
    term_2 = ((a_axis_cube ** 2 - b_axis_cube ** 2) / (a_axis_cube ** 2 + b_axis_cube ** 2) ** (3 / 2)) * ((a_axis_cube ** 2) / np.sqrt(2)) * \
             (np.cos(omega_cube) ** 2 - np.sin(omega_cube) ** 2) * np.sin(theta_cube) ** 2

    return (thickness / (2 * wl_axis_cube)) * (term_1 + term_2)


def uniaxial_crystal_2D(wavelength, thickness, theta, omega, alpha=0):
    """
    Calculate phase delay due to uniaxial crystal across 2D grid of incidence and azimuthal angles.
    
    :param wavelength: wavelength in [m]
    :param n_e: extraordinary refractive index
    :param n_o: ordinary refractive index
    :param thickness: crystal thickness [m]
    :param theta: angle of incidence [rad]
    :param omega: angle of azimuth [rad]
    :param alpha: optic axis angle [rad]
    :return: phase [waves]
    """

    biref, n_e, n_o = pycis.model.bbo(wavelength, 0) ### TODO: CHANGE THIS TO THE CONFIG DEFAULT

    term_1 = np.sqrt(n_o ** 2 - np.sin(theta) ** 2)
    term_2 = (n_o ** 2 - n_e ** 2) * \
             (np.sin(alpha) * np.cos(alpha) * np.cos(omega) * np.sin(theta)) / \
             (n_e ** 2 * sin(alpha) ** 2 + n_o ** 2 * cos(alpha) ** 2)
    term_3 = - n_o * np.sqrt((n_e ** 2 * (n_e ** 2 * np.sin(alpha) ** 2 + n_o ** 2 * np.cos(alpha) ** 2)) -
                          ((n_e ** 2 - (n_e ** 2 - n_o ** 2) * np.cos(alpha) ** 2 * np.sin(omega) ** 2) * np.sin(theta) ** 2)) / \
             (n_e ** 2 * np.sin(alpha) ** 2 + n_o ** 2 * np.cos(alpha) ** 2)

    return (thickness / wavelength) * (term_1 + term_2 + term_3)



def savart_plate_2D(wavelength, thickness, theta, omega):
    """ Returns phase delay due to savart plate across 2D grid of incidence and azimuthal angles. 
    
    :param wavelength: wavelength in [m]
    :param n_e: extraordinary refractive index
    :param n_o: ordinary refractive index
    :param thickness: crystal thickness [m]
    :param theta: angle of incidence [rad]
    :param omega: angle of azimuth [rad]
    :return: phase [waves]
    
    """

    biref, n_e, n_o = pycis.model.bbo(wavelength, 0)

    a = 1 / n_e
    b = 1 / n_o

    term_1 = ((a ** 2 - b ** 2 ) / (a ** 2 + b ** 2)) * (np.cos(omega) + np.sin(omega)) * np.sin(theta)
    term_2 = ((a ** 2 - b ** 2) / (a ** 2 + b ** 2) ** (3 / 2)) * ((a ** 2) / np.sqrt(2)) * \
             (np.cos(omega) ** 2 - np.sin(omega) ** 2) * np.sin(theta) ** 2

    return (thickness / (2 * wavelength)) * (term_1 + term_2)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef double uniaxial_crystal(double wavelength, double n_e, double n_o, double L_wp, double theta, double omega, double alpha=0):
    """ Returns phase delay due to waveplate --at single point--. Complete expression is from Veiras et al. 2010.
    
    :param wavelength: wavelength in [m]
    :param n_e: extraordinary refractive index
    :param n_o: ordinary refractive index
    :param L_wp: crystal thickness [m]
    :param theta: angle of incidence [rad]
    :param omega: angle of azimuth [rad]
    :param alpha: optic axis angle [rad]
    :return: phase [waves]
    """

    cdef double term_1 = sqrt(n_o ** 2 - sin(theta) ** 2)
    cdef double term_2 = (n_o ** 2 - n_e ** 2) * \
                                 (sin(alpha) * cos(alpha) * cos(omega) * sin(theta)) / \
                         (n_e ** 2 * sin(alpha) ** 2 + n_o ** 2 * cos(alpha) ** 2)
    cdef double term_3 = - n_o * sqrt((n_e ** 2 * (n_e ** 2 * sin(alpha) ** 2 + n_o ** 2 * cos(alpha) ** 2)) -
                                      ((n_e ** 2 - (n_e ** 2 - n_o ** 2) * cos(alpha) ** 2 * sin(omega) ** 2) * sin(theta) ** 2)) / \
                         (n_e ** 2 * sin(alpha) ** 2 + n_o ** 2 * cos(alpha) ** 2)

    return - (L_wp / wavelength) * (term_1 + term_2 + term_3)  # negative sign added here by me to make consistent with the simple phase delay equation at zero-incidence, zero-alpha




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef double savart_plate(double wavelength, double n_e, double n_o, double L_sp, double theta, double omega):
    """ Returns phase delay due to savart plate --at single point--. Complete expression is from Francon & Mallick. 
    
    :param wavelength: wavelength in [m]
    :param n_e: extraordinary refractive index
    :param n_o: ordinary refractive index
    :param L_sp: crystal thickness [m]
    :param theta: angle of incidence [rad]
    :param omega: angle of azimuth [rad]
    :return: phase [waves]
    """

    cdef double a = 1 / n_e
    cdef double b = 1 / n_o

    cdef double term_1 = ((a ** 2 - b ** 2 ) / (a ** 2 + b ** 2)) * (cos(omega) + sin(omega)) * sin(theta)
    cdef double term_2 = ((a ** 2 - b ** 2) / (a ** 2 + b ** 2) ** (3 / 2)) * ((a ** 2) / sqrt(2)) * \
             (cos(omega) ** 2 - sin(omega) ** 2) * sin(theta) ** 2

    return (L_sp / (2 * wavelength)) * (term_1 + term_2)


