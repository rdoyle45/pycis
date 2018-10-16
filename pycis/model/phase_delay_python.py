
from pycis.model.bbo import bbo

import numpy as np

import pycis


def uniaxial_crystal_3d_python(wl_axis, thickness, theta, omega, alpha=0):
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

    biref_axis, n_e_axis, n_o_axis, _, _, _, _ = pycis.model.bbo_array(wl_axis, 0)  ### TODO: CHANGE THIS TO THE CONFIG DEFAULT

    wl_axis_cube = np.tile(wl_axis, reps_axis)
    n_e_axis_cube = np.tile(n_e_axis, reps_axis)
    n_o_axis_cube = np.tile(n_o_axis, reps_axis)

    theta_cube = np.moveaxis(np.tile(theta, reps_img), 0, -1)
    omega_cube = np.moveaxis(np.tile(omega, reps_img), 0, -1)

    term_1 = np.sqrt(n_o_axis_cube ** 2 - np.sin(theta_cube) ** 2)
    term_2 = (n_o_axis_cube ** 2 - n_e_axis_cube ** 2) * \
             (np.sin(alpha) * np.cos(alpha) * np.cos(omega_cube) * np.sin(theta_cube)) / \
             (n_e_axis_cube ** 2 * np.sin(alpha) ** 2 + n_o_axis_cube ** 2 * np.cos(alpha) ** 2)
    term_3 = - n_o_axis_cube * np.sqrt(
        (n_e_axis_cube ** 2 * (n_e_axis_cube ** 2 * np.sin(alpha) ** 2 + n_o_axis_cube ** 2 * np.cos(alpha) ** 2)) -
        ((n_e_axis_cube ** 2 - (n_e_axis_cube ** 2 - n_o_axis_cube ** 2) * np.cos(alpha) ** 2 * np.sin(
            omega_cube) ** 2) * np.sin(theta_cube) ** 2)) / \
             (n_e_axis_cube ** 2 * np.sin(alpha) ** 2 + n_o_axis_cube ** 2 * np.cos(alpha) ** 2)

    return (thickness / wl_axis_cube) * (term_1 + term_2 + term_3)


def savart_plate_3d_python(wl_axis, thickness, theta, omega):
    img_dim = theta.shape
    wl_axis_length = len(wl_axis)

    reps_img = [wl_axis_length, 1, 1]
    reps_axis = [img_dim[0], img_dim[1], 1]

    biref_axis, n_e_axis, n_o_axis, _, _, _, _ = pycis.model.bbo_array(wl_axis, 0)  ### TODO: CHANGE THIS TO THE CONFIG DEFAULT

    a_axis = 1 / n_e_axis
    b_axis = 1 / n_o_axis

    wl_axis_cube = np.tile(wl_axis, reps_axis)
    a_axis_cube = np.tile(a_axis, reps_axis)
    b_axis_cube = np.tile(b_axis, reps_axis)

    theta_cube = np.moveaxis(np.tile(theta, reps_img), 0, -1)
    omega_cube = np.moveaxis(np.tile(omega, reps_img), 0, -1)

    term_1 = ((a_axis_cube ** 2 - b_axis_cube ** 2) / (a_axis_cube ** 2 + b_axis_cube ** 2)) * (
    np.cos(omega_cube) + np.sin(omega_cube)) * np.sin(theta_cube)
    term_2 = ((a_axis_cube ** 2 - b_axis_cube ** 2) / (a_axis_cube ** 2 + b_axis_cube ** 2) ** (3 / 2)) * (
    (a_axis_cube ** 2) / np.sqrt(2)) * \
             (np.cos(omega_cube) ** 2 - np.sin(omega_cube) ** 2) * np.sin(theta_cube) ** 2

    return (thickness / (2 * wl_axis_cube)) * (term_1 + term_2)
