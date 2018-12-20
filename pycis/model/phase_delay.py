import numpy as np
import pycis


def uniaxial_crystal(wl, thickness, theta, omega, alpha=0., material='a-BBO'):
    """
    phase delay due to uniaxial crystal

    :param wl: wavelength [ m ]
    :param thickness: [ m ]
    :param theta: incidence angle [ rad ]
    :param omega: azimuthal angle [ rad ]
    :param alpha: angle between crystal optic axis and crystal front face [ rad ]
    :param material
    
    :return: phase delay [ fringes ]
    """

    biref, n_e, n_o = pycis.model.dispersion(wl, material)

    if not pycis.tools.is_scalar(wl) and not pycis.tools.is_scalar(theta) and not pycis.tools.is_scalar(omega):
        # wl, theta and omega are arrays

        img_dim = theta.shape
        assert img_dim == omega.shape

        wl_length = len(wl)

        reps_img = [wl_length, 1, 1]
        reps_axis = [img_dim[0], img_dim[1], 1]

        wl = np.tile(wl, reps_axis)
        n_e = np.tile(n_e, reps_axis)
        n_o = np.tile(n_o, reps_axis)

        theta = np.moveaxis(np.tile(theta, reps_img), 0, -1)
        omega = np.moveaxis(np.tile(omega, reps_img), 0, -1)

    term_1 = np.sqrt(n_o ** 2 - np.sin(theta) ** 2)

    term_2 = (n_o ** 2 - n_e ** 2) * \
             (np.sin(alpha) * np.cos(alpha) * np.cos(omega) * np.sin(theta)) / \
             (n_e ** 2 * np.sin(alpha) ** 2 + n_o ** 2 * np.cos(alpha) ** 2)

    term_3 = - n_o * np.sqrt(
        (n_e ** 2 * (n_e ** 2 * np.sin(alpha) ** 2 + n_o ** 2 * np.cos(alpha) ** 2)) -
        ((n_e ** 2 - (n_e ** 2 - n_o ** 2) * np.cos(alpha) ** 2 * np.sin(
            omega) ** 2) * np.sin(theta) ** 2)) / \
             (n_e ** 2 * np.sin(alpha) ** 2 + n_o ** 2 * np.cos(alpha) ** 2)

    return (thickness / wl) * (term_1 + term_2 + term_3)


def savart_plate(wl, thickness, theta, omega, material='a-BBO'):
    """
    calculate the phase delay due to a Savart plate
    
    :param wl: wavelength [ m ]
    :param thickness: [ m ]
    :param theta: incidence angle [ rad ]
    :param omega: azimuthal angle [ rad ]
    :param material: 
    
    :return: phase delay [ fringes ]
    """

    biref, n_e, n_o = pycis.model.dispersion(wl, material)

    a = 1 / n_e
    b = 1 / n_o

    if not pycis.tools.is_scalar(wl) and not pycis.tools.is_scalar(theta) and not pycis.tools.is_scalar(omega):
        # wl, theta and omega are arrays

        img_dim = theta.shape
        assert img_dim == omega.shape

        wl_length = len(wl)

        reps_img = [wl_length, 1, 1]
        reps_axis = [img_dim[0], img_dim[1], 1]

        wl = np.tile(wl, reps_axis)
        a = np.tile(a, reps_axis)
        b = np.tile(b, reps_axis)

        theta = np.moveaxis(np.tile(theta, reps_img), 0, -1)
        omega = np.moveaxis(np.tile(omega, reps_img), 0, -1)

    term_1 = ((a ** 2 - b ** 2) / (a ** 2 + b ** 2)) * (np.cos(omega) + np.sin(omega)) * np.sin(theta)

    term_2 = ((a ** 2 - b ** 2) / (a ** 2 + b ** 2) ** (3 / 2)) * ((a ** 2) / np.sqrt(2)) * \
             (np.cos(omega) ** 2 - np.sin(omega) ** 2) * np.sin(theta) ** 2

    return (thickness / (2 * wl)) * (term_1 + term_2)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    theta = np.linspace(0, 7, 1000) * np.pi / 180
    omega = np.linspace(0, 2 * np.pi, 1000)
    alpha = 0 * np.pi / 180
    wl = np.linspace(465e-9, 469e-9, 2)
    biref, n_e, n_o = pycis.model.dispersion(wl, material='a-BBO')

    theta_m, omega_m = np.meshgrid(theta, omega)

    l_wp = 4.48e-3
    l_sp = 6.2e-3

    s_new = time.time()
    phase_wp_new = uniaxial_crystal(wl, l_wp, theta_m, omega_m, alpha=alpha)
    phase_sp_new = savart_plate(wl, l_sp, theta_m, omega_m)
    e_new = time.time()
    time_new = e_new - s_new
    print(time_new)

    s_old = time.time()
    phase_wp_old = pycis.model.uniaxial_crystal_3d(wl, l_wp, theta_m, omega_m, alpha=alpha)
    phase_sp_old = pycis.model.savart_plate_3d(wl, l_sp, theta_m, omega_m)
    e_old = time.time()
    time_old = e_old - s_old

    print(time_old)

    # plt.figure()
    # plt.imshow(phase_wp_new[:, :, 0] - phase_wp_old[:, :, 0])
    # plt.colorbar()

    plt.figure()
    plt.imshow(phase_sp_new[:, :, 0])
    plt.colorbar()

    plt.figure()
    plt.imshow(phase_sp_old[:, :, 0])
    plt.colorbar()

    plt.figure()
    plt.imshow(phase_sp_new[:, :, 0] - phase_sp_old[:, :, 0])
    plt.colorbar()

    plt.show()

    a = 5

