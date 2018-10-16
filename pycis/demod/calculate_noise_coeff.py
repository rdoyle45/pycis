import numpy as np


def calculate_noise_coeff(window_2d):

    imdim_x, imdim_y = np.shape(window_2d)

    x_arr = np.arange(0, imdim_x / 2 + 1)
    y_arr = np.arange(0, imdim_y / 2 + 1)

    x_mesh, y_mesh = np.meshgrid([x_arr, y_arr])

    # currently assums that the fringes are horizontal
    noise_coeff = np.sqrt(np.trapz(abs(window_2d) ** 2, x_arr, axis=0))



    return



if __name__ == '__main__':
    calculate_noise_coeff()