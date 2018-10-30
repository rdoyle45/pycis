import numpy as np
import matplotlib.pyplot as plt

def photron_sa4_noise_lookup(mean_signal_level):
    """

    :param mean_signal_level: in counts
    :return:
    """

    k = 0.086  # counts / e-
    sigma_d = np.sqrt(1700)  # e-

    noise_sigma = np.sqrt(k ** 2 * sigma_d ** 2 + k * mean_signal_level)

    return noise_sigma