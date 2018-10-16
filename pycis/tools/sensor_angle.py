# CIS
# sensor_angle

# jsallcock
# created: 20/06/2017

import matplotlib as mpl
import numpy as np
import scipy as sp
import scipy.io as sio
import os
import glob

from matplotlib import pyplot as plt
from matplotlib import image as mpimg


def sensor_angle(sensor_dim, pix_dim, f3):
    # handy calculations for pixel array geometry:
    # Position of detector centre relative to (0,0) corner [m]:
    centre = [pix_dim * sensor_dim[0] / 2, pix_dim * sensor_dim[1] / 2]
    y_pos = np.arange(0, sensor_dim[0])
    y_pos = (y_pos - 0.5) * pix_dim - centre[0]  # [m]
    x_pos = np.arange(0, sensor_dim[1])
    x_pos = (x_pos - 0.5) * pix_dim - centre[1]  # [m]

    # x and y position arrays across detector:
    ypos_array = np.tile(y_pos, [sensor_dim[1], 1])
    y_pos_array = np.transpose(ypos_array)
    x_pos_array = np.tile(x_pos, [sensor_dim[0], 1])

    theta = np.arctan(np.sqrt((x_pos_array) ** 2 + (y_pos_array) ** 2) / (f3))
    theta *= 360 / (2 * np.pi)  # [rad] --> [deg]

    return theta


if __name__ == '__main__':
    sensor_angle()