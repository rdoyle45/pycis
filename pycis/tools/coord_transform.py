# CIS_modelling
# coord_transform

# jsallcock
# created: 16/02/2017

import matplotlib as mpl
import numpy as np
import scipy as sp
import scipy.io as sio
import os
import glob

from matplotlib import pyplot as plt
from matplotlib import image as mpimg


def cart2pol(y, x):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return y, x

if __name__ == '__main__':
    pass