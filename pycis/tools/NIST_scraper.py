# CIS_modelling
# NIST_lines

# Calls A. Tronchin's NISTASD routine to scrape atomic spectra synth_data from the lines database.

# Inputs:
#    spec: string, the name of the element
#    savepath: filepath for the pickled output
#    lowwl: [nm] wavelength lower bound
#    uppwl: [nm] wavelength upper bound

# About:
# MUST BE RUN IN PYTHON 2.7.*!!!!!
# Defaults to deuterium, complete scpectrum.

# To do:
# Convert NISTASD to python3?


# jsallcock
# created: 27/01/2017

import matplotlib as mpl
import numpy as np
import scipy as sp
import scipy.io as sio
import os
import glob
import NISTASD
import pickle

import pycis

from matplotlib import pyplot as plt
from matplotlib import image as mpimg


def NIST_scraper(spec='D', savename='D_all', lowwl=1, uppwl=1000, savepath=pycis.paths.lines_path):
    # convert wl bounds [nm] --> [A]
    lowwl *= 1e1
    uppwl *= 1e1
    aspec = NISTASD.NISTASD(spec=spec, plot=False, lowwl=lowwl, uppwl=uppwl)
    lines = aspec.lines[:]

    # Save the list into a pickle file.
    pickle.dump(lines, open(savepath + savename + "_lines.p", "wb"))
    return


if __name__ == '__main__':
    NIST_scraper(spec='Cd', savename='Cd', lowwl=467, uppwl=468)