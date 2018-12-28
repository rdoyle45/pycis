import numpy as np
import pickle
import pycis
import os.path
from pycis.model.Lineshape import Lineshape


class SpectraCherab(object):
    """ Creates a 'spectral raw_data cube' for input into 'SynthImage' class.

     Since the class takes as its input the raw cherab output, some raw_data manipulation is required. """

    def __init__(self, wavelength_axis, spectra, instrument, name):

        # take absolute intensity (need to ask Matt Carr about negative intensities!)
        spectra = abs(spectra)
        spectra = np.rot90(spectra)

        self.wavelength_axis = wavelength_axis
        self.spectra = spectra
        self.instrument = instrument
        self.name = name

    def save(self):
        pickle.dump(self, open(os.path.join(pycis.paths.spectra_path, self.name + '.p'), 'wb'))
        return




