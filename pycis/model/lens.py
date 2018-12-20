import numpy as np
import pickle
import pycis
import os.path
import csv
import matplotlib.pyplot as plt

# Currently, model uses thin lens approximation.

class Lens(object):
    """  """

    def __init__(self, name, focal_length):
        #
        # if type != 'waveplate' or 'Savart':
        #     raise Exception('# ERROR #   Accepted inputs for "type" are "waveplate" or "Savart"')

        self.name = name
        self.focal_length = focal_length


    def save(self):
        pickle.dump(self, open(os.path.join(pycis.paths.lens_path, self.name + '.p'), 'wb'))
        return

if __name__ == '__main__':

    # # create 'sigma_150mm' lens:
    #
    # lens = pycis.model.Lens('sigma_150mm', focal_length=150e-3)
    # lens.save()

    # create 'GA_Nikon_85mm' lens:

    lens = pycis.model.Lens('GA_Nikon_85mm', focal_length=85e-3)
    lens.save()


