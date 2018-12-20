import numpy as np
import pickle
import pycis
import os.path
import csv
import matplotlib.pyplot as plt


class Crystal(object):
    """ Class containing information on a birefringent crystal. currently, two 'types' are accepted as string arguments:
     'waveplate' and 'Savart'. """

    def __init__(self, name, thickness):
        #
        # if type != 'waveplate' or 'Savart':
        #     raise Exception('# ERROR #   Accepted inputs for "type" are "waveplate" or "Savart"')

        self.name = name
        self.thickness = thickness

    def save(self):
        pickle.dump(self, open(os.path.join(pycis.paths.crystal_path, self.name + '.p'), 'wb'))
        return


class Uniaxial(Crystal):

    def __init__(self, name, thickness, cut_angle):
        super().__init__(name, thickness)
        self.cut_angle = cut_angle

    def optical_path_difference(self):
        pass

class Waveplate(Uniaxial):

    def __init__(self, name, thickness, cut_angle=0):
        super().__init__(name, thickness, cut_angle)


    def optical_path_difference(self):
        pass



class Savartplate(Crystal):

    def __init__(self, name, thickness):
        super().__init__(name, thickness)


    def optical_path_difference(self):
        pass


if __name__ == '__main__':

    # create 'ccfe_4.6mm_waveplate'  crystal:

    crystal = pycis.model.Waveplate('ccfe_4.6mm_waveplate', 4.48e-3)
    crystal.save()

    # create 'ccfe_6.2mm_savartplate'  crystal:

    crystal = pycis.model.Savartplate('ccfe_6.2mm_savartplate', 6.2e-3)
    crystal.save()

    # create 'ccfe_6.2mm_savartplate'  crystal:

    crystal = pycis.model.Savartplate('10mm_savartplate', 10.0e-3)
    crystal.save()






