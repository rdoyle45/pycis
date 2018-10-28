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



class SpectraCalib(object):
    """ Creates a 'spectral raw_data cube' for input into 'SynthImage' class. 

    """

    def __init__(self, line_name, I0, vi, Ti, instrument, b_field=0, theta=0, LUT=False):
        """    Three modes of operation:

               'UNIFORM': When all inputs are given as single floats, uniform input is modelled across camera. Saves BIGLY on
               computation time for SynthImage since only a single instance of Lineshape need be evaluated. Intended for testing.

               'LUT': When all inputs are given as full-sized arrays with simplified spatial profiles. If there exists a small number
               of unique Lineshapes that characterises the input, only these Lineshapes are evaluated and a look-up table is created
               for the image. Saves a lot of time in generation of synthetic image, intended for quickly testing demodulation
               routines.

               'PROFILE': Full-sized arrays specifying complicated spatial profiles. Super slow. Intended for
               generating realistic synthetic images

               Inputs:
               line_name: string name of the line in question eg. 'CIII'
               I0: Line intensity (integral over intensity spectrum I(nu)) [ photons / pixel area / camera integration step ]
               vi: line-of-sight ion velocity [m / s]  <--- May change this input to vi
               Ti: ion temperature [eV]
               B: Magnetic field strength [T]
               theta: angle between B-field and line-of-sight [rad]
               LUT: Set to true to attempt to simplify and speed up the analysis_tools. Defaults to False.
           """


        self.line_name = line_name
        self.instrument = instrument
        dim = self.instrument.camera.sensor_dim

        if isinstance(I0, (float, int)) and isinstance(vi, (float, int)) and isinstance(Ti, (float, int)) and isinstance(theta, (float, int)):
            # -- UNIFORM MODE -- #
            # mode allows simplified, uniform spectra to be stored as single instance of Lineshape class:

            self.mode = 'uniform'
            self.spec_cube = Lineshape(self.line_name, I0, vi, Ti, b_field, theta)
            self.m_i = self.spec_cube.m_i

        else:
            # Allow for uniform inputs in other modes:

            if isinstance(I0, (float, int)):
                I0 = np.ones(dim) * I0
            if isinstance(vi, (float, int)):
                vi = np.ones(dim) * vi
            if isinstance(Ti, (float, int)):
                Ti = np.ones(dim) * Ti
            if isinstance(b_field, (float, int)):
                b_field = np.ones(dim) * b_field
            if isinstance(theta, (float, int)):
                theta = np.ones(dim) * theta

            # synth spectra lut:


            LUT_process = 'unattempted'
            if LUT:
                # -- LUT MODE -- #
                # Attempts to calculate how many unique parameter combinations exist across (x, y) space, ie how many different lineshapes
                # need be produced to generate synthetic image. If this is more than 20, it gives up:

                img_LUT_flat = np.ones(dim, dtype=np.int).flatten() * -1  # an array of ID's characterising the lineshape at each pixel
                unique_lineshapes_found = 0
                LUT_process = 'attempting'
                I0_flat = I0.flatten()
                vi_flat = vi.flatten()
                Ti_flat = Ti.flatten()
                b_field_flat = b_field.flatten()
                theta_flat = theta.flatten()

                I0_LUT = np.array([])
                vi_LUT = np.array([])
                Ti_LUT = np.array([])
                B_LUT = np.array([])
                theta_LUT = np.array([])

                while np.any(img_LUT_flat == -1):
                    # this method becomes slower than pixel-by-pixel analysis_tools for a high number of unique lineshapes
                    # stop trying to find all unique lineshapes if some threshhold is reached:
                    if unique_lineshapes_found == 5000:
                        LUT_process = 'failed'
                        break

                    # locate first index in flattened array with value -1:
                    pix_index_flat = next((idx, val) for idx, val in enumerate(img_LUT_flat) if val == -1)[0]
                    # pix_index = np.unravel_index(pix_index_flat, [y_pix, x_pix])
                    img_LUT_flat[np.where((I0_flat == I0_flat[pix_index_flat]) & (vi_flat == vi_flat[pix_index_flat]) & (
                        Ti_flat == Ti_flat[pix_index_flat]) & (b_field_flat == b_field_flat[pix_index_flat]) & (
                                              theta_flat == theta_flat[pix_index_flat]))] = unique_lineshapes_found

                    I0_LUT = np.append(I0_LUT, I0_flat[pix_index_flat])
                    vi_LUT = np.append(vi_LUT, vi_flat[pix_index_flat])
                    Ti_LUT = np.append(Ti_LUT, Ti_flat[pix_index_flat])
                    B_LUT = np.append(B_LUT, b_field_flat[pix_index_flat])
                    theta_LUT = np.append(theta_LUT, theta_flat[pix_index_flat])

                    unique_lineshapes_found += 1
                    print(unique_lineshapes_found)

                if LUT_process != 'failed':
                    LUT_process = 'successful'
                    self.mode = 'LUT'
                    self.unique_lineshape_no = len(set(img_LUT_flat))  # number of 'unique' lineshapes to be generated
                    self.img_LUT = img_LUT_flat.reshape(dim)

                    # Now define spectral cube with reference to small look-up table of lineshapes
                    self.spec_cube = []
                    for idx_LUT in range(0, self.unique_lineshape_no):
                        self.spec_cube.append(
                            Lineshape(name=line_name, I0=I0_LUT[idx_LUT], vi=vi_LUT[idx_LUT], Ti=Ti_LUT[idx_LUT],
                                      b_field=B_LUT[idx_LUT],
                                      theta=theta_LUT[idx_LUT]))
                    self.m_i = self.spec_cube[0].m_i

            print(LUT_process)
            y = 0
            # -- PROFILE MODE -- #
            if LUT_process == 'unattempted' or LUT_process == 'failed':
                self.mode = 'profile'
                # nested list structure represents pixel array, with Lineshape object generated and saved at each pixel:
                self.spec_cube = []
                for x in range(0, dim[1]):
                    self.spec_cube.append([])
                    for y in range(0, dim[0]):
                        self.spec_cube[x].append(
                            Lineshape(name=line_name, I0=I0[y, x], vi=vi[y, x], Ti=Ti[y, x], b_field=b_field[y, x], theta=theta[y, x]))
                self.m_i = self.spec_cube[0][0].m_i

            self.mode = 'profile'
            # Allow for uniform inputs in other modes:
            dim = self.instrument.camera.sensor_dim

        if isinstance(I0, (float, int)):
            I0 = np.ones(dim) * I0
        if isinstance(vi, (float, int)):
            vi = np.ones(dim) * vi
        if isinstance(Ti, (float, int)):
            Ti = np.ones(dim) * Ti
        if isinstance(b_field, (float, int)):
            b_field = np.ones(dim) * b_field
        if isinstance(theta, (float, int)):
            theta = np.ones(dim) * theta

        # self.spec_cube = []
        # for x in range(0, dim[1]):
        #     print(x)
        #     self.spec_cube.append([])
        #     for y in range(0, dim[0]):
        #         self.spec_cube[x].append(Lineshape(name=line_name, I0=I0[y, x], vi=vi[y, x], Ti=Ti[y, x], B=b_field[y, x], theta=theta[y, x]))
        # self.m_i = self.spec_cube[0][0].m_i

        self.I0 = I0
        self.vi = vi
        self.Ti = Ti
        self.b_field = b_field
        self.theta = theta

    def save(self, savepath, savename):
        pickle.dump(self, open(savepath + savename + '.p', 'wb'))
        return



