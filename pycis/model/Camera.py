import numpy as np
import pickle
import pycis
import os.path
import multiprocessing as mp
import random
import matplotlib.pyplot as plt


class Camera(object):
    """ Container for information on the camera sensor. """

    def __init__(self, name, bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise):
        """
        :param name: 
        :param bit_depth: 
        :param sensor_dim: 
        :param pix_size: pixel dimension [m].
        :param qe: Quantum efficiency or sensor.
        :param epercount: Conversion gain of sensor.
        :param cam_noise: Camera noise standard deviation [electrons].
        """

        self.name = name
        self.pix_size = pix_size
        self.sensor_dim = sensor_dim
        self.qe = qe
        self.epercount = epercount
        self.cam_noise = cam_noise
        self.bit_depth = bit_depth

    def capture(self, photon_fluence, clean=False, display=False):
        """ model sensor signal given photon fluence (photons / pixel / camera timestep).
        
        :param photon_fluence: numpy array of camera  
        :param clean: 
        :return: camera_signal
        """

        np.random.seed()

        # account for quantum efficiency
        electron_fluence = photon_fluence * self.qe

        # add shot noise
        shot_noise = np.random.poisson(electron_fluence) - electron_fluence
        electron_fluence += shot_noise

        if not clean:
            # add camera noise
            if self.cam_noise != 0:
                electron_fluence += np.random.normal(0, self.cam_noise, self.sensor_dim)

        # apply gain
        signal = electron_fluence / self.epercount

        # digitise at bitrate of sensor
        signal = np.digitize(signal, np.arange(0, 2 ** self.bit_depth))

        if display:
            plt.figure()
            plt.imshow(signal, 'gray')
            plt.colorbar()
            plt.show()

        return signal

    def capture_stack(self, photon_fluence, num_stack, display=False):
        """ Quickly capture of a stack of image frames, returning the total signal. """

        # older implementation loops over Camera.capture() method and is far, far slower:
        # pool = mp.Pool(processes=2)
        # total_signal = sum(pool.map(self.capture, [photon_fluence] * num_stack))

        stacked_photon_fluence = num_stack * photon_fluence

        electron_fluence = photon_fluence * self.qe
        stacked_electron_fluence = stacked_photon_fluence * self.qe

        electron_noise_std = np.sqrt(electron_fluence + self.cam_noise ** 2)
        stacked_electron_noise_std = np.sqrt(num_stack * (electron_noise_std))

        stacked_electron_fluence += np.random.normal(0, stacked_electron_noise_std, self.sensor_dim)

        # apply gain
        # signal = electron_fluence / self.epercount
        stacked_signal = stacked_electron_fluence / self.epercount

        # digitise at bitrate of sensor
        # signal = np.digitize(signal, np.arange(2 ** self.bit_depth))
        stacked_signal = np.digitize(stacked_signal, np.arange(num_stack * 2 ** self.bit_depth))

        if display:
            plt.figure()
            plt.imshow(stacked_signal, 'gray')
            plt.colorbar()
            plt.show()

        return stacked_signal

    def save(self):
        pickle.dump(self, open(os.path.join(pycis.paths.camera_path, self.name + '.p'), 'wb'))
        return


if __name__ == '__main__':

    # name = 'fibre'  # a single pixel 'fibre' view for quick phase comparisons
    # bit_depth = 12
    # sensor_dim = [1, 1]
    # pix_size = 20e-6
    # QE = 0.45
    # epercount = 23.3
    # cam_noise = 37

    # name = 'test_cam'  # a 50 x 50 pixel, noiseless test camera for speedy synthetic image generation
    # bit_depth = 12
    # sensor_dim = [50, 50]
    # pix_size = 20e-6
    # QE = 0.45
    # epercount = 23.3
    # cam_noise = 0

    # name = 'photron_SA4_clean'
    # bit_depth = 12
    # sensor_dim = [1024, 1024]
    # pix_size = 20e-6
    # QE = 0.45
    # epercount = 23.3
    # cam_noise = 0

    name = 'photron_SA4'
    bit_depth = 12
    sensor_dim = [1024, 1024]
    pix_size = 20e-6
    qe = 0.45
    epercount = 11.6
    cam_noise = 41.2

    # name = 'pco.edge 5.5'
    # bit_depth = 16
    # sensor_dim = [2560, 2160]
    # pix_size = 6.5e-6
    # QE = 0.35
    # epercount = 0.46  # [e / count]
    # cam_noise = 3 * 2.5 / epercount  # [e]

    # name = 'pco.edge 5.5 CLEAN'
    # bit_depth = 16
    # sensor_dim = [2560, 2160]
    # pix_size = 6.5e-6
    # QE = 0.35
    # epercount = 0.46  # [e / count]
    # cam_noise = 0  # [e]

    camera = pycis.model.Camera(name, bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise)
    camera.save()

