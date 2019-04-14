import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import pycis


class Camera(object):
    """
    Camera base class

    """

    def __init__(self, bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise):
        """

        :param bit_depth: 
        :param sensor_dim: (y, x)
        :param pix_size: pixel dimension [ m ].
        :param qe: Quantum efficiency or sensor.
        :param epercount: Conversion gain of sensor.
        :param cam_noise: Camera noise standard deviation [ e- ].

        """

        self.pix_size = pix_size
        self.sensor_dim = sensor_dim
        self.qe = qe
        self.epercount = epercount
        self.cam_noise = cam_noise
        self.bit_depth = bit_depth

        self.x, self.y = self._get_sensor_coords()
        # TODO downsampling and cropping?

    def _get_sensor_coords(self):
        """
        get sensor pixel coordinates in xy plane in [ m ]

        (x = 0, y = 0) is the optic axis. By default this is aligned with the centrepoint of the sensor.

        :return:
        """
        centre_pos = self.pix_size * np.array(self.sensor_dim) / 2  # [ m ]

        x = (np.arange(self.sensor_dim[0]) + 0.5) * self.pix_size - centre_pos[0]  # [ m ]
        y = (np.arange(self.sensor_dim[1]) + 0.5) * self.pix_size - centre_pos[1]  # [ m ]

        x = xr.DataArray(x, dims=('x',), coords=(x,), name='x')
        y = xr.DataArray(y, dims=('y',), coords=(y,), name='y')

        return x, y

    def capture(self, spec, clean=False, display=False):
        """ model sensor signal given photon fluence (photons / pixel / camera timestep).
        
        :param intensity: stokes vector
        :param clean: 
        :return: camera_signal
        """

        # np.random.seed()
        np.random.seed()

        # spectral response of camera assumed uniform
        spec = spec.sum('wavelength')

        # no polarisation sensitivity
        if 'stokes' in spec.dims:
            spec = spec.isel(stokes=0).squeeze('stokes')

        # account for quantum efficiency
        electron_fluence = spec * self.qe

        if not clean:
            # add shot noise
            shot_noise = np.random.poisson(electron_fluence) - electron_fluence
            electron_fluence += shot_noise

            # add camera noise
            electron_fluence += np.random.normal(0, self.cam_noise, self.sensor_dim)

        # apply gain
        signal = electron_fluence / self.epercount

        # digitise at bitrate of sensor
        signal.values = np.digitize(signal.values, np.arange(0, 2 ** self.bit_depth))

        if display:
            fig, ax = plt.subplots()
            im = ax.imshow(signal, 'gray')
            cbar = fig.colorbar(im, ax=ax)
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


class PolCamera(Camera):
    """ camera with micro-polariser array"""

    def __init__(self, bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise, pol_angles=np.array([[0, 45], [135, 90]])):
        """

        :param pol_angles: [ deg ]
        :type pol_angles: np.ndarray

        """

        assert sensor_dim[0] % 2 == 0
        assert sensor_dim[1] % 2 == 0

        super().__init__(bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise)
        self.pol_angles = pol_angles

        # define Mueller matrices for the 4 polariser orientations

        self.mm_0deg = 0.5 * np.array([[1, -1, 0, 0],
                                      [-1, 1, 0, 0],
                                      [0, 0, 0, 0],
                                      [0, 0, 0, 0]])

        self.mm_45deg = 0.5 * np.array([[1, 0, 1, 0],
                                       [0, 0, 0, 0],
                                       [1, 0, 1, 0],
                                       [0, 0, 0, 0]])

        self.mm_90deg = 0.5 * np.array([[1, 1, 0, 0],
                                       [1, 1, 0, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0]])

        self.mm_m45deg = 0.5 * np.array([[1, 0, -1, 0],
                                        [0, 0, 0, 0],
                                        [-1, 0, 1, 0],
                                        [0, 0, 0, 0]])

        # pad to generate pixel array of Mueller matrices

        # there is probably a better way of doing this...

        pix_idxs_y = np.arange(0, sensor_dim[0], 2)
        pix_idxs_x = np.arange(0, sensor_dim[1], 2)
        pix_idxs_y, pix_idxs_x = np.meshgrid(pix_idxs_y, pix_idxs_x)

        mueller_matrix = np.zeros([4, 4, self.sensor_dim[0], self.sensor_dim[1]])

        mueller_matrix[:, :, pix_idxs_y, pix_idxs_x] = self.mm_0deg[:, :, np.newaxis, np.newaxis]
        mueller_matrix[:, :, pix_idxs_y, pix_idxs_x + 1] = self.mm_45deg[:, :, np.newaxis, np.newaxis]
        mueller_matrix[:, :, pix_idxs_y + 1, pix_idxs_x] = self.mm_90deg[:, :, np.newaxis, np.newaxis]
        mueller_matrix[:, :, pix_idxs_y + 1, pix_idxs_x + 1] = self.mm_m45deg[:, :, np.newaxis, np.newaxis]

        self.mueller_matrix = mueller_matrix

    def _get_micropolariser_mueller_mat(self, pol_angle):
        lp = pycis.model.LinearPolariser()


    def capture(self, intensity, clean=False, display=False):
        """

        :param intensity:
        :param clean:
        :param display:
        :return:
        """

        assert isinstance(intensity, np.ndarray)
        assert intensity.shape[0] == 4
        assert intensity.shape[1] == self.sensor_dim[0]
        assert intensity.shape[2] == self.sensor_dim[1]

        # matrix multiplication (mueller matrix axes are the first two axes)
        subscripts = 'ij...,j...->i...'
        stokes_vector_out = np.einsum(subscripts, self.mueller_matrix, intensity)

        intensity = stokes_vector_out[0]

        np.random.seed()

        # account for quantum efficiency
        electron_fluence = intensity * self.qe

        if not clean:
            # add shot noise
            shot_noise = np.random.poisson(electron_fluence, size=self.sensor_dim) - electron_fluence
            electron_fluence += shot_noise

            # add camera noise
            electron_fluence += np.random.normal(0, self.cam_noise, size=self.sensor_dim)

        # apply gain
        signal = electron_fluence / self.epercount

        # digitise at bitrate of sensor
        signal = np.digitize(signal, np.arange(0, 2 ** self.bit_depth))

        if display:
            fig, ax = plt.subplots()
            im = ax.imshow(signal, 'gray')
            cbar = fig.colorbar(im, ax=ax)
            plt.show()

        return signal

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
    sensor_dim = [256, 216]
    pix_size = 20e-6
    qe = 0.3
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

    camera = PolCamera(bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise)
