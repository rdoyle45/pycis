import numpy as np
import pickle
import os.path
import pycis.paths
import matplotlib.pyplot as plt
import copy

import scipy.interpolate


class Instrument(object):
    """ 
    CIS instrument class. Facilitates synthetic image generation.
    """

    def __init__(self, name, camera, back_lens, crystals, bandpass_filter=None,
                 instrument_contrast=0.5):
        """
        
        :param name: 
        :param camera: pycis.model.Camera or a string-type camera name
        :param back_lens: pycis.model.Lens or a string-type lens name
        :param crystals: list of pycis.model.Crystal
        :param crystal_orientations: corresponding list of orientations in radians
        :param bandpass_filter: pycis.model.BandpassFilter or string-type filter name
        :param instrument_contrast: 
        """

        self.name = name
        self.instrument_contrast = instrument_contrast

        # camera
        if isinstance(camera, pycis.model.Camera):
            self.camera = camera
        elif isinstance(camera, str):
            self.camera = pycis.model.load_component(camera, type='camera')
        else:
            raise Exception('argument: camera must be of type pycis.model.Camera or string')

        # back_lens
        if isinstance(back_lens, pycis.model.Lens):
            self.back_lens = back_lens

        elif isinstance(back_lens, str):
            self.back_lens = pycis.model.load_component(back_lens, type='lens')
        else:
            raise Exception('argument: lens must be of type pycis.model.Camera or string')

        # crystals
        assert all(isinstance(c, pycis.model.BirefringentComponent) for c in crystals)

        self.crystals = crystals

        # bandpass_filter
        if bandpass_filter is None:
            pass

        elif isinstance(bandpass_filter, pycis.model.BandpassFilter):
            self.bandpass_filter = bandpass_filter

        elif isinstance(bandpass_filter, str):
            self.bandpass_filter = pycis.model.FilterFromName(bandpass_filter)

        else:
            raise Exception('argument: bandpass_filter must be of type pycis.model.Camera or string')

        # TODO include crystal misalignment
        self.chi = [0, 0]  # placeholder, this will be gotten rid of in time

    def calculate_ray_angles(self, downsample=None, letterbox=None, display=False):
        """
        Calculate incidence and azimuthal angles for each pixel's 'sightline' through each crystal
        
        :param downsample: defaults to None, downsample the data for fitting
        :param letterbox:
        :param display: 
        :return: inc_angles, azim_angles [ rad ]
        """

        cam = self.camera
        f_3 = self.back_lens.focal_length
        sensor_dim = list(copy.copy(cam.sensor_dim))

        if letterbox is not None:
            sensor_dim[0] -= 2 * letterbox

        # Define x,y detector coordinates:
        #
        # tilt_offset_y = f_3 * np.tan(self.chi[0])  # vertical tilt
        # tilt_offset_x = f_3 * np.tan(self.chi[1])  # horizontal tilt

        centre = [(cam.pix_size * (sensor_dim[0] - 2) / 2),  # + tilt_offset_y,
                  (cam.pix_size * (sensor_dim[1] - 2) / 2)]  # + tilt_offset_x]

        if downsample is None:
            y_pos = np.arange(0, sensor_dim[0])
            x_pos = np.arange(0, sensor_dim[1])
        else:
            y_pos = np.arange(0, sensor_dim[0])[::downsample]
            x_pos = np.arange(0, sensor_dim[1])[::downsample]

        y_pos = (y_pos - 0.5) * cam.pix_size - centre[0]  # [m]
        x_pos = (x_pos - 0.5) * cam.pix_size - centre[1]  # [m]

        x, y = np.meshgrid(x_pos, y_pos)

        # assuming for now that waveplate and Savart plate are parallel and perfectly alligned, their incidence angle
        # projections are now the same.

        inc_angles = np.arctan2(np.sqrt(x ** 2 + y ** 2), f_3)

        # azimuthal angles vary with crystal so are calculated separately
        crystal_azim_angles = []

        for crystal in self.crystals:

            orientation = crystal.orientation

            # Rotate x, y coordinates by crystal orientation:
            x_rot = (np.cos(orientation) * x) + (np.sin(orientation) * y)
            y_rot = (- np.sin(orientation) * x) + (np.cos(orientation) * y)

            crystal_azim_angles.append(np.arctan2(x_rot, y_rot))

        if display:
            # TODO account for multiple crystals
            num_crystals = len(crystal_azim_angles)

            fig1 = plt.figure()
            ax1 = fig1.add_subplot(121)
            ax2 = fig1.add_subplot(122)

            im1 = ax1.imshow(inc_angles)
            cbar = plt.colorbar(im1, ax=ax1)

            im2 = ax2.imshow(crystal_azim_angles)
            cbar = plt.colorbar(im2, ax=ax2)

            plt.tight_layout()
            plt.show()

        return inc_angles, crystal_azim_angles

    def calculate_mueller_matrix(self, wl):
        """ calculate the total Mueller matrix for the interferometer at specified wavelength / s """

        # calculate the angles of each pixel's line of sight through the interferometer
        inc_angles, crystal_azim_angles = self.calculate_ray_angles()

        polariser_1 = pycis.LinearPolariser(0)
        polariser_2 = pycis.LinearPolariser(0)

        mueller_mat = polariser_1.calculate_mueller_mat()
        fmt = 'ij...,jl...->il...'

        for crystal, azim_angles in zip(self.crystals, crystal_azim_angles):
            # matrix multiplication
            mueller_mat = np.einsum(fmt, crystal.calculate_mueller_mat(wl, inc_angles, azim_angles), mueller_mat)

        mueller_mat = np.einsum(fmt, polariser_2.calculate_mueller_mat(), mueller_mat)

        return mueller_mat


    def calculate_phase_delay(self, wl, n_e=None, n_o=None, downsample=None, letterbox=None, output_components=False):
        """
        accounting for crystal orientation + alignment
        
        :param wl: 
        :param n_e: 
        :param n_o: 
        :param downsample: 
        :param letterbox: 
        :param output_components: 
        :return: 
        """

        # calculate the angles of each pixel's line of sight through the interferometer
        inc_angles, crystal_azim_angles = self.calculate_ray_angles(downsample=downsample, letterbox=letterbox)

        # calculate phase delay contribution due to each crystal
        phase = 0
        for crystal, azim_angles in zip(self.crystals, crystal_azim_angles):
            phase += crystal.calculate_phase_delay(wl, inc_angles, azim_angles, n_e=n_e, n_o=n_o)

        if not output_components:
            return phase

        else:
            # calculate the phase_offset and phase_shape
            phase_offset = self.calculate_phase_offset(wl, n_e=n_e, n_o=n_o)
            phase_shape = phase - phase_offset

            return phase_offset, phase_shape

    def calculate_phase_offset(self, wl, n_e=None, n_o=None):
        """
        
        :param wl: 
        :param n_e: 
        :param n_o: 
        :return: phase_offset [ rad ]
        """

        phase_offset = 0
        for crystal in self.crystals:
            phase_offset += crystal.calculate_phase_delay(wl, 0., 0., n_e=n_e, n_o=n_o)

        return phase_offset












    def get_snr_intensity(self, line_name, snr):
        """ Given spectral line and desired approximate image snr (central ROI), return the necessary input intensity I0 in units 
        of [photons/pixel/timestep]. """

        # load line
        line = pycis.model.Lineshape(line_name, 1, 0, 1)

        _, _, _, wavelength_com = line.make(1000)

        if self.bandpass_filter is not None:
            # estimate filters transmission at line wavelength
            t_filter = self.bandpass_filter.interp_tx(wavelength_com)
        else:
            t_filter = 1

        qe = self.camera.qe
        sigma_cam_e = self.camera.cam_noise

        a = (t_filter * qe) ** 2
        b = - snr ** 2 * t_filter * qe
        c = - (snr * sigma_cam_e) ** 2

        i0_1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        i0_2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

        return i0_1

    def get_contrast_ti(self, line_name, contrast):
        """ Given spectral line and desired approximate image contrast, return necessary input ion temp in [eV]."""

        C = pycis.tools.c
        k_B = pycis.tools.k_B
        E = pycis.tools.e

        # load line
        line = pycis.model.Lineshape(line_name, 1, 0, 1)

        _, _, _, wavelength_com = line.make(1000)

        # calculate characteristic temperature

        biref, n_e, n_o, kappa, dBdlambda, d2Bdlambda2, d3Bdlambda3 = pycis.model.bbo_slo(wavelength_com)

        phase_wp = uniaxial_crystal(wavelength_com, n_e, n_o, self.waveplate.thickness, 0, 0)

        phase_sp = savart_plate(wavelength_com, n_e, n_o, self.savartplate.thickness, 0, 0)

        phase = phase_wp + phase_sp

        group_delay  = kappa * phase

        t_characteristic = (line.m_i * C ** 2) / (2 * k_B * (np.pi * group_delay) ** 2)

        # calculate multiplet contrast

        coherence_m = 0  # multiplet complex coherence
        # loop over multiplet transitions:
        line_no = line.line_no
        lines_wl = np.zeros(line_no)
        lines_rel_int = np.zeros(line_no)
        for i in range(0, line_no):
            lines_wl[i] = line.lines['wave_obs'].iloc[i]
            lines_rel_int[i] = line.lines['rel_int'].iloc[i]

        nu_0 = C / wavelength_com

        for m in range(0, line_no):
            nu_m = C / (lines_wl[m])  # [Hz]
            xi_m = (nu_0 - nu_m) / nu_0
            coherence_m += np.exp(2 * np.pi * 1j * group_delay * xi_m) * lines_rel_int[m]
        contrast_m = abs(coherence_m)
        phase_m = np.angle(coherence_m) / (2 * np.pi)  # [waves]

        t_ion = - t_characteristic * np.log(contrast / contrast_m)  # [K]

        t_ion *= (k_B / E)  # [eV]

        return t_ion

    def apply_vignetting(self, photon_fluence):
        """account for instrument etendue to first order based on Scott's predictive matlab code. """


        etendue = np.load(os.path.join(pycis.paths.instrument_path, 'etendue.npy'))
        sensor_distance = np.load(os.path.join(pycis.paths.instrument_path, 'sensor_distance.npy'))

        # convert sensor distance to (m)
        sensor_distance *= 1e-3

        sensor_distance_sym = np.concatenate([-sensor_distance[1:][::-1], sensor_distance])

        # normalise etendue to value at centre
        norm_etendue = etendue / etendue[0]

        norm_etendue_sym = np.concatenate([norm_etendue[1:][::-1], norm_etendue])

        # calculate how far sensor extends relative to sensor_distance
        # assume a square sensor for now!

        sensor_width = self.camera.pix_size * self.camera.sensor_dim[1]
        sensor_half_width = sensor_width / 2
        sensor_diagonal = np.sqrt(2) * sensor_width

        sensor_half_diagonal = sensor_diagonal / 2

        # linear splines for interpolation / extrapolation
        f = scipy.interpolate.InterpolatedUnivariateSpline(sensor_distance_sym, norm_etendue_sym, k=1)

        sensor_diagonal_axis = np.linspace(-sensor_half_diagonal, sensor_half_diagonal, self.camera.sensor_dim[1])

        w1d = f(sensor_diagonal_axis)

        #w1d = np.interp(sensor_diagonal_axis, sensor_distance_sym, norm_etendue_sym)

        l = self.camera.sensor_dim[1]
        m = sensor_half_width
        xx = np.linspace(-m, m, l)

        [x, y] = np.meshgrid(xx, xx)
        r = np.sqrt(x ** 2 + y ** 2)
        w2d_2 = np.zeros([l, l])
        w2d_2[r <= sensor_half_diagonal] = np.interp(r[r<=sensor_half_diagonal], sensor_diagonal_axis, w1d)

        w2d_2[w2d_2 < 0] = 0

        # w2d_2 = pycis.tools.gauss2d(self.camera.sensor_dim[1], 1200)

        vignetted_photon_fluence = photon_fluence * w2d_2
        vignetted_photon_fluence[vignetted_photon_fluence < 0] = 0

        return vignetted_photon_fluence

    def save(self):
        pickle.dump(self, open(os.path.join(pycis.paths.instrument_path, self.name + '.p'), 'wb'))
        return

# if __name__ == '__main__':
#     ...

