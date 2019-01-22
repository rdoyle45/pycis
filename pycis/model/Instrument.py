import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy.interpolate

import pycis


class Instrument(object):
    """ 
    Coherence imaging spectroscopy instrument
    """

    def __init__(self, camera, back_lens, interferometer, bandpass_filter=None, interferometer_orientation=0):
        """
        :param camera: 
        :type camera pycis.model.Camera
        
        :param back_lens: 
        :type back_lens: pycis.model.Lens
        
        :param interferometer: a list of instances of pycis.model.InterferometerComponent
        :type interferometer: list
        
        :param bandpass_filter: pycis.model.BandpassFilter or string-type filter name
        """

        self.camera = camera
        self.back_lens = back_lens
        self.interferometer = interferometer

        self.crystals = self.get_crystals()
        self.polarisers = self.get_polarisers()

        self.interferometer_orientation = interferometer_orientation

        # TODO is bandpass_filter implemented correctly?
        self.bandpass_filter = bandpass_filter
        self.input_checks()

        self.x, self.y = self.calculate_sensor_coords()

        # assign instrument 'type' based on interferometer layout
        self.inst_type = self.check_instrument_type()
        print(self.inst_type)

        # TODO crystal misalignment
        self.chi = [0, 0]  # placeholder

    def input_checks(self):
        """
        check validity of __init__ arguments

        """

        assert isinstance(self.camera, pycis.model.Camera)
        assert isinstance(self.back_lens, pycis.model.Lens)
        assert all(isinstance(c, pycis.model.InterferometerComponent) for c in self.interferometer)

        if self.bandpass_filter is not None:
            assert isinstance(self.bandpass_filter, pycis.model.BandpassFilter)

    def get_crystals(self):
        """
        list the birefringent components in the interferometer, subset of self.interferometer list

        """

        return [c for c in self.interferometer if isinstance(c, pycis.BirefringentComponent)]

    def get_polarisers(self):
        """
        list the polarisers in the interferometer, subset of self.interferometer list

        """

        return [c for c in self.interferometer if isinstance(c, pycis.LinearPolariser)]

    def calculate_sensor_coords(self, x_coord=None, y_coord=None, crop=None, downsample=1, display=False):
        """
        Calculate x-y coordinates of the camera sensor for ray geometry calculations, in [ m ]

        Converts from pixel coordinates (origin top left) to spatial coordinates (origin centre). If x_coord and
        y_coord are not specified, entire sensor will be evaluated (with specified cropping and downsampling).

        Coordinates can be cropped and downsampled to facilitate direct comparison with cropped / downsampled
        experimental image, forgoing the need to generate a full-sensor synthetic image, as some of that output won't be
        used. can this be a pycis.Camera method? If both crop and downsample are specified, crop carried out first.

        :param x_coord: array of pixel coordinates (x)
        :param y_coord: array of pixel coordinates (y)
        :param crop: (y1, y2, x1, x2)
        :param downsample:
        :param display:

        :return: x_pos, y_pos [ m ]

        """

        # TODO clean up!

        sensor_dim = np.array(self.camera.sensor_dim)
        centre_pos = self.camera.pix_size * (sensor_dim - 2) / 2

        if x_coord is not None and y_coord is not None:
            # pixel coordinates on sensor have been manually specified

            assert np.shape(x_coord) == np.shape(y_coord)

            x_pos = (x_coord - 0.5) * self.camera.pix_size - centre_pos[1]  # [ m ]
            y_pos = (y_coord - 0.5) * self.camera.pix_size - centre_pos[0]  # [ m ]

        else:
            # entire sensor will be evaluated

            if crop is None:
                crop = (0, sensor_dim[0], 0, sensor_dim[1])

            y_coord = np.arange(crop[0], crop[1])[::downsample]
            x_coord = np.arange(crop[2], crop[3])[::downsample]

            y_pos = (y_coord - 0.5) * self.camera.pix_size - centre_pos[0]  # [ m ]
            x_pos = (x_coord - 0.5) * self.camera.pix_size - centre_pos[1]  # [ m ]

            x_pos, y_pos = np.meshgrid(x_pos, y_pos)

        if display:

            fig = plt.figure()
            axx = fig.add_subplot(121)
            axy = fig.add_subplot(122)

            imx = axx.imshow(x_coord, origin='lower')
            cbarx = fig.colorbar(imx, ax=axx)

            imy = axy.imshow(y_coord, origin='lower')
            cbary = fig.colorbar(imy, ax=axy)

            plt.tight_layout()
            plt.show()

        return x_pos, y_pos

    def calculate_ray_inc_angles(self, x_pos, y_pos):
        """
        incidence angles will be the same for all interferometer components (until arbitrary component misalignment 
        implemented)
        
        :param x_pos: x position (s), centred sensor coordinates [ m ]
        :param y_pos: y position (s), centred sensor coordinates [ m ]

        :return: incidence angles [ rad ]
        """

        assert np.shape(x_pos) == np.shape(y_pos)

        inc_angles = np.arctan2(np.sqrt(x_pos ** 2 + y_pos ** 2), self.back_lens.focal_length)

        return inc_angles

    def calculate_ray_azim_angles(self, x_pos, y_pos, crystal):
        """
        calculate azimuthal angles of rays through interferometer, given
        azimuthal angles vary with crystal orientation so are calculated separately

        :param x_pos: x position (s), centred sensor coordinates [ m ]
        :param y_pos: y position (s), centred sensor coordinates [ m ]
        :param crystal: instance of pycis.model.BirefringentComponent

        :return: azimuthal angles [ rad ]
        """

        assert np.shape(x_pos) == np.shape(y_pos)

        orientation = crystal.orientation + self.interferometer_orientation

        # Rotate x, y coordinates
        x_rot = (np.cos(orientation) * x_pos) + (np.sin(orientation) * y_pos)
        y_rot = (- np.sin(orientation) * x_pos) + (np.cos(orientation) * y_pos)

        return np.arctan2(x_rot, y_rot)

    def calculate_matrix(self, wl):
        """
        calculate the total Mueller matrix for the interferometer

        :param wl: [ m ]

        :return: instrument matrix
        """

        x, y = self.calculate_sensor_coords()
        inc_angle = self.calculate_ray_inc_angles(x, y)

        subscripts = 'ij...,jl...->il...'
        instrument_matrix = np.identity(4)

        for component in self.interferometer:

            azim_angle = self.calculate_ray_azim_angles(x, y, component)
            component_matrix = component.calculate_matrix(wl, inc_angle, azim_angle)

            # matrix multiplication
            instrument_matrix = np.einsum(subscripts, component_matrix, instrument_matrix)

        # account for orientation of the interferometer itself.
        rot_mat = pycis.model.calculate_rot_mat(self.interferometer_orientation)

        return np.einsum(subscripts, rot_mat, instrument_matrix)

    def calculate_ideal_phase_delay(self, wl, x=None, y=None, n_e=None, n_o=None, crop=None, downsample=1,
                                    output_components=False):
        """
        assumes all crystal's phase contributions combine constructively -- method used only when instrument.type =
        'two-beam'

        crop, downsample, x, y, n_e, n_o kwargs included for fitting purposes.

        :param wl:
        :param x:
        :param y:
        :param n_e:
        :param n_o:
        :param crop:
        :param downsample:
        :param output_components:
        :return:
        """

        # calculate the angles of each pixel's line of sight through the interferometer

        if x is None and y is None:
            x, y = self.calculate_sensor_coords(crop=crop, downsample=downsample)
        else:
            x, y = self.calculate_sensor_coords(x_coord=x, y_coord=y)

        inc_angles = self.calculate_ray_inc_angles(x, y)
        azim_angles = self.calculate_ray_azim_angles(x, y, self.crystals[0])

        # calculate phase delay contribution due to each crystal
        phase = 0
        for crystal in self.crystals:
            phase += crystal.calculate_phase_delay(wl, inc_angles, azim_angles, n_e=n_e, n_o=n_o)

        if not output_components:
            return phase

        else:
            # calculate the phase_offset and phase_shape
            phase_offset = self.calculate_ideal_phase_offset(wl, n_e=n_e, n_o=n_o)
            phase_shape = phase - phase_offset

            return phase_offset, phase_shape

    def calculate_ideal_contrast(self):

        contrast = 1
        for crystal in self.crystals:
            contrast *= crystal.contrast

        return contrast

    def calculate_ideal_transmission(self):
        """ transmission is decreased due to polarisers, calculate this factor analytically for the special case of 
        ideal interferometer"""

        pol_1, pol_2 = self.polarisers
        tx = (pol_1.tx_1 ** 2 + pol_1.tx_2 ** 2) * (pol_2.tx_1 ** 2 + pol_2.tx_2 ** 2)

        return tx

    def check_instrument_type(self):
        """
        instrument type determines best way to generate the observed interference pattern
        
        in the case of a perfectly aligned coherence imaging diagnostic in a simple 'two-beam' configuration, skip the 
        Mueller matrix calculation to the final result.
        
        :return: type (str)
        """

        orientations = []
        for crystal in self.crystals:
            orientations.append(crystal.orientation)

        # if all crystal orientations are the same and are at 45 degrees to the polarisers, perform a simplified 2-beam
        # interferometer calculation -- avoiding the full Mueller matrix treatment

        # are there two polarisers, at the front and back of the interferometer?
        if len(self.polarisers) == 2 and (isinstance(self.interferometer[0], pycis.LinearPolariser) and
                                              isinstance(self.interferometer[-1], pycis.LinearPolariser)):

            # are they alligned?
            pol_1_orientation = self.interferometer[0].orientation
            pol_2_orientation = self.interferometer[-1].orientation

            if pol_1_orientation == pol_2_orientation:

                # are all crystals alligned?
                crystal_1_orientation = self.crystals[0].orientation

                if all(c.orientation == crystal_1_orientation for c in self.crystals):

                    # ...at 45 degrees to the polarisers?
                    if abs(pol_1_orientation - crystal_1_orientation) == np.pi / 4:
                        return 'two-beam'

        return 'general'

    def calculate_ideal_phase_offset(self, wl, n_e=None, n_o=None):
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

