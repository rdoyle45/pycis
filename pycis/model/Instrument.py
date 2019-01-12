import numpy as np
import os.path
import pycis.paths
import copy
import scipy.interpolate


class Instrument(object):
    """ 
    Coherence imaging instrument class, facilitates synthetic image generation
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
        self.bandpass_filter = bandpass_filter
        self.input_checks()

        self.x, self.y = self.get_sensor_coords()

        # assign instrument 'type' based on interferometer layout
        self.inst_type = self.check_instrument_type()

        # TODO include crystal misalignment
        self.chi = [0, 0]  # placeholder

    def input_checks(self):
        """
        checks for __init__ arguments
        :return: 
        """

        assert isinstance(self.camera, pycis.model.Camera)
        assert isinstance(self.back_lens, pycis.model.Lens)
        assert all(isinstance(c, pycis.model.InterferometerComponent) for c in self.interferometer)

        if self.bandpass_filter is not None:
            assert isinstance(self.bandpass_filter, pycis.model.BandpassFilter)

    def get_crystals(self):
        """ list the birefringent components, subset of interferometer """

        return [c for c in self.interferometer if isinstance(c, pycis.BirefringentComponent)]

    def get_polarisers(self):
        """ get a list of the polarisers, subset of interferometer """

        return [c for c in self.interferometer if isinstance(c, pycis.LinearPolariser)]

    def get_sensor_coords(self, downsample=None, letterbox=None):
        """  
        can this be a pycis.Camera method? 
        
        :param downsample: 
        :param letterbox: 
        :return: 
        """

        cam = self.camera
        sensor_dim = np.array(list(copy.copy(cam.sensor_dim)))

        if letterbox is not None:
            sensor_dim[0] -= 2 * letterbox

        centre = cam.pix_size * (sensor_dim - 2) / 2

        if downsample is None:
            y_pos = np.arange(sensor_dim[0])
            x_pos = np.arange(sensor_dim[1])
        else:
            y_pos = np.arange(sensor_dim[0])[::downsample]
            x_pos = np.arange(sensor_dim[1])[::downsample]

        y_pos = (y_pos - 0.5) * cam.pix_size - centre[0]  # [m]
        x_pos = (x_pos - 0.5) * cam.pix_size - centre[1]  # [m]

        x, y = np.meshgrid(x_pos, y_pos)

        return x, y

    def calculate_ray_inc_angle(self):
        """
        incidence angles will be the same for all interferometer components (until arbitrary component misalignment 
        implemented)
        
        :param downsample: 
        :param letterbox: 
        :return: 
        """

        # assuming crystals are perfectly alligned, their incidence angle projections are the same.
        inc_angles = np.arctan2(np.sqrt(self.x ** 2 + self.y ** 2), self.back_lens.focal_length)

        return inc_angles

    def calculate_ray_azim_angle(self, crystal):
        """
        azimuthal angles vary with crystal orientation so are calculated separately
        
        :param crystal: 
        :return: 
        """

        orientation = crystal.orientation + self.interferometer_orientation

        # Rotate x, y coordinates
        x_rot = (np.cos(orientation) * self.x) + (np.sin(orientation) * self.y)
        y_rot = (- np.sin(orientation) * self.x) + (np.cos(orientation) * self.y)

        return np.arctan2(x_rot, y_rot)

    def calculate_ray_angle(self, downsample=None, letterbox=None):
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
            y_pos = np.arange(sensor_dim[0])
            x_pos = np.arange(sensor_dim[1])
        else:
            y_pos = np.arange(sensor_dim[0])[::downsample]
            x_pos = np.arange(sensor_dim[1])[::downsample]

        y_pos = (y_pos - 0.5) * cam.pix_size - centre[0]  # [m]
        x_pos = (x_pos - 0.5) * cam.pix_size - centre[1]  # [m]

        x, y = np.meshgrid(x_pos, y_pos)

        # assuming crystals are perfectly alligned, their incidence angle projections are the same.
        inc_angles = np.arctan2(np.sqrt(x ** 2 + y ** 2), f_3)

        # azimuthal angles vary with crystal so are calculated separately
        crystal_azim_angles = []

        for crystal in self.crystals:

            orientation = crystal.orientation + self.interferometer_orientation

            # Rotate x, y coordinates by crystal orientation
            x_rot = (np.cos(orientation) * x) + (np.sin(orientation) * y)
            y_rot = (- np.sin(orientation) * x) + (np.cos(orientation) * y)

            crystal_azim_angles.append(np.arctan2(x_rot, y_rot))

        return inc_angles, crystal_azim_angles

    def calculate_matrix(self, wl):
        """ calculate the total Mueller matrix for the interferometer """

        inc_angle = self.calculate_ray_inc_angle()

        instrument_matrix = np.identity(4)
        subscripts = 'ij...,jl...->il...'

        for component in self.interferometer:

            azim_angle = self.calculate_ray_azim_angle(component)
            component_matrix = component.calculate_matrix(wl, inc_angle, azim_angle)

            # matrix multiplication
            instrument_matrix = np.einsum(subscripts, component_matrix, instrument_matrix)

        # account for orientation of the interferometer itself.
        rot_mat = pycis.model.calculate_rot_mat(self.interferometer_orientation)
        return np.einsum(subscripts, rot_mat, instrument_matrix)

    def calculate_ideal_phase_delay(self, wl, n_e=None, n_o=None, downsample=None, letterbox=None, output_components=False):
        """
        TAKE CARE -- method assumes that all instances of pycis.model.InterferomterComponent supplied to 
        Instrument are aligned, such that their phase delays add constructively. If you are playing with arbitrary 
        combinations and orientations of crystal then this will not be accurate. 
        
        :param wl: 
        :param n_e: 
        :param n_o: 
        :param downsample: 
        :param letterbox: 
        :param output_components: 
        :return: 
        """

        # calculate the angles of each pixel's line of sight through the interferometer
        inc_angles, crystal_azim_angles = self.calculate_ray_angle(downsample=downsample, letterbox=letterbox)

        # calculate phase delay contribution due to each crystal
        phase = 0
        for crystal, azim_angles in zip(self.crystals, crystal_azim_angles):
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
        find the best way to generate the observed interference pattern
        
        in the case of a perfectly aligned coherence imaging diagnostic in a simple 'two-beam' configuration, skip the 
        Mueller matrix calculation to the final result.
        
        :return: type
        """

        orientations = []
        for crystal in self.crystals:
            orientations.append(crystal.orientation)

        # if all crystal orientations are the same and are at 45 degrees to the polarisers, perform a simplified 2-beam
        # interferometer calculation -- avoiding the full Mueller matrix treatment

        if all(o == orientations[0] for o in orientations) and (orientations[0] + np.pi / 4) % (np.pi / 2) == 0:
            inst_type = 'two-beam'
        else:
            inst_type = 'general'

        print(inst_type)
        return inst_type

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

    def save(self):
        pickle.dump(self, open(os.path.join(pycis.paths.instrument_path, self.name + '.p'), 'wb'))
        return
