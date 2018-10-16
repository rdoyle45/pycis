import numpy as np
import pickle
import os.path
import pycis.paths
from pycis.tools.coord_transform import *
from pycis.model.phase_delay import uniaxial_crystal, savart_plate
import scipy.interpolate



class Instrument(object):
    """ Stores information on CIS optical componenents + optical setup. Currently, only the presence of the added
    config can be specified, not the position within the instrument.
     
     Currently, must have Waveplate, Savart plate, an final lens added in order to be used. """

    def __init__(self, name, instrument_contrast):
        """
        :param name: 
     
        """
        self.name = name

        self.camera = None
        self.lens_3 = None
        self.optical_filter = None
        self.waveplate = None
        self.wp_orientation = None
        self.savartplate = None
        self.sp_orientation = None

        self.instrument_contrast = instrument_contrast

        self.chi = [0, 0]  # placeholder, this will be gotten rid of in time


    def add_camera(self, name):
        self.camera = pycis.model.load_component(name, type='camera')

    def add_lens_3(self, name):
        self.lens_3 = pycis.model.load_component(name, type='lens')
        return

    def add_filter(self, name):
        self.optical_filter = pycis.model.load_component(name, type='filter')
        return

    def add_crystal(self, name, orientation):
        """

        :param name: name of saved instance of Crystal
        :param orientation: orientation of crystal relative to instrument y-axis, see docs.
        :return: 
        """

        crystal = pycis.model.load_component(name, type='crystal')

        # check crystal type:

        if isinstance(crystal, pycis.model.Waveplate):
            self.waveplate = crystal
            self.wp_orientation = orientation

        elif isinstance(crystal, pycis.model.Savartplate):
            self.savartplate = crystal
            self.sp_orientation = orientation

        return

    def get_angles(self, display=False):
        """
        Calculates the distributions of incidence angle and azimuthal angle for each crystal component, as projected onto
        the camera's sensor.
        
        returns in radians.
        """

        # some shorthand:

        cam = self.camera
        f_3 = self.lens_3.focal_length

        # Define x,y detector coordinates:

        tilt_offset_y = f_3 * np.tan(self.chi[0])  # vertical tilt
        tilt_offset_x = f_3 * np.tan(self.chi[1])  # horizontal tilt

        centre = [(cam.pix_size * (cam.sensor_dim[0] - 2) / 2) + tilt_offset_y,
                  (cam.pix_size * (cam.sensor_dim[1] - 2) / 2) + tilt_offset_x]
        y_pos = np.arange(0, cam.sensor_dim[0])
        y_pos = (y_pos - 0.5) * cam.pix_size - centre[0]  # [m]
        x_pos = np.arange(0, cam.sensor_dim[1])
        x_pos = (x_pos - 0.5) * cam.pix_size - centre[1]  # [m]

        x, y = np.meshgrid(x_pos, y_pos)

        # assuming for now that waveplate and Savart plate are parallel and perfectly alligned, their incidence angle
        # projections are now the same.

        inc_angles = np.arctan2(np.sqrt(x ** 2 + y ** 2), f_3)

        # azimuthal angles may differ so must be calculated separately:

        # Rotate x, y coordinates by crystal orientation angle:

        x_rot_wp = (np.cos(self.wp_orientation) * x) + (np.sin(self.wp_orientation) * y)
        y_rot_wp = (- np.sin(self.wp_orientation) * x) + (np.cos(self.wp_orientation) * y)

        x_rot_sp = (np.cos(self.sp_orientation) * x) + (np.sin(self.sp_orientation) * y)
        y_rot_sp = (- np.sin(self.sp_orientation) * x) + (np.cos(self.sp_orientation) * y)

        # calculate incidence and azimuthal angles:

        azim_angles_wp = np.arctan2(x_rot_wp, y_rot_wp)
        azim_angles_sp = np.arctan2(x_rot_sp, y_rot_sp)


        if display:

            fig1 = plt.figure()
            ax1 = fig1.add_subplot(121)
            ax2 = fig1.add_subplot(122)

            im1 = ax1.imshow(inc_angles)
            cbar = plt.colorbar(im1, ax=ax1)

            im2 = ax2.imshow(azim_angles_wp)
            cbar = plt.colorbar(im2, ax=ax2)


            plt.show()

        return inc_angles, azim_angles_wp, azim_angles_sp

    def get_snr_intensity(self, line_name, snr):
        """ Given spectral line and desired approximate image snr (central ROI), return the necessary input intensity I0 in units 
        of [photons/pixel/timestep]. """

        # load line
        line = pycis.model.Lineshape(line_name, 1, 0, 1)

        _, _, _, wavelength_com = line.make_curve(1000)

        if self.optical_filter is not None:
            # estimate filter transmission at line wavelength
            t_filter = self.optical_filter.get_transmission(wavelength_com)
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

        _, _, _, wavelength_com = line.make_curve(1000)

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

if __name__ == '__main__':

    # inst = pycis.model.Instrument('mastu_cdi_508nm')
    #
    # inst.add_camera('photron_SA4')
    # inst.add_lens_3('sigma_150mm')
    # inst.add_filter('ccfe_CII_m')
    #
    # inst.add_crystal('ccfe_4.6mm_waveplate', orientation=3 * np.pi / 4)
    # inst.add_crystal('ccfe_6.2mm_savartplate', orientation=1 * np.pi / 4)
    #
    # inst.save()
    inst = pycis.model.load_component('mastu_ciii', type='instrument')
    inst.apply_vignetting()

