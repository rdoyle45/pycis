import numpy as np
import xarray as xr
import os.path
import scipy.interpolate
import pycis
from pycis.tools import is_scalar
from pycis.model import mueller_prod


class Instrument(object):
    """
    CI instrument

    """

    def __init__(self, camera, back_lens, interferometer, bandpass_filter=None, interferometer_orientation=0):
        """
        :param camera:
        :type camera pycis.model.Camera

        :param back_lens:
        :type back_lens: pycis.model.Lens

        :param interferometer: a list of instances of pycis.model.InterferometerComponent. The first component in
        the list is the first component that the light passes through.
        :type interferometer: list

        :param bandpass_filter: pycis.model.BandpassFilter or string-type filter name (not actually implemented
        properly yet)
        """

        self.camera = camera
        self.back_lens = back_lens
        self.interferometer = interferometer
        self.crystals = self.get_crystals()
        self.polarisers = self.get_polarisers()
        self.interferometer_orientation = interferometer_orientation

        # TODO implement bandpass_filter
        self.bandpass_filter = bandpass_filter
        self.input_checks()

        self.x_pos, self.y_pos = self.calculate_pixel_pos()

        # assign instrument 'mode'
        self.inst_mode = self.check_inst_mode()

        # TODO implement crystal misalignment

    def input_checks(self):
        assert isinstance(self.camera, pycis.model.Camera)
        assert isinstance(self.back_lens, pycis.model.Lens)
        assert all(isinstance(c, pycis.model.InterferometerComponent) for c in self.interferometer)
        if self.bandpass_filter is not None:
            assert isinstance(self.bandpass_filter, pycis.model.BandpassFilter)

    def get_crystals(self):
        return [c for c in self.interferometer if isinstance(c, pycis.BirefringentComponent)]

    def get_polarisers(self):
        return [c for c in self.interferometer if isinstance(c, pycis.LinearPolariser)]

    def calculate_pixel_pos(self, crop=None, downsample=1):
        """
        Calculate x-y coordinates of the pixels of the camera's sensor -- for ray geometry calculations
        Converts from pixel coordinates (origin top left) to spatial coordinates (origin centre). If x_coord and
        y_coord are not specified, entire sensor will be evaluated (with specified cropping and downsampling).

        Coordinates can be cropped and downsampled. If both crop and downsample are specified, crop is carried out
        first.

        :param crop: (y1, y2, x1, x2)
        :param downsample:
        :return: x_pos, y_pos [ m ]

        # TODO rewrite for arbitrary sensor subsections

        """

        sensor_dim = np.array(self.camera.sensor_dim)
        centre_pos = self.camera.pix_size * (sensor_dim - 2) / 2

        if crop is None:
            crop = (0, sensor_dim[0], 0, sensor_dim[1])

        y_coord = np.arange(crop[0], crop[1])[::downsample]
        x_coord = np.arange(crop[2], crop[3])[::downsample]
        y = (y_coord - 0.5) * self.camera.pix_size - centre_pos[0]
        x = (x_coord - 0.5) * self.camera.pix_size - centre_pos[1]
        y = xr.DataArray(y, dims=('y', ), coords=(y, ), )
        x = xr.DataArray(x, dims=('x', ), coords=(x, ), )
        return x, y

    def calculate_inc_angles(self, x_pos, y_pos):
        """
        calculate incidence angles of ray(s) through crystal

        :param x_pos: (xr.DataArray) x position(s) on sensor plane [ m ]
        :param y_pos: (xr.DataArray) y position(s) on sensor [ m ]
        :return: incidence angles [ rad ]

        """

        return np.arctan2(np.sqrt(x_pos ** 2 + y_pos ** 2), self.back_lens.focal_length)

    def calculate_azim_angles(self, x_pos, y_pos, crystal):
        """
        calculate azimuthal angles of rays through crystal (varies with crystal orientation)

        :param x_pos: (xr.DataArray) x position(s) on sensor plane [ m ]
        :param y_pos: (xr.DataArray) y position(s) on sensor [ m ]
        :param crystal: (pycis.model.BirefringentComponent)
        :return: azimuthal angles [ rad ]

        """

        orientation = crystal.orientation + self.interferometer_orientation
        return np.arctan2(y_pos, x_pos) + np.pi - orientation

    def calculate_matrix(self, spec):
        """
        calculate total Mueller matrix for interferometer

        :param spec: (xr.DataArray) see spec argument for instrument.capture
        :return: Mueller matrix

        """

        inc_angle = self.calculate_inc_angles(spec.x, spec.y)
        instrument_mat = xr.DataArray(np.identity(4), dims=('mueller_v', 'mueller_h'), )

        for component in self.interferometer:
            azim_angle = self.calculate_azim_angles(spec.x, spec.y, component)
            component_mat = component.calculate_matrix(spec.wavelength, inc_angle, azim_angle)
            instrument_mat = mueller_prod(component_mat, instrument_mat)

        rot_mat = pycis.model.calculate_rot_mat(self.interferometer_orientation)
        return mueller_prod(rot_mat, instrument_mat)

    def capture(self, spec, display=False, color=False, ):
        """
        capture image of scene

        :param spec: (xr.DataArray) input spectrum with dimensions 'wavelength', 'x', 'y' and (optionally) 'stokes'. If
        no stokes dim then it is assumed that light is unpolarised (i.e. the spec supplied is the S_0 Stokes parameter)
        :param display: (bool) whether to display
        :param color: (bool) true for color display, else monochrome
        :return:

        """

        if 'stokes' not in spec.dims:
            a0 = xr.zeros_like(spec)
            spec = xr.combine_nested([spec, a0, a0, a0], concat_dim=('stokes', ))

        mueller_mat = self.calculate_matrix(spec)
        spec = mueller_prod(mueller_mat, spec)

        image = self.camera.capture(spec)
        return image

    def calculate_ideal_phase_delay(self, wl, x_coord=None, y_coord=None, n_e=None, n_o=None, crop=None, downsample=1,
                                    output_components=False):
        """
        assumes all crystal's phase contributions combine constructively -- method used only when instrument.type =
        'two-beam'. kwargs included for fitting purposes.
        :param wl:
        :param x_coord:
        :param y_coord:
        :param n_e:
        :param n_o:
        :param crop:
        :param downsample:
        :param output_components:
        :return: phase delay [ rad ]
        """

        # calculate the angles of each pixel's line of sight through the interferometer
        x_pos, y_pos = self.calculate_pixel_pos(x_coord=x_coord, y_coord=y_coord, crop=crop, downsample=downsample)
        inc_angles = self.calculate_inc_angles(x_pos, y_pos)
        azim_angles = self.calculate_azim_angles(x_pos, y_pos, self.crystals[0])

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

    def check_inst_mode(self):
        """
        instrument mode determines best way to calculate the interference pattern

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

            # ...are they alligned?
            pol_1_orientation = self.interferometer[0].orientation
            pol_2_orientation = self.interferometer[-1].orientation

            if pol_1_orientation == pol_2_orientation:

                # ...are all crystals alligned?
                crystal_1_orientation = self.crystals[0].orientation

                if all(c.orientation == crystal_1_orientation for c in self.crystals):

                    # ...at 45 degrees to the polarisers?
                    if abs(pol_1_orientation - crystal_1_orientation) == np.pi / 4:

                        # ...and is the camera a standard camera?
                        if not isinstance(self.camera, pycis.PolCamera):
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

        return -phase_offset

    def apply_vignetting(self, photon_fluence):
        """account for instrument etendue to first order based on Scott's predictive matlab code. """

        # TODO old, clean up / throw away

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

        # w1d = np.interp(sensor_diagonal_axis, sensor_distance_sym, norm_etendue_sym)

        l = self.camera.sensor_dim[1]
        m = sensor_half_width
        xx = np.linspace(-m, m, l)

        [x, y] = np.meshgrid(xx, xx)
        r = np.sqrt(x ** 2 + y ** 2)
        w2d_2 = np.zeros([l, l])
        w2d_2[r <= sensor_half_diagonal] = np.interp(r[r <= sensor_half_diagonal], sensor_diagonal_axis, w1d)

        w2d_2[w2d_2 < 0] = 0

        # w2d_2 = pycis.tools.gauss2d(self.camera.sensor_dim[1], 1200)

        vignetted_photon_fluence = photon_fluence * w2d_2
        vignetted_photon_fluence[vignetted_photon_fluence < 0] = 0

        return vignetted_photon_fluence