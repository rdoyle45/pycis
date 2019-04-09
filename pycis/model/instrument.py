import numpy as np
import pycis


class Instrument:
    """
    Coherence imaging instrument base class

    """
    def __init__(self, camera, back_lens, interferometer, bandpass_filter=None, int_orient=0):
        """
        :param camera:
        :type camera pycis.model.Camera

        :param back_lens:
        :type back_lens: pycis.model.Lens

        :param interferometer: a list of instances of pycis.model.InterferometerComponent. The first component in
        the list is the first component that the light passes through.
        :type interferometer: list

        :param bandpass_filter: pycis.model.BandpassFilter or string-type filter name
        :type bandpass_filter: pycis.model.BandpassFilter

        :param int_orient: interferometer orientation [ rad ]
        :type int_orient: float
        """

        # input checks
        assert isinstance(camera, pycis.model.Camera)
        assert isinstance(back_lens, pycis.model.Lens)
        assert isinstance(interferometer, list)
        assert all(isinstance(c, pycis.model.InterferometerComponent) for c in interferometer)
        if bandpass_filter is not None:
            assert isinstance(bandpass_filter, pycis.model.BandpassFilter)

        self.camera = camera
        self.back_lens = back_lens
        self.interferometer = interferometer
        self.bandpass_filter = bandpass_filter
        self.int_orient = int_orient

        self.crystals = self._get_crystals()
        self.polarisers = self._get_polarisers()
        self.x_pos, self.y_pos = self._calculate_pixel_pos()
        self.inst_mode = self.check_inst_mode()

        # TODO implement bandpass_filter properly
        # TODO implement crystal misalignment
        # TODO order and position of interferometer components

    def make_image(self, wl, spec):
        """
        :param wl: wavelengths [ m ]
        :type wl: Union[float, np.ndarray]

        :param spec: wavelength spectrum of the light observed [ ph / pixel / m / exposure ]
        :type spec: Union[int, float, np.ndarray]

        :return:
        """
        raise NotImplementedError

    def _get_crystals(self):
        """
        list birefringent components present in the interferometer ( subset of self.interferometer list )
        """

        return [c for c in self.interferometer if isinstance(c, pycis.BirefringentComponent)]

    def _get_polarisers(self):
        """
        list polarisers present in the interferometer ( subset of self.interferometer list )
        """

        return [c for c in self.interferometer if isinstance(c, pycis.LinearPolariser)]

    def _calculate_pixel_pos(self, x_coord=None, y_coord=None, crop=None, downsample=1):
        """
        Calculate x-y coordinates of the pixels of the camera's sensor -- for ray geometry calculations
        Converts from pixel coordinates (origin top left) to spatial coordinates (origin centre). If x_coord and
        y_coord are not specified, entire sensor will be evaluated (with specified cropping and downsampling).

        Coordinates can be cropped and downsampled to facilitate direct comparison with cropped / downsampled
        experimental image, forgoing the need to generate a full-sensor synthetic image.

        can this be a pycis.Camera method?

        If both crop and downsample are specified, crop is carried out first.

        :param x_coord: array of pixel coordinates (x)
        :param y_coord: array of pixel coordinates (y)
        :param crop: (y1, y2, x1, x2)
        :param downsample:
        :return: x_pos, y_pos [ m ]
        """

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

        return x_pos, y_pos

    def _calculate_ray_inc_angles(self, x_pos, y_pos):
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

        orientation = crystal.orientation + self.int_orient

        # rotate x, y coordinates
        x_rot = (np.cos(orientation) * x_pos) + (np.sin(orientation) * y_pos)
        y_rot = (- np.sin(orientation) * x_pos) + (np.cos(orientation) * y_pos)

        return np.arctan2(x_rot, y_rot)


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

        return phase_offset


class MatrixInstrument(Instrument):
    """
    Instrument whose interferogram is calculated using Mueller calculus

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_matrix(self, wl):
        """
        calculate Mueller matrix for interferometer

        :param wl: [ m ]
        :type wl: np.ndarray

        :return: instrument matrix
        """

        x_pos, y_pos = self._calculate_pixel_pos()
        inc_angle = self._calculate_ray_inc_angles(x_pos, y_pos)

        subscripts = 'ij...,jl...->il...'
        instrument_matrix = np.identity(4)

        for component in self.interferometer:
            azim_angle = self.calculate_ray_azim_angles(x_pos, y_pos, component)
            component_matrix = component.calculate_matrix(wl, inc_angle, azim_angle)

            # matrix multiplication
            instrument_matrix = np.einsum(subscripts, component_matrix, instrument_matrix)

        # account for orientation of the interferometer itself.
        rot_mat = pycis.model.calculate_rot_mat(self.int_orient)

        return np.einsum(subscripts, rot_mat, instrument_matrix)


class AnalyticInstrument(Instrument):
    """
    Instrument whose interferogram is generated analytically

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_phase(self, wl, x_coord=None, y_coord=None, n_e=None, n_o=None, crop=None, downsample=1,
                        output_components=False):
        raise NotImplementedError

    def calculate_contrast(self, wl, spec):
        raise NotImplementedError

    def calculate_brightness(self, wl, spec):
        raise NotImplementedError

    def calculate_tx(self):
        raise NotImplementedError


class SimpleCisInstrument(AnalyticInstrument):
    """
    Special-case coherence imaging instrument for simple CIS

    the birefringent interferometer components are all aligned / orthogonal and are sandwiched between aligned /
    orthogonal linear polarisers, which are offset by 45 degrees.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_phase(self, wl, x_coord=None, y_coord=None, n_e=None, n_o=None, crop=None, downsample=1,
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
        x_pos, y_pos = self._calculate_pixel_pos(x_coord=x_coord, y_coord=y_coord, crop=crop, downsample=downsample)

        inc_angles = self._calculate_ray_inc_angles(x_pos, y_pos)
        azim_angles = self.calculate_ray_azim_angles(x_pos, y_pos, self.crystals[0])

        # calculate phase delay contribution due to each crystal
        phase = 0
        for crystal in self.crystals:
            phase += crystal.calculate_phase_delay(wl, inc_angles, azim_angles, n_e=n_e, n_o=n_o)

        return phase

    def calculate_contrast(self):
        """

        :return:
        """

        contrast = 1
        for crystal in self.crystals:
            contrast *= crystal.contrast

        return contrast

    def make_image(self, wl, spec):
        """

        :param wl:
        :param spec:
        :return:
        """

        i0 = np.trapz(spec, wl, axis=0)
        spec_norm = np.divide(spec, i0, where=i0 > 0)

        phase = self.calculate_phase(wl)
        contrast = self.calculate_contrast()
        degree_coherence = np.trapz(spec_norm * contrast * np.exp(1j * phase), wl, axis=0)

        igram = i0 / 4 * (1 + np.real(degree_coherence))

    def calculate_tx(self):
        """
        transmission is decreased due to polarisers, calculate this factor analytically for the special case of
        ideal interferometer

        """

        pol_1, pol_2 = self.polarisers
        tx = (pol_1.tx_1 ** 2 + pol_1.tx_2 ** 2) * (pol_2.tx_1 ** 2 + pol_2.tx_2 ** 2)

        return tx

class SpectroPolInstrument(Instrument):
    """

    """

