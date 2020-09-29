import numpy as np
import xarray as xr
import os.path
import scipy.interpolate
from scipy.constants import c
import pycis
from pycis.tools import is_scalar
from pycis.model import mueller_product


class Instrument(object):
    """
    coherence imaging instrument

    """

    def __init__(self, camera, optics, interferometer, bandpass_filter=None, interferometer_orientation=0):
        """
        TODO some of the methods of this class take 'spectrum' as an input, but do not use the values of spectrum.
            Rather, they just use the information from the coordinates. This probably makes the code harder to
            understand.

        :param camera:
        :type camera pycis.model.Camera

        :param optics: the focal lengths of the three lenses used in the standard CI configuration (see e.g. my thesis
        or Scott Silburn's): [f_1, f_2, f_3] where f_1 is the objective lens.
        :type optics: list of floats

        :param interferometer: a list of instances of pycis.model.InterferometerComponent. The first component in
        the list is the first component that the light passes through.
        :type interferometer: list

        :param bandpass_filter: pycis.model.BandpassFilter or string-type filter name (not actually implemented
        properly yet)
        """

        self.camera = camera
        self.optics = optics
        self.interferometer = interferometer
        self.crystals = self.get_crystals()
        self.polarisers = self.get_polarisers()
        self.interferometer_orientation = interferometer_orientation

        # TODO properly implement bandpass_filter
        self.bandpass_filter = bandpass_filter
        self.input_checks()
        self.x_pos, self.y_pos = self.calculate_pixel_position()

        # assign instrument 'type'
        self.instrument_type = self.check_instrument_type()

    def input_checks(self):
        assert isinstance(self.camera, pycis.model.Camera)
        assert isinstance(self.optics, list)
        assert all(isinstance(c, pycis.model.InterferometerComponent) for c in self.interferometer)
        if self.bandpass_filter is not None:
            assert isinstance(self.bandpass_filter, pycis.model.BandpassFilter)

    def get_crystals(self):
        return [c for c in self.interferometer if isinstance(c, pycis.BirefringentComponent)]

    def get_polarisers(self):
        return [c for c in self.interferometer if isinstance(c, pycis.LinearPolariser)]

    def calculate_pixel_position(self, x_pixel=None, y_pixel=None, crop=None, downsample=1):
        """
        Calculate pixel positions (in m) on the camera's sensor plane (the x-y plane).

        The origin of the x-y coordinate system is the centre of the sensor. Pixel positions correspond to the pixel
        centres. If x_pixel and y_pixel are specified then only returns the position of that pixel. crop and downsample
        are essentially legacy kwargs from my thesis work.

        :param x_pixel:
        :param y_pixel:
        :param crop: (y1, y2, x1, x2)
        :param downsample:
        :return: x_pos, y_pos, both instances of xr.DataArray
        """

        sensor_format = np.array(self.camera.sensor_format)
        centre_pos = self.camera.pixel_size * sensor_format / 2  # relative to x=0, y=0 pixel

        if crop is None:
            crop = (0, sensor_format[0], 0, sensor_format[1])

        x_coord = np.arange(crop[0], crop[1])[::downsample]
        y_coord = np.arange(crop[2], crop[3])[::downsample]
        x = (x_coord + 0.5) * self.camera.pixel_size - centre_pos[0]
        y = (y_coord + 0.5) * self.camera.pixel_size - centre_pos[1]
        x = xr.DataArray(x, dims=('x', ), coords=(x, ), )
        y = xr.DataArray(y, dims=('y',), coords=(y,), )

        # add pixel numbers as non-dimension coordinates -- just for explicit indexing and plotting
        x_pixel_coord = xr.DataArray(np.arange(sensor_format[0], ), dims=('x', ), coords=(x, ), )
        y_pixel_coord = xr.DataArray(np.arange(sensor_format[1], ), dims=('y', ), coords=(y, ), )
        x = x.assign_coords({'x_pixel': ('x', x_pixel_coord), }, )
        y = y.assign_coords({'y_pixel': ('y', y_pixel_coord), }, )

        if x_pixel is not None:
            x = x.isel(x=x_pixel)
        if y_pixel is not None:
            y = y.isel(y=y_pixel)

        return x, y

    def calculate_inc_angles(self, x, y):
        """
        calculate incidence angles of ray(s) through crystal

        :param x: (xr.DataArray) x position(s) on sensor plane [ m ]
        :param y: (xr.DataArray) y position(s) on sensor [ m ]
        :return: incidence angles [ rad ]

        """

        return np.arctan2(np.sqrt(x ** 2 + y ** 2), self.optics[2], )

    def calculate_azim_angles(self, x, y, crystal):
        """
        calculate azimuthal angles of rays through crystal (varies with crystal orientation)

        :param x: (xr.DataArray) x position(s) on sensor plane [ m ]
        :param y: (xr.DataArray) y position(s) on sensor plane [ m ]
        :param crystal: (pycis.model.BirefringentComponent)
        :return: azimuthal angles [ rad ]

        """

        orientation = crystal.orientation + self.interferometer_orientation
        return np.arctan2(y, x) + np.pi - orientation

    def calculate_matrix(self, spectrum):
        """
        calculate the total Mueller matrix for the interferometer

        :param spectrum: (xr.DataArray) see spectrum argument for instrument.capture
        :return: Mueller matrix

        """

        inc_angle = self.calculate_inc_angles(spectrum.x, spectrum.y)
        total_matrix = xr.DataArray(np.identity(4), dims=('mueller_v', 'mueller_h'), )

        for component in self.interferometer:
            azim_angle = self.calculate_azim_angles(spectrum.x, spectrum.y, component)
            component_matrix = component.calculate_matrix(spectrum.wavelength, inc_angle, azim_angle)
            total_matrix = mueller_product(component_matrix, total_matrix)

        rot_matrix = pycis.calculate_rot_matrix(self.interferometer_orientation)
        return mueller_product(rot_matrix, total_matrix)

    def capture_image(self, spectrum, ):
        """
        capture image of scene

        :param spectrum: (xr.DataArray) photon fluence spectrum with units of ph / m [hitting the pixel area during exposure
         time] and with dimensions 'wavelength', 'x', 'y' and (optionally) 'stokes'. If no stokes dim then it is assumed
        that light is unpolarised (i.e. the spec supplied is the S_0 Stokes parameter only)
        :param color: (bool) true for color display, else monochrome
        :return:

        """

        if self.instrument_type == 'two_beam' and 'stokes' not in spectrum.dims:
            # analytical calculation to save time
            print('1')
            total_intensity = spectrum.integrate(dim='wavelength', )

            spec_freq = spectrum.rename({'wavelength': 'frequency'})
            spec_freq['frequency'] = c / spec_freq['frequency']
            spec_freq = spec_freq * c / spec_freq['frequency'] ** 2
            freq_com = (spec_freq * spec_freq['frequency']).integrate(dim='frequency') / total_intensity
            delay = self.calculate_ideal_delay(c / freq_com)

            coherence = xr.where(total_intensity > 0,
                                 pycis.calculate_coherence(spec_freq, delay, material=self.crystals[0].material,
                                                           freq_com=freq_com),
                                 0)
            spectrum = 1 / 4 * (total_intensity + xr.ufuncs.real(coherence))

        elif self.instrument_type == 'general':
            # full Mueller matrix calculation
            if 'stokes' not in spectrum.dims:
                a0 = xr.zeros_like(spectrum)
                spectrum = xr.combine_nested([spectrum, a0, a0, a0], concat_dim=('stokes',))

            mueller_mat = self.calculate_matrix(spectrum)
            spectrum = mueller_product(mueller_mat, spectrum)

        image = self.camera.capture_image(spectrum)
        return image

    def calculate_ideal_delay(self, wavelength, ):
        """
        calculate the interferometer phase delay (in rad) at the given wavelength(s)

        assumes all crystal's phase contributions combine constructively -- method used only when instrument.type =
        'two-beam'. kwargs included for fitting purposes.

        TODO This method needs to be more general if its going to be properly useful
        :param spectrum:
        :return:
        """

        # calculate the angles of each pixel's line of sight through the interferometer
        inc_angles = self.calculate_inc_angles(wavelength.x, wavelength.y)
        azim_angles = self.calculate_azim_angles(wavelength.x, wavelength.y, self.crystals[0])

        # calculate phase delay contribution due to each crystal
        phase = 0
        for crystal in self.crystals:
            phase += crystal.calculate_delay(wavelength, inc_angles, azim_angles, )

        return phase

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
        instrument_type determines best way to calculate the interference pattern

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
                crystal_1_material = self.crystals[0].material

                if all(c.orientation == crystal_1_orientation for c in self.crystals) and \
                        all(c.material == crystal_1_material for c in self.crystals):

                    # ...at 45 degrees to the polarisers?
                    if abs(pol_1_orientation - crystal_1_orientation) == np.pi / 4:

                        # ...and is the camera a standard camera?
                        if not isinstance(self.camera, pycis.PolCamera):
                            return 'two_beam'

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
            phase_offset += crystal.calculate_delay(wl, 0., 0., n_e=n_e, n_o=n_o)

        return -phase_offset