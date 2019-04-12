import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

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
        self.inst_mode = self.check_inst_mode()

        # TODO implement bandpass_filter properly
        # TODO implement crystal misalignment
        # TODO position of interferometer components

    def make_image(self, spec, savepath=None):
        """
        calculate the intensity pattern (interferogram) at the output of the interferometer and capture an image.

        :param spec:
        :type spec: xr.DataArray

        :param savepath:
        :type savepath: str

        :return:
        """

        igram = self.make_igram(spec)
        image = self.camera.capture(igram)

        if savepath is not None:
            fig, ax = plt.subplots(111)
            image.plot(ax=ax)
            plt.savefig(savepath)

        return image

    def make_igram(self, spec):
        """

        :param spec:
        :return:
        """
        raise NotImplementedError

    def _input_checks(self):
        """

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

    def _get_ray_angles(self, component):
        """
        get the angles of incidence and azimuth for the specified interferometer component

        :param crystal:
        :type crystal: pycis.model.InterferometerComponent

        :return: ray_inc_angles [ rad ], ray_azim_angles [ rad ]
        """

        # TODO specify misalignments here

        # incidence angles
        ray_inc_angle = np.arctan2(np.sqrt(self.camera.x ** 2 + self.camera.y ** 2), self.back_lens.focal_length)

        # azimuthal angles
        orientation = component.orientation + self.int_orient
        ray_azim_angle = np.arctan2(np.cos(orientation) * self.camera.x + np.sin(orientation) * self.camera.y,
                                     -np.sin(orientation) * self.camera.x + np.cos(orientation) * self.camera.y)

        return ray_inc_angle, ray_azim_angle

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


class MatrixInstrument(Instrument):
    """
    Instrument whose interferogram is calculated using Mueller calculus. Used primarily for consistency checking
    AnalyticInstrument subclasses, and for modelling optical misalignments.

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

    def calculate_phase(self, spec):
        raise NotImplementedError

    def calculate_contrast(self, spec):
        raise NotImplementedError

    # def calculate_brightness(self, spec):
    #     raise NotImplementedError

    # def calculate_tx(self):
    #     raise NotImplementedError


class SimpleCisInstrument(AnalyticInstrument):
    """
    Special-case coherence imaging instrument for simple CIS

    the birefringent interferometer components are all aligned / orthogonal and are sandwiched between aligned /
    orthogonal linear polarisers, which are offset by 45 degrees.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_igram(self, spec):
        """
        calculation follows p35 of S. Silburn's thesis

        :param spec:
        :type spec: xr.DataArray
        :return:
        """

        # input checks
        assert isinstance(spec, xr.DataArray)
        assert 'wavelength' in spec.dims
        assert 'x' in spec.dims
        assert 'y' in spec.dims

        i0 = spec.sum('wavelength')
        spec_norm = np.divide(spec, i0)
        phase = self.calculate_phase(spec['wavelength'])
        contrast = self.calculate_contrast(spec)
        degree_coherence = (spec_norm * contrast * np.exp(1j * phase))

        return i0 / 4 * (1 + degree_coherence.real)

    def calculate_phase(self, wavelength):
        """

        :param wavelength:
        :type wavelength: xr.DataArray

        :return: phase [ rad ]
        """

        # calculate phase delay contribution due to each crystal
        # TODO subtract phase for anti-aligned components?
        phase = 0
        for crystal in self.crystals:
            ray_inc_angle, ray_azim_angle = self._get_ray_angles(crystal)
            phase += crystal.calculate_phase(wavelength, ray_inc_angle, ray_azim_angle)

        return phase


    def calculate_ideal_phase_offset(self, wl, n_e=None, n_o=None):
        """
        :param wl:
        :param n_e:
        :param n_o:
        :return: phase_offset [ rad ]
        """

        phase_offset = 0
        for crystal in self.crystals:
            phase_offset += crystal.calculate_phase(wl, 0., 0., n_e=n_e, n_o=n_o)

        return phase_offset



    def calculate_contrast(self, spec):
        """

        :return:
        """

        contrast = 1
        for crystal in self.crystals:
            contrast *= crystal.contrast

        return contrast


class SimplePolCamCisInstrument(AnalyticInstrument):
    """
    Special-case coherence imaging instrument for simple polarisation camera CIS

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_igram(self, spec, savepath=None):
        """
        calculation follows p35 of S. Silburn's thesis

        :param spec:
        :type spec: xr.DataArray
        :return:
        """

        # input checks
        assert isinstance(spec, xr.DataArray)
        assert 'wavelength' in spec.dims
        assert 'x' in spec.dims
        assert 'y' in spec.dims

        i0 = spec.sum('wavelength')
        spec_norm = np.divide(spec, i0)
        phase = self.calculate_phase(spec['wavelength'])
        contrast = self.calculate_contrast(spec)
        degree_coherence = (spec_norm * contrast * np.exp(1j * phase))

        return i0 / 4 * (1 + degree_coherence.real)

    def calculate_phase(self, wavelength):
        """

        :param wavelength:
        :type wavelength: xr.DataArray

        :return: phase [ rad ]
        """

        # calculate phase delay contribution due to each crystal
        # TODO subtract phase for anti-aligned components?
        phase = 0
        for crystal in self.crystals:
            ray_inc_angle, ray_azim_angle = self._get_ray_angles(crystal)
            phase += crystal.calculate_phase(wavelength, ray_inc_angle, ray_azim_angle)

        return phase

    def calculate_contrast(self, spec):
        """

        :return:
        """

        contrast = 1
        for crystal in self.crystals:
            contrast *= crystal.contrast

        return contrast


class DoubleDelayPolCamCisInstrument(AnalyticInstrument):
    """
    Special-case coherence imaging instrument for double delay polarisation camera CIS setup

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_igram(self, spec):
        """
        calculation follows p35 of S. Silburn's thesis

        :param spec:
        :type spec: xr.DataArray
        :return:
        """

        # input checks
        assert isinstance(spec, xr.DataArray)
        assert 'wavelength' in spec.dims
        assert 'x' in spec.dims
        assert 'y' in spec.dims

        i0 = spec.sum('wavelength')
        spec_norm = np.divide(spec, i0)
        phase = self.calculate_phase(spec['wavelength'])
        contrast = self.calculate_contrast(spec)
        degree_coherence = (spec_norm * contrast * np.exp(1j * phase)).sum('wavelength')

        return i0 / 4 * (1 + degree_coherence.real)

    def calculate_phase(self, wavelength):
        """

        :param wavelength:
        :type wavelength: xr.DataArray

        :return: phase [ rad ]
        """

        # calculate phase delay contribution due to each crystal
        # TODO subtract phase for anti-aligned components?
        phase = 0
        for crystal in self.crystals:
            ray_inc_angle, ray_azim_angle = self._get_ray_angles(crystal)
            phase += crystal.calculate_phase(wavelength, ray_inc_angle, ray_azim_angle)

        return phase

    def calculate_contrast(self, spec):
        """

        :return:
        """

        contrast = 1
        for crystal in self.crystals:
            contrast *= crystal.contrast

        return contrast

