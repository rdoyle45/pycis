import os.path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib

import pycis
from pycis.tools import is_scalar


class SynthImage:
    """
    generate synthetic coherence imaging spectroscopy (CIS) image

    """

    def __init__(self, instrument, wl, spec):
        """
        :param instrument: instance of pycis.model.Instrument
        :type instrument: pycis.model.Instrument

        :param wl: wavelengths [ m ]. See SynthImage.check_spec_mode for formatting
        :type wl: Union[float, np.ndarray]

        :param input_spec: wavelength spectrum of the light observed [ ph / pixel / m / exposure ]. See
        SynthImage.check_spec_mode for formatting
        :type input_spec: Union[float, np.ndarray]

        """

        self.instrument = instrument
        self.wl = wl
        self.spec = spec
        self.spec_mode = self.check_spec_mode(wl, spec)

        print('--pycis: inst_mode = ' + self.instrument.inst_mode)
        print('--pycis: spec_mode = ' + self.spec_mode)

        self.igram, self.i0 = self.make()

    def check_spec_mode(self, wl, spec):
        """
        handling input spectra for synthetic image generation

        :param wl: wavelength [ m ]
        :type wl: Union[float, np.ndarray]

        :param spec:
        :type spec: Union[int, float, np.ndarray]

        :return str spec_mode

        """

        if is_scalar(wl) and is_scalar(spec):
            # monochromatic image, unpolarised light, uniform intensity. Used for quick testing.

            spec_mode = 'unpolarised, monochromatic, uniform'

        else:

            assert isinstance(wl, np.ndarray) and isinstance(spec, np.ndarray)
            assert wl.ndim == 1

            if spec.ndim == 3:
                # unpolarised light, axes: [wavelength, sensor_y, sensor_x]

                assert spec.shape[0] == wl.shape[0]
                assert spec.shape[1] == self.instrument.camera.sensor_dim[0]
                assert spec.shape[2] == self.instrument.camera.sensor_dim[1]

                spec_mode = 'unpolarised'

            elif spec.ndim == 4:
                # partially polarised light, axes: [stokes vector, wavelength, sensor_y, sensor_x]

                assert spec.shape[0] == 4
                assert spec.shape[1] == wl.shape[0]
                assert spec.shape[2] == self.instrument.camera.sensor_dim[0]
                assert spec.shape[3] == self.instrument.camera.sensor_dim[1]

                spec_mode = 'partially polarised'

            else:
                raise Exception('cannot interpret spec')

        return spec_mode

    def make(self):
        """
        generate interferogram image

        calculation depends on the instrument specified and the format of wl / spec inputs. Some repeated code here,
        can be written better!

        :return: igram, i0

        """

        if self.instrument.inst_mode == 'two-beam' and self.spec_mode != 'partially polarised':
            # use analytical expression for idealised CIS (eqn. 2.5.37, p35 S. Silburn's thesis)

            if self.spec_mode == 'unpolarised, monochromatic, uniform':

                i0 = self.spec
                phase = self.instrument.calculate_ideal_phase_delay(self.wl)
                contrast = self.instrument.calculate_ideal_contrast()
                degree_coherence = contrast * np.exp(1j * phase)

                igram = i0 / 4 * (1 + np.real(degree_coherence))
                i0 = np.ones_like(igram) * i0

            elif self.spec_mode == 'unpolarised':

                i0 = np.trapz(self.spec, self.wl, axis=0)
                spec_norm = np.divide(self.spec, i0, where=i0 > 0)

                phase = self.instrument.calculate_ideal_phase_delay(self.wl)
                contrast = self.instrument.calculate_ideal_contrast()
                degree_coherence = np.trapz(spec_norm * contrast * np.exp(1j * phase), self.wl, axis=0)

                igram = i0 / 4 * (1 + np.real(degree_coherence))

            else:
                raise Exception('unable to interpret self.spec_mode')

            # simulate camera capture
            # standard-type camera
            igram = self.instrument.camera.capture(igram)
            i0 = self.instrument.camera.capture(i0, clean=True)

        elif self.instrument.inst_mode == 'general':
            # Use general Mueller calculus

            # input formatting depends on spec_mode, arrange into Stokes' vector format:
            if self.spec_mode == 'unpolarised, monochromatic, uniform':
                spec = np.array([self.spec, 0, 0, 0])

            elif self.spec_mode == 'unpolarised':
                a0 = np.zeros_like(self.spec)
                spec = np.array([self.spec, a0, a0, a0])

            elif self.spec_mode == 'partially polarised':
                spec = self.spec

            else:
                raise Exception('unable to interpret self.spec_mode')

            # Mueller matrix multiplication (mueller matrix axes are always the first two array axes)
            mueller_matrix = self.instrument.calculate_matrix(self.wl)
            subscripts = 'ij...,j...->i...'
            stokes_vector_out = np.einsum(subscripts, mueller_matrix, spec)

            # output formatting depends on spec_mode:
            if self.spec_mode == 'unpolarised, monochromatic, uniform':
                igram = stokes_vector_out
                sd = self.instrument.camera.sensor_dim
                i0 = np.tile(spec[:, np.newaxis, np.newaxis], [1, sd[0], sd[1]])

            elif self.spec_mode == 'unpolarised' or self.spec_mode == 'partially polarised':
                igram = np.trapz(stokes_vector_out, self.wl, axis=1)
                i0 = np.trapz(spec, self.wl, axis=1)

            else:
                raise Exception('unable to interpret self.spec_mode')

            # simulate camera capture
            if isinstance(self.instrument.camera, pycis.PolCamera):
                # polarisation-type camera, complete Stokes' vector required
                igram = self.instrument.camera.capture(igram)
                i0 = self.instrument.camera.capture(i0, clean=True)

            else:
                # standard-type camera, only detects S0 Stokes' parameter
                igram = self.instrument.camera.capture(igram[0])
                i0 = self.instrument.camera.capture(i0[0], clean=True)

        else:
            raise Exception('unable to interpret self.instrument.inst_mode')

        return igram, i0

    def _imshow(self, param, colormap, label, save, savename, ticks=True, **kwargs):
        """"""

        plt.figure(figsize=(8, 6))
        plt.imshow(param, colormap, origin='lower', interpolation='nearest', **kwargs)

        label_size = 15
        plt.xlabel('x pixel', size=label_size)
        plt.ylabel('y pixel', size=label_size)
        cbar = plt.colorbar()
        cbar.set_label(label, size=label_size)

        if ticks is not True:
            # Remove tick labels:
            plt.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom='off',  # ticks along the bottom edge are off
                top='off',  # ticks along the top edge are off
                labelbottom='off',  # labels along the bottom edge are off
                right='off',
                left='off',
                labelleft='off')

        if save:
            plt.savefig(os.path.join(pycis.paths.images_path, savename + '_' + self.name + '_' + self.dob_sf + '.png'), bbox_inches='tight', pad_inches=0)
            plt.close()
        return

    def img_igram(self, save=False, vmin=0, vmax=None, clean=False):
        """ Image synthetic image. """

        if vmax is None:
            vmax = 2 ** self.instrument.camera.bit_depth

        savename = 'img'
        self._imshow(self.igram, 'gray', 'camera signal (ADU)', save, savename, vmin=vmin, vmax=vmax)
        return

    def img_dc(self, save=False, vmin=0, vmax=None):
        if vmax is None:
            vmax = 2 ** self.instrument.camera.bit_depth

        savename = 'signal_no_interferometer'
        self._imshow(self.dc, 'gray', 'camera signal (ADU)', save, savename, vmin=vmin, vmax=vmax)
        return

    def img_fft(self, save=False):

        ft = np.fft.fft2(self.igram)
        ft = np.fft.fftshift(ft)

        plt.figure(figsize=(10, 8))
        plt.imshow(abs(ft), 'plasma', interpolation='nearest', norm=matplotlib.colors.LogNorm())
        plt.title('absolute ft')
        cbar = plt.colorbar()
        cbar.set_label(r'ft [/pixel]', size=9)




