import pickle
import os.path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib
from scipy.constants import c, e, k

import pycis
from pycis.tools import is_scalar


class SynthImage:
    """
    generate synthetic coherence imaging spectroscopy (CIS) image

    """

    def __init__(self, instrument, wl, spec, stokes=False):
        """
        :param input_spec: frequency spectrum of the light observed.
        :type input_spec: Union[pycis.InputSpec, Dict]
        
        :param instrument: instance of pycis.model.Instrument
        :type instrument: pycis.model.Instrument
        
        :param name: string

        """

        self.instrument = instrument
        self.wl, self.spec = self.format_input_spec(wl, spec, stokes)

        # calculate total number of photons incident, along each pixel's sightline
        dc_ph = np.trapz(self.spec, self.wl, axis=-3)

        # generate interference pattern, method used depends on instrument type
        if self.instrument.inst_type == 'two-beam':
            igram_ph = self._make_ideal(dc_ph)
        elif self.instrument.inst_type == 'general':
            igram_ph = self._make_mueller()
        else:
            raise Exception()

        # measure interference pattern
        self.igram = self.instrument.camera.capture(igram_ph)
        self.dc = self.instrument.camera.capture(dc_ph, clean=True)

    def format_input_spec(self, wl, spec, stokes=False):
        """
        handling input spectra for synthetic image generation

        defaults to unpolarised light (stokes=False)

        Additional functionality to be added for arbitrary Stokes' vector input (stokes=True):
        the stokes axis of the nd arrays should always be the first axis! (TODO xarray!)

        order of spec axes goes [stokes parameters, wavelengths, sensor_y, sensor_x]

        :param wl: wavelength [ m ]
        :type wl: Union[float, np.ndarray]

        :param spec:
        :type spec: Union[int, float, np.ndarray]

        :param spec_units: 'counts',  'photons'
        :type spec_units: str

        """

        if stokes is False:

            if is_scalar(wl) and is_scalar(spec):
                # monochromatic image, uniform intensity

                # we need to generate an arbitrary spectrum for compatibility with the rest of the calculations. ie
                # units of spec need to be converted from ph -> ph / m. This is a bit of a hack
                wl = np.array([wl * 0.99, wl])
                spec = np.array([0, 2 * spec / (wl[1] - wl[0])])

                # padding
                sd = self.instrument.camera.sensor_dim
                spec = np.tile(spec[np.newaxis, :, np.newaxis, np.newaxis], [4, 1, sd[0], sd[1]])
                spec[1:, :, :, :] = 0

            elif isinstance(wl, np.ndarray) and isinstance(spec, np.ndarray):
                # unique spectrum defined for each pixel

                assert wl.ndim == 1
                assert spec.ndim == 3
                assert spec.shape[0] == wl.shape[0]
                assert spec.shape[0] == self.instrument.camera.sensor_dim[0]
                assert spec.shape[1] == self.instrument.camera.sensor_dim[1]

                # padding
                spec = np.tile(spec[np.newaxis, :, :, :], [4, 1, 1, 1])

            else:
                raise TypeError
        else:
            raise NotImplementedError

        return wl, spec

    def _make_mueller(self):
        """

        :return:

        """

        mueller_matrix = self.instrument.calculate_matrix(self.wl)

        # matrix multiplication
        subscripts = 'ij...,j...->i...'
        stokes_vector_out = np.einsum(subscripts, mueller_matrix, self.spec)

        # integrate over wavelength
        igram_ph = np.trapz(stokes_vector_out[0], self.wl, axis=0)

        return igram_ph

    def _make_ideal(self, dc_ph):
        """
        calculate interference for the special case of an idealised two-beam interferometer. Analytical calculation
        avoids Mueller calculus for speed.

        :return:

        """

        # TODO can get rid of Stokes axis for the idealised calculation?

        phase = self.instrument.calculate_ideal_phase_delay(self.wl)
        spec_norm = np.where(self.spec > 0, self.spec / dc_ph[:, np.newaxis], 0)

        contrast = self.instrument.calculate_ideal_contrast()
        degree_coherence = np.trapz(spec_norm * contrast * np.exp(1j * phase), self.wl, axis=1)
        igram_ph = dc_ph / 4 * (1 + np.real(degree_coherence))

        return igram_ph[0]

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

    def img_phase(self, save=False):
        savename = 'phase'
        self._imshow(self.phase, 'viridis', r'$\phi$ [rad]', save, savename)

    def img_contrast(self, save=False, vmin=0, vmax=1):
        savename = 'contrast'
        self._imshow(self.contrast, 'viridis', r'$\zeta$', save, savename, vmin=vmin, vmax=vmax)
        return

    def img_fft(self, save=False):
        savename='fft'

        ft = np.fft.fft2(self.igram)
        ft = np.fft.fftshift(ft)

        plt.figure(figsize=(10, 8))
        plt.imshow(abs(ft), 'plasma', interpolation='nearest', norm=matplotlib.colors.LogNorm())
        plt.title('absolute ft')
        cbar = plt.colorbar()
        cbar.set_label(r'ft [/pixel]', size=9)

        if save:
            plt.savefig(pycis.paths.images_path + savename + '_' + self.line_name + '_' + self.dob_sf + '.eps')
            plt.close()
        else:
            plt.show()
        return

    def save(self):
        pickle.dump(self, open(os.path.join(pycis.paths.synth_images_path, self.name + '.p'), 'wb'))
        return


def load_synth_image(name):
    image = pickle.load(open(os.path.join(pycis.paths.synth_images_path, name + '.p'), 'rb'))
    return image



