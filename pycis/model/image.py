import pickle
import os.path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib
from scipy.constants import c, e, k
import pycis


class SynthImage(object):
    """ synthetic coherence imaging spectroscopy (CIS) image."""

    def __init__(self, instrument, spec):
        """
        :param spec: frequency spectrum of the light observed.  
        
        :param instrument: instance of pycis.model.Instrument
        :type instrument: pycis.model.Instrument
        
        :param name: string
        """

        self.instrument = instrument
        self.spec = spec

        # TODO the vectorisation needs a clean-up and to be thoroughly checked
        # TODO radiometric units
        # TODO Stokes vector compatibility

        # generate interference pattern
        if self.instrument.inst_type == 'two-beam':
            igram_ph, dc_ph = self._make_ideal()

        else:
            # general treatment
            igram_ph, dc_ph = self._make_mueller()

        # measure interference pattern
        self.igram = self.instrument.camera.capture(igram_ph)
        self.dc = self.instrument.camera.capture(dc_ph, clean=True)

    def _make_mueller(self):
        """

        :return: 
        """

        wl = self.spec['wl']
        spec = self.spec['spec']
        units = self.spec['spec units']

        if pycis.tools.is_scalar(wl):

            if pycis.tools.is_scalar(spec):

                if units == 'cnts':
                    dc_ph = np.ones(self.instrument.camera.sensor_dim) * spec * \
                            self.instrument.camera.epercount / self.instrument.camera.qe

                elif units == 'ph':
                    dc_ph = np.ones(self.instrument.camera.sensor_dim) * spec / 2

            else:

                assert np.shape(spec) == self.instrument.camera.sensor_dim

                if units == 'cnts':
                    dc_ph = spec * self.instrument.camera.epercount / self.instrument.camera.qe

                elif units == 'ph':
                    dc_ph = spec / 2

        else:

            # calculate contrast degradation due to spectrum
            assert np.shape(spec) == self.instrument.camera.sensor_dim
            assert units == 'ph'

            dc_ph = np.trapz(spec, axis=-1) / 2
            normalised_spectra = spec / np.moveaxis(np.tile(2 * dc_ph, [len(wl), 1, 1]), 0, -1)
            normalised_spectra[np.isnan(normalised_spectra)] = 0.

        a0 = np.zeros_like(dc_ph)
        stokes_vector_in = np.array([dc_ph, a0, a0, a0])

        subscripts = 'ij...,j...->i...'
        transfer_matrix = self.instrument.calculate_transfer_matrix(wl)
        stokes_vector_out = np.einsum(subscripts, transfer_matrix, stokes_vector_in)
        igram_ph = stokes_vector_out[0]

        return igram_ph, dc_ph

    def _make_ideal(self):
        """ Needs looking over """

        wl = self.spec['wl']
        spec = self.spec['spec']
        units = self.spec['spec units']
        phase = self.instrument.calculate_ideal_phase_delay(wl)

        if pycis.tools.is_scalar(wl):

            if pycis.tools.is_scalar(spec):

                if units == 'cnts':
                    dc_ph = np.ones(self.instrument.camera.sensor_dim) * spec * \
                            self.instrument.camera.epercount / self.instrument.camera.qe / self.instrument.calculate_ideal_transmission()

                elif units == 'ph':
                    dc_ph = np.ones(self.instrument.camera.sensor_dim) * spec / 2

            else:

                assert np.shape(spec) == self.instrument.camera.sensor_dim

                if units == 'cnts':
                    dc_ph = spec * self.instrument.camera.epercount / self.instrument.camera.qe / self.instrument.calculate_ideal_transmission()

                elif units == 'ph':
                    dc_ph = spec / 2

            degree_coherence = self.instrument.calculate_ideal_contrast() * np.exp(1j * phase)

        else:
            # calculate contrast degradation due to spectrum
            assert np.shape(spec) == self.instrument.camera.sensor_dim
            assert units == 'ph'

            dc_ph = np.trapz(spec, axis=-1) / 2
            normalised_spectra = spec / np.moveaxis(np.tile(2 * dc_ph, [len(wl), 1, 1]), 0, -1)
            normalised_spectra[np.isnan(normalised_spectra)] = 0.

            contrast = self.instrument.calculate_ideal_contrast()

            degree_coherence = np.trapz(normalised_spectra * contrast * np.exp(1j * phase), axis=-1)

        igram_ph = dc_ph / 4 * (1 + np.real(degree_coherence))

        # phase = np.angle(degree_coherence)
        # contrast = np.abs(degree_coherence)

        return igram_ph, dc_ph

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



