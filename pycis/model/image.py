import pickle
import os.path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib
from scipy.constants import c, e, k
import scipy.signal

import pycis
from pycis.model.spectra import SpectraCherab


class SynthImage(object):
    """ Base class for CIS synthetic image."""

    def __init__(self, instrument, spectra, name):
        """
        :param spectra: 
        :param instrument: pycis.model.Instrument
        :param name: string

        """

        self.instrument = instrument
        self.spectra = spectra
        self.name = name

        # Initialise attributes
        self.igram = None
        self.igram_ph = None

        self.dc = None
        self.dc_ph = None

        self.phase = None
        self.contrast = None

        self.dc_demod = None
        self.phase_demod = None
        self.contrast_demod = None

    def _make(self):
        """ Generate synthetic image. """
        raise NotImplementedError()

    def measure(self, ph, clean=False):
        return self.instrument.camera.capture(ph, clean=clean)

    def _demod(self):
        """ Demodulate synthetic image. """

        dc_demod, phase_demod, contrast_demod = pycis.demod.fourier_demod_1d(self.igram)

        return dc_demod, phase_demod, contrast_demod

    def update_demod(self):
        """ Update the demodulated quantities using latest version of uncertainty. """

        self.dc_demod, phase_demod, self.contrast_demod = pycis.demod.fourier_demod_1d(self.igram)
        self.phase_demod = pycis.demod.unwrap(phase_demod)
        self.save()

        return

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

    def img_dc(self, fliplr=False, save=False, vmin=0, vmax=None):
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

    def img_brightness_demod(self, save=False):
        savename = 'intensity_demod'
        self._imshow(self.dc_demod, 'gray', 'label', save, savename)
        return

    def img_phase_demod(self, save=False):
        savename = 'phase_demod'
        self._imshow(self.phase_demod, 'viridis', r'$\phi_{uncertainty}$ [rad]', save, savename)
        return

    def img_contrast_demod(self, save=False, vmin=0, vmax=1):
        savename = 'contrast_demod'
        self._imshow(self.contrast_demod, 'viridis', r'$\zeta_{uncertainty}$', save, savename, vmin=vmin, vmax=vmax)
        return

    def plot_demod_column(self, column):
        """ Compare demodulated and synthetic quantities for a given image column. 
         
        This is a quick visual check that the demodulation routine used is working well.
        
        NB. This currently only returns an accurate phase comparison when the lineshape considered is a simple, single 
        gaussian. When considering a multiplet line such as CIII, multiplet phase / contrast must be accounted for TODO:
        allow for the easy accounting of these!!
        """

        pix = np.linspace(0, self.instrument.camera.sensor_dim[0] - 1, self.instrument.camera.sensor_dim[0])

        plt.figure(figsize=(14, 9))
        plt.suptitle('Demodulated Column: ' + str(column), size=20)

        plt.subplot(2, 2, 1)
        plt.plot(pix, self.brightness_img[:, column], label='intensity')
        plt.plot(pix, self.dc_demod[:, column], label='intensity_demod')
        plt.plot(pix, self.dc_demod[:, column] - self.brightness_img[:, column], label=r'$\Delta$S')

        plt.title('Intensity')
        plt.xlabel('y pix')
        plt.ylabel('[ADU]')
        plt.xlim(0, np.size(pix) + 1)
        plt.legend(loc=0)

        plt.subplot(2, 2, 2)
        plt.plot(pix, self.phase[:, column], label=r'$\phi_{synth}$')
        plt.plot(pix, self.phase_demod[:, column], label=r'$\phi_{uncertainty}$')
        plt.plot(pix, self.phase_demod[:, column] - self.phase[:, column], label=r'$\Delta\phi$')
        plt.title('Phase')
        plt.xlabel('y pix')
        plt.ylabel('[rad]')
        plt.xlim(0, np.size(pix) + 1)
        plt.legend(loc=0)


        plt.subplot(2, 2, 3)
        plt.plot(pix, self.contrast[:, column], label=r'$\zeta_{synth}$')
        plt.plot(pix, self.contrast_demod[:, column], label=r'$\zeta_{uncertainty}$')
        plt.plot(pix, self.contrast_demod[:, column] - self.contrast[:, column], label=r'$\Delta\zeta$')
        plt.title('Contrast')
        plt.xlabel('y pix')
        plt.ylabel('[dimensionless]')
        plt.xlim(0, np.size(pix) + 1)
        plt.legend(loc=0)

        # Now compare synthetic raw_data column with demodulated raw_data column
        plt.subplot(2, 2, 4)
        plt.plot(pix, ((self.brightness_img / 2) * (1 + (self.contrast * np.cos(self.phase))))[:, column], label='synthetic')
        plt.plot(pix, ((self.dc_demod / 2) * (1 + (self.contrast_demod * np.cos(self.phase_demod))))[:, column], label='uncertainty')
        plt.title('S', size=15)
        plt.xlabel('y pix', size=15)
        plt.ylabel('[Camera ADU]', size=15)
        plt.xlim(0, np.size(pix) + 1)
        plt.legend(loc=0)


        #plt.tight_layout()
        plt.show()

        _, _, _ = pycis.demod.fourier_demod_column(self.igram[:, column], display=True)

        return

    def img_demod(self):
        plt.figure(figsize=(15, 12))

        # I0
        plt.subplot(3, 3, 1)
        plt.imshow(self.S_int, 'gray', interpolation='nearest', vmin=0, vmax=4096)
        plt.colorbar()

        plt.subplot(3, 3, 2)
        plt.imshow(self.dc_demod, 'gray', interpolation='nearest', vmin=0, vmax= (2 ** self.camera.bit_depth))
        plt.colorbar()

        plt.subplot(3, 3, 3)
        plt.imshow(self.dc_demod - self.S_int, 'gray', interpolation='nearest')
        plt.colorbar()


        # phi
        plt.subplot(3, 3, 4)
        plt.imshow(self.phase_0_uw / (2 * np.pi), 'viridis', interpolation='nearest')
        plt.colorbar()


        plt.subplot(3, 3, 5)
        plt.imshow(self.phase_demod_uw / (2 * np.pi), 'viridis', interpolation='nearest')
        plt.colorbar()

        plt.subplot(3, 3, 6)
        plt.imshow((self.phase_0_uw - self.phase_demod_uw) / (2 * np.pi), 'viridis', interpolation='nearest')
        plt.colorbar()

        # zeta
        plt.subplot(3, 3, 7)
        plt.imshow(self.contrast, 'viridis', interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar()

        plt.subplot(3, 3, 8)
        plt.imshow(self.contrast_demod, 'viridis', interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar()

        plt.subplot(3, 3, 9)
        plt.imshow(self.contrast - self.contrast_demod, 'viridis', interpolation='nearest')
        plt.colorbar()

        plt.show()


        return

    def slice(self, col):
        plt.plot(self.igram[:, col])
        plt.xlabel('ypix', size=9)
        plt.ylabel('Intensity [camera ADU]', size=9)
        plt.show()
        return

    def save(self):
        pickle.dump(self, open(os.path.join(pycis.paths.synth_images_path, self.name + '.p'), 'wb'))
        return

    @classmethod
    def check_spectra(cls, spectra):
        """ Determines whether input spectra type is appropriate for the particular subclass of SynthImage.  """
        return type(spectra) == cls.INPUT_SPECTRA_TYPE


class SynthImageCherab(SynthImage):

    INPUT_SPECTRA_TYPE = SpectraCherab

    def __init__(self, inst, spectra, name):
        super().__init__(inst, spectra, name)

        self.inst = self.instrument
        self.cam = inst.camera

        # generate the line-integrated coherence properties, and the resulting interferogram intensity
        self.igram_intensity, self.bg_intensity, self.phase, self.contrast = self._make()

        self.instrument_phase = self._instrument_coherence()
        self.doppler_phase = pycis.demod.wrap(self.phase - self.instrument_phase, units='rad')

        self.doppler_flow = c * self.doppler_phase / 1400  # TODO fix this hack
        # self.intensity_demod, self.phase_demod, self.contrast_demod = self._demod()

    def _instrument_coherence(self):
        """ Calculate the instrumental phase and contrast. """

        rest_frame_wl = 465.01e-9  #TODO this really shouldn't be hard-coded.

        # Define some shorthand:
        inst = self.instrument
        cam = inst.camera
        dim = cam.sensor_dim

        t_wp = inst.waveplate.thickness
        t_sp = inst.savartplate.thickness
        wls = self.spectra.wavelength_axis
        spectra = self.spectra.spectra

        inc_angles, azim_angles_wp, azim_angles_sp = inst.calculate_ray_angles()

        I0 = np.trapz(spectra, axis=-1)
        norm_spectra = spectra / np.moveaxis(np.tile(I0, [len(wls), 1, 1]), 0, -1)

        instrument_phase = pycis.model.uniaxial_crystal_2D(rest_frame_wl, t_wp, inc_angles, azim_angles_wp) + \
                pycis.model.savart_plate_2D(rest_frame_wl, t_sp, inc_angles, azim_angles_sp)

        instrument_degree_coherence = np.exp(2 * np.pi * 1j * instrument_phase)


        instrument_phase = np.angle(instrument_degree_coherence)

        return instrument_phase

    def _make(self):
        """ Create synthetic image. """

        t_wp = self.inst.waveplate.thickness
        t_sp = self.inst.savartplate.thickness
        wls = self.spectra.wavelength_axis
        spectra = self.spectra.spectra

        inc_angles, azim_angles_wp, azim_angles_sp = self.inst.calculate_ray_angles()

        I0 = np.trapz(spectra, axis=-1)
        normalised_spectra = spectra / np.moveaxis(np.tile(I0, [len(wls), 1, 1]), 0, -1)
        normalised_spectra[np.isnan(normalised_spectra)] = 0.

        phase = pycis.model.uniaxial_crystal(wls, t_wp, inc_angles, azim_angles_wp) + \
                pycis.model.savart_plate(wls, t_sp, inc_angles, azim_angles_sp)

        degree_coherence = np.trapz(normalised_spectra * np.exp(2 * np.pi * 1j * phase), axis=-1)

        igram_intensity = (I0 / 2) * (1 - np.real(degree_coherence))
        brightness = (I0 / 2)
        phase = np.angle(degree_coherence)
        contrast = np.abs(degree_coherence)

        return igram_intensity, I0, phase, contrast

    def _threshold(self):
        """ return only the pixels for which the intensity meets some threshold value -- to speed up loop."""

        sum_spectra_img = np.sum(self.spectra.spectra, axis=2)

        self.threshold = np.zeros(self.instrument.camera.sensor_dim, dtype=bool)
        self.threshold[sum_spectra_img > 1.] = True

    def imshow_flow(self, save=False):
        savename = 'flow'
        self._imshow(self.doppler_flow, 'viridis', 'flow [km / s]', save, savename)


class SynthImagePhaseCalib(SynthImage):
    """ Synthetic image """

    INPUT_SPECTRA_TYPE = dict

    def __init__(self, instrument, spectra, name):
        super().__init__(instrument, spectra, name)

        # generate synthetic image:
        self.igram_ph, self.dc_ph = self._make()

        self.igram = self.measure(self.igram_ph)
        self.dc = self.measure(self.dc_ph, clean=True)

        # phase_offset = self.instrument.calculate_phase_offset(self.spectra['wl'])
        # self.phase -= divmod(phase_offset, 2 * np.pi)[0] * 2 * np.pi

        # uncertainty
        # self.intensity_demod, self.phase_demod, self.contrast_demod = self._demod()

    def _make(self):
        """ Create synthetic image."""

        wl = self.spectra['wl']
        spec = self.spectra['spec']
        units = self.spectra['spec units']
        phase = self.instrument.calculate_phase_delay(wl)

        if pycis.tools.is_scalar(wl):

            if pycis.tools.is_scalar(spec):

                if units == 'cnts':
                    dc_ph = np.ones_like(phase) * spec * \
                            self.instrument.camera.epercount / self.instrument.camera.qe

                elif units == 'ph':
                    dc_ph = np.ones_like(phase) * spec / 2

            else:

                assert np.shape(spec) == np.shape(phase)

                if units == 'cnts':
                    dc_ph = spec * self.instrument.camera.epercount / self.instrument.camera.qe

                elif units == 'ph':
                    dc_ph = spec / 2

            # single wavelength: the only contrast degradation is instrument contrast
            contrast = np.ones_like(phase) * self.instrument.instrument_contrast

        else:

            # calculate contrast degradation due to spectrum

            assert np.shape(spec) == np.shape(phase)
            assert units == 'ph'

            dc_ph = np.trapz(spec, axis=-1) / 2
            normalised_spectra = spec / np.moveaxis(np.tile(2 * dc_ph, [len(wl), 1, 1]), 0, -1)
            normalised_spectra[np.isnan(normalised_spectra)] = 0.

        a0 = np.zeros_like(dc_ph)
        stokes_vector_in = np.array([dc_ph, a0, a0, a0])

        subscripts = 'ij...,j...->i...'
        instrument_mueller_mat = self.instrument.calculate_transfer_matrix(wl)
        stokes_vector_out = np.einsum(subscripts, instrument_mueller_mat, stokes_vector_in)
        igram_ph = stokes_vector_out[0]

        return igram_ph, dc_ph


def create_synth_image(instrument, spectra, name):
    """ Factory function creates SynthImage instance of correct type.  """

    SYNTH_IMAGE_TYPES = [SynthImageCalib, SynthImageCherab, SynthImagePhaseCalib]

    for type in SYNTH_IMAGE_TYPES:
        if type.check_spectra(spectra):
            return type(instrument, spectra, name)


def load_synth_image(name):

    image = pickle.load(open(os.path.join(pycis.paths.synth_images_path, name + '.p'), 'rb'))
    return image



