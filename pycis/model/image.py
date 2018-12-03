import pickle
import time
import sys

import os.path
import itertools
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib
from scipy.constants import c, e, k
import scipy.signal

import pycis
from pycis.model.degree_coherence import degree_coherence_analytical, degree_coherence_numerical
from pycis.model.spectra import SpectraCalib, SpectraCherab


########## This script still needs tidying up, January 2018,

# uniform mode is only working mode at the moment!


class SynthImage(object):
    """ Base class for CIS synthetic image."""

    def __init__(self, instrument, spectra, name):
        """
        :param instrument:
        :param spectra:
        :param name

        """

        self.instrument = instrument
        self.spectra = spectra
        self.name = name

        # Initialise attributes
        self.igram_intensity = None
        self.bg_intensity = None

        self.phase = None
        self.contrast = None

        self.inc_angles = None
        self.azim_angles_wp = None
        self.azim_angles_sp = None

        self.intensity_demod = None
        self.phase_demod = None
        self.contrast_demod = None

        # timing
        self.start_time = time.time()
        self.DOB = time.strftime("%a, %d %b %Y %H:%M:%S")
        self.DOB_sf = time.strftime("%y%m%d")


    def _make(self):
        """ Generate synthetic image. """
        raise NotImplementedError()

    def measure(self, clean=False):
        """ Generate synthetic image. """
        raise NotImplementedError()

    def _demod(self):
        """ Demodulate synthetic image. """

        intensity_demod, phase_demod, contrast_demod = pycis.demod.fd_image_1d(self.igram_intensity)
        phase_demod = pycis.demod.unwrap(phase_demod)

        return intensity_demod, phase_demod, contrast_demod

    def update_demod(self):
        """ Update the demodulated quantities using latest version of uncertainty. """

        self.intensity_demod, phase_demod, self.contrast_demod = pycis.demod.fd_image_1d(self.igram)
        self.phase_demod = pycis.demod.unwrap(phase_demod)
        self.save()

        return

    def _imshow(self, param, colormap, label, save, savename, ticks=True, **kwargs):
        """"""

        plt.figure(figsize=(8, 6))
        plt.imshow(param, colormap, interpolation='nearest', **kwargs)

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
            plt.savefig(os.path.join(pycis.paths.images_path, savename + '_' + self.name + '_' + self.DOB_sf + '.png'), bbox_inches='tight', pad_inches=0)
            plt.close()
        return

    def img_igram_intensity(self, save=False, vmin=0, vmax=None, clean=False):
        """ Image synthetic image. """

        igram_intensity_image, bg_intensity_image = self.measure(clean=clean)

        if vmax is None:
            vmax = 2 ** self.instrument.camera.bit_depth

        savename = 'img'
        self._imshow(scipy.signal.medfilt(np.fliplr(self.igram_intensity), kernel_size=3), 'gray', 'camera signal (ADU)', save, savename, vmin=vmin, vmax=vmax)
        return

    def img_intensity(self, fliplr=False, save=False, vmin=0, vmax=None):
        if vmax is None:
            vmax = 2 ** self.instrument.camera.bit_depth

        savename = 'signal_no_interferometer'
        self._imshow(scipy.signal.medfilt(np.fliplr(self.bg_intensity), kernel_size=3), 'gray', 'camera signal (ADU)', save, savename,vmin=vmin, vmax=vmax)
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
        self.ft = np.fft.fftshift(ft)

        plt.figure(figsize=(10, 8))
        plt.imshow(abs(self.ft), 'plasma', interpolation='nearest', norm=matplotlib.colors.LogNorm())
        plt.title('absolute ft')
        cbar = plt.colorbar()
        cbar.set_label(r'ft [/pixel]', size=9)

        if save:
            plt.savefig(pycis.paths.images_path + savename + '_' + self.line_name + '_' + self.DOB_sf + '.eps')
            plt.close()
        else:
            plt.show()
        return

    def img_theta(self, save=False):
        savename = 'theta'
        self._imshow(self.inc_angles, 'viridis', r'$\theta$ [rad]', save, savename)
        return

    def img_brightness_demod(self, save=False):
        savename = 'intensity_demod'
        self._imshow(self.intensity_demod, 'gray', 'label', save, savename)
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
        plt.plot(pix, self.intensity_demod[:, column], label='intensity_demod')
        plt.plot(pix, self.intensity_demod[:, column] - self.brightness_img[:, column], label=r'$\Delta$S')

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
        plt.plot(pix, ((self.intensity_demod / 2) * (1 + (self.contrast_demod * np.cos(self.phase_demod))))[:, column], label='uncertainty')
        plt.title('S', size=15)
        plt.xlabel('y pix', size=15)
        plt.ylabel('[Camera ADU]', size=15)
        plt.xlim(0, np.size(pix) + 1)
        plt.legend(loc=0)


        #plt.tight_layout()
        plt.show()

        _, _, _ = pycis.demod.fd_column(self.igram[:, column], display=True)

        return

    def img_demod(self):
        plt.figure(figsize=(15, 12))

        # I0
        plt.subplot(3, 3, 1)
        plt.imshow(self.S_int, 'gray', interpolation='nearest', vmin=0, vmax=4096)
        plt.colorbar()

        plt.subplot(3, 3, 2)
        plt.imshow(self.intensity_demod, 'gray', interpolation='nearest', vmin=0, vmax= (2 ** self.camera.bit_depth))
        plt.colorbar()

        plt.subplot(3, 3, 3)
        plt.imshow(self.intensity_demod - self.S_int, 'gray', interpolation='nearest')
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

        inc_angles, azim_angles_wp, azim_angles_sp = inst.get_angles()

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

        inc_angles, azim_angles_wp, azim_angles_sp = self.inst.get_angles()

        I0 = np.trapz(spectra, axis=-1)
        normalised_spectra = spectra / np.moveaxis(np.tile(I0, [len(wls), 1, 1]), 0, -1)
        normalised_spectra[np.isnan(normalised_spectra)] = 0.

        phase = pycis.model.uniaxial_crystal_3d_python(wls, t_wp, inc_angles, azim_angles_wp) + \
                pycis.model.savart_plate_3d_python(wls, t_sp, inc_angles, azim_angles_sp)

        degree_coherence = np.trapz(normalised_spectra * np.exp(2 * np.pi * 1j * phase), axis=-1)

        igram_intensity = (I0 / 2) * (1 - np.real(degree_coherence))
        brightness = (I0 / 2)
        phase = np.angle(degree_coherence)
        contrast = np.abs(degree_coherence)

        return igram_intensity, I0, phase, contrast

    def measure(self, clean=False):
        """ measure the stored interferogram intensity pattern using the modelled camera sensor. """

        # simulate the camera capture process
        print('-- Calculating camera effects...')
        igram_intensity_image = self.cam.capture(self.igram_intensity, clean=clean)
        bg_intensity_image = self.cam.capture(self.bg_intensity, clean=clean)

        return igram_intensity_image, bg_intensity_image

    def _threshold(self):
        """ return only the pixels for which the intensity meets some threshold value -- to speed up loop."""

        sum_spectra_img = np.sum(self.spectra.spectra, axis=2)

        self.threshold = np.zeros(self.instrument.camera.sensor_dim, dtype=bool)
        self.threshold[sum_spectra_img > 1.] = True

    def imshow_flow(self, save=False):
        savename = 'flow'
        self._imshow(self.doppler_flow, 'viridis', 'flow [km / s]', save, savename)

    def imshow_temperature(self, save=False):
        savename = 'flow'
        self._imshow(self.doppler_flow, 'viridis', 'flow [km / s]', save, savename)


class SynthImageCalib(SynthImage):
    """ Synthetic image generated entirely in pycis using simple spatial distributions - for creating calib images.
    
    This type of synthetic image creates simple images but provides much more information so is useful for debugging and
     testing demodulation algorithms. """

    INPUT_SPECTRA_TYPE = SpectraCalib


    def __init__(self, instrument, spectra, name, n=30):
        super().__init__(instrument, spectra, name)

        self.n = n

        # generate synthetic image:
        signal_terms, photon_terms, phase, contrast, angles, filter_transmission_factor = self._make()


        # unpack grouped terms

        self.signal = signal_terms['signal']
        self.signal_no_interferometer = signal_terms['signal_no_interferometer']

        self.photon_fluence = photon_terms['photon_fluence']
        self.photon_fluence_no_interferometer = photon_terms['photon_fluence_no_interferometer']

        self.phase = phase
        self.contrast = contrast

        self.inc_angles = angles['inc_angles']
        self.azim_angles_wp = angles['azim_angles_wp']
        self.azim_angles_sp = angles['azim_angles_sp']

        self.filter_transmission_factor = filter_transmission_factor

        # wrap-unwrap
        self.phase = pycis.demod.wrap(self.phase)
        self.phase = pycis.demod.unwrap(self.phase)

        # uncertainty
        self.intensity_demod, self.phase_demod, self.contrast_demod = self._demod()

        # timing
        duration = time.time() - self.start_time
        print('-- Duration: {0} seconds...'.format(pycis.tools.to_precision(duration, 3)))
        self.gestation_time = duration

    def _make(self):
        """ Create synthetic image."""

        # Define some shorthand:
        inst = self.instrument
        cam = inst.camera
        dim = cam.sensor_dim

        t_wp = inst.waveplate.thickness
        t_sp = inst.savartplate.thickness

        inc_angles, azim_angles_wp, azim_angles_sp = inst.get_angles()

        if inst.optical_filter is not None:
            optical_filter = self.instrument.optical_filter

        # Define output arrays:
        degree_coherence = np.zeros(dim, dtype=np.complex64)
        filter_transmission_factor = np.ones(dim)
        photon_fluence_no_interferometer = np.ones(dim) * self.spectra.spec_cube.I0

        if self.spectra.mode == 'uniform':

            m_i = self.spectra.spec_cube.m_i
            t_i = self.spectra.spec_cube.Ti

            # convert the multiplet lines to a cython-friendly format:
            lines = self.spectra.spec_cube.lines
            raw_lines = self.spectra.spec_cube.raw_lines

            line_no = len(lines)
            lines_wl = np.zeros(line_no)
            raw_lines_wl = np.zeros(line_no)
            lines_rel_int = np.zeros(line_no)

            for i in range(line_no):
                lines_wl[i] = lines['wave_obs'].iloc[i]
                raw_lines_wl[i] = raw_lines['wave_obs'].iloc[i]
                lines_rel_int[i] = lines['rel_int'].iloc[i]

            print('-- Looping over pixels...')

            for x, y in itertools.product(range(dim[1]), range(dim[0])):

                # filters spectrum
                if inst.optical_filter is not None:
                    lines_rel_int_filtered = optical_filter.apply(lines_wl, lines_rel_int, inc_angle=inc_angles[y, x])
                    filter_transmission_factor[y, x] = lines_rel_int_filtered.sum()
                    lines_rel_int_filtered /= filter_transmission_factor[y, x]
                else:
                    lines_rel_int_filtered = lines_rel_int

                v_thermal = np.sqrt((2 * k * t_i) / m_i)
                degree_coherence[y, x] = calculate_degree_coherence(lines_wl, raw_lines_wl, lines_rel_int_filtered,
                                                          inc_angles[y, x],
                                                          azim_angles_wp[y, x],
                                                          azim_angles_sp[y, x],
                                                          t_wp, t_sp, v_thermal)

        elif self.spectra.mode == 'LUT':
            # LUT mode:

            # create empty lists of parameters to be populated by LUT values:

            lut_total_intensity = []
            lut_lines_wl = []
            lut_raw_lines_wl = []
            lut_lines_rel_int = []
            lut_v_thermal = []


            # loop over unique Lineshape instances:
            for l_idx in range(0, self.spectra.unique_lineshape_no):

                # convert the multiplet lines to a cython-friendly format:
                lines = self.spectra.spec_cube[l_idx].lines
                raw_lines = self.spectra.spec_cube[l_idx].raw_lines

                line_no = len(lines)
                lines_wl = np.zeros(line_no)
                raw_lines_wl = np.zeros(line_no)
                lines_rel_int = np.zeros(line_no)
                for i in range(0, line_no):
                    lines_wl[i] = lines['wave_obs'].iloc[i]
                    raw_lines_wl = raw_lines['wave_obs'].iloc[i]
                    lines_rel_int[i] = lines['rel_int'].iloc[i]

                # populate LUTs:

                v_thermal = np.sqrt((2 * k * self.spectra.spec_cube[l_idx].Ti) / self.spectra.spec_cube[l_idx].m_i)

                lut_v_thermal.append(v_thermal)
                lut_total_intensity.append(self.spectra.spec_cube[l_idx].I0)

                lut_lines_wl.append(lines_wl)
                lut_raw_lines_wl.append(raw_lines_wl)
                lut_lines_rel_int.append(lines_rel_int)

            # Loop over detector pixels:
            for x in range(0, dim[1]):
                print(x)
                for y in range(0, dim[0]):
                    lut_id = self.spectra.img_LUT[y, x]  # LUT id for this pixel

                    # filters spectrum
                    if inst.optical_filter is not None:
                        lut_lines_rel_int_xy = optical_filter.apply(lut_lines_wl[lut_id], lut_lines_rel_int[lut_id], inc_angle=inc_angles[y, x])  # account for changes from the filters transmission to the lines multiplet structure
                        lut_lines_rel_int_xy /= lut_lines_rel_int_xy.sum()
                    else:
                        lut_lines_rel_int_xy = lut_lines_rel_int[lut_id]

                    degree_coherence[y, x] = degree_coherence(lut_total_intensity[lut_id],
                                                              lut_lines_wl[lut_id], lut_raw_lines_wl[lut_id],
                                                              lut_lines_rel_int_xy,
                                                              inc_angles[y, x],
                                                              azim_angles_wp[y, x],
                                                              azim_angles_sp[y, x],
                                                              t_wp, t_sp, lut_v_thermal,
                                                              0.6)

                    photon_fluence_no_interferometer[y, x] = lut_total_intensity[lut_id]

        elif self.spectra.mode == 'profile':  # THIS SECTION NEEDS TO BE UPDATED
            # Profile mode:
            for x in range(0, dim[1]):
                print(x)
                for y in range(0, dim[0]):
                    degree_coherence[y, x] = degree_coherence(wavelength, intensity_spectrum, theta[y, x], eta_wp[y, x],
                                                              eta_sp[y, x],
                                                              alpha, contrast, inst.L_wp, inst.L_sp)

        phase = np.angle(degree_coherence)  # (radians)
        contrast = np.abs(degree_coherence) * inst.instrument_contrast

        photon_fluence_no_interferometer = photon_fluence_no_interferometer * filter_transmission_factor
        photon_fluence = (photon_fluence_no_interferometer / 2) * (1 + contrast * np.cos(phase))

        # account for vignetting
        photon_fluence = inst.apply_vignetting(photon_fluence)
        photon_fluence_no_interferometer = inst.apply_vignetting(photon_fluence_no_interferometer)

        # model camera capture
        print('-- Calculating camera effects...')
        signal = cam.capture(photon_fluence)
        signal_no_interferometer = cam.capture(photon_fluence_no_interferometer, clean=True)

        # group output terms
        signal_terms = {'signal': signal, 'signal_no_interferometer': signal_no_interferometer}
        intensity_terms = {'photon_fluence': photon_fluence, 'photon_fluence_no_interferometer': photon_fluence_no_interferometer}
        angle_terms = {'inc_angles': inc_angles, 'azim_angles_wp': azim_angles_wp, 'azim_angles_sp': azim_angles_sp}

        return signal_terms, intensity_terms, phase, contrast, angle_terms, filter_transmission_factor

    def change_intensity_level(self, new_I0):
        """ Quickly change the intensity level of the input spectra """


        #TODO make this compatible with modes other than just 'uniform'

        # Define some shorthand:
        inst = self.instrument
        cam = inst.camera
        dim = cam.sensor_dim

        self.spectra.spec_cube.I0 = new_I0

        new_photon_fluence_no_interferometer = np.ones(dim) * self.spectra.spec_cube.I0
        new_photon_fluence_no_interferometer *= self.filter_transmission_factor

        new_photon_fluence = (new_photon_fluence_no_interferometer / 2) * (1 + self.contrast * np.cos(self.phase))

        # account for vignetting
        new_photon_fluence = inst.apply_vignetting(new_photon_fluence)
        new_photon_fluence_no_interferometer = inst.apply_vignetting(new_photon_fluence_no_interferometer)

        # model camera capture
        print('-- Calculating camera effects...')
        new_signal = cam.capture(new_photon_fluence)
        new_signal_no_interferometer = cam.capture(new_photon_fluence_no_interferometer, clean=True)

        self.photon_fluence = new_photon_fluence
        self.photon_fluence_no_interferometer = new_photon_fluence_no_interferometer

        self.signal = new_signal
        self.signal_no_interferometer = new_signal_no_interferometer

        # uncertainty
        self.intensity_demod, self.phase_demod, self.contrast_demod = self._demod()

        # save
        self.save()

        return

    def estimate_phase_demod_error(self, num_samples, num_stack=1, output_path=None, display=False):
        """ Repeatedly capture photon flux and then demodulate to build up information on the phase error distribution 
        as compared to the analytic value. """

        # set + enforce valid range for num_samples
        num_samples_lower_lim = 20
        try:
            num_samples = int(num_samples)
        except:
            raise Exception('number_samples input must be a number')
        if  num_samples < num_samples_lower_lim:
            raise Exception('number_samples input must be larger than ' + str(num_samples_lower_lim))

        # ensure '_make()' has already been run
        if hasattr(self, 'photon_fluence'):
            pass
        else:
            raise Exception('this must be run after _make() has been called.')

        # ensure output directory exists
        if output_path is not None:
            if os.path.isdir(output_path):
                pass
            else:
                raise Exception('Output path specified does not exist.')

        phase_residual = []
        roi_dim = [100, 100]

        # Initial call to print 0% progress
        # pycis.tools.printProgressBar(0, num_samples, prefix='Progress:', suffix='Complete', length=50)

        for i in range(0, num_samples):
            # pycis.tools.printProgressBar(i, num_samples, prefix='Progress:', suffix='Complete', length=50)

            if num_stack == 1:
                total_signal = self.instrument.camera.capture(self.photon_fluence)
            else:
                total_signal = self.instrument.camera.capture_stack(self.photon_fluence, num_stack=num_stack)

            # demodulate
            intensity_demod, phase_demod, contrast_demod = pycis.demod.fd_image_1d(total_signal, column_range=[400, 600])

            phase_demod = pycis.demod.unwrap(phase_demod)

            # extract ROI
            phase_demod_roi = pycis.tools.get_roi(phase_demod, roi_dim=roi_dim)
            phase_roi = pycis.tools.get_roi(self.phase, roi_dim=roi_dim)

            # calculate mean residual phase
            phase_demod_mean = np.mean(phase_demod_roi)
            phase_mean = np.mean(phase_roi)

            phase_residual.append(phase_mean - phase_demod_mean)


        phase_residual = np.array(phase_residual)

        np.save(os.path.join(output_path, 'phase_residual_array.npy'), phase_residual)

        if display:

            print(np.mean(phase_residual), np.std(phase_residual))

            plt.figure()
            plt.hist(phase_residual, bins=20)
            plt.xlabel('phase residual (rad)')
            plt.ylabel('frequency')
            plt.show()

        return

    def img_stack(self, num_stack):

        self.instrument.camera.capture_stack(self.photon_fluence, num_stack, display=True)

        # self._img(self.signal_no_interferometer, 'gray', 'Intensity [Camera ADU]', save, savename,
        #           limits=[0, (2 ** self.instrument.camera.bit_depth)])

        return

    def img_intensity(self, fliplr=False, save=False):
        """ Image DC intensity image 

        as would be detected by the camera without interferometric config (but with the
        same throughput). Comparison for the demodulated intensity:."""

        savename = 'intensity'
        self._imshow(np.fliplr(self.signal_no_interferometer), 'gray', 'Intensity [Camera ADU]', save, savename,
                     limits=[0, (2 ** self.instrument.camera.bit_depth)])
        return

    def img_phase(self, save=False, **kwargs):
        """ image output phase profile. """
        savename = 'phase'
        self._imshow(self.phase, 'viridis', r'$\phi$ [rad]', save, savename, **kwargs)
        return

    def img_contrast(self, save=False, vmin=0, vmax=1):
        """ Image total contrast. """
        savename = 'contrast'
        self._imshow(self.contrast, 'viridis', r'$\zeta$ [dimensionless]', save, savename, vmin=vmin, vmax=vmax)
        return



def create_synth_image(instrument, spectra, name):
    """ Factory function creates SynthImage instance of correct type.  """

    SYNTH_IMAGE_TYPES = [SynthImageCalib, SynthImageCherab]

    for type in SYNTH_IMAGE_TYPES:
        if type.check_spectra(spectra):
            return type(instrument, spectra, name)


def load_synth_image(name):

    image = pickle.load(open(os.path.join(pycis.paths.synth_images_path, name + '.p'), 'rb'))
    return image


if __name__ == '__main__':

    img = pycis.model.load_synth_image('demo_calib_Cd_lamp_no_filter')
    img.estimate_phase_demod_error(200, 1)





