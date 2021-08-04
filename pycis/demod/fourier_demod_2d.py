import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import time, copy
import pycis
import scipy
from skimage.filters import window

def fourier_demod_2d(img, despeckle=False, mask=True, uncertainty_out=False, camera=None, nfringes=None,
                     notch_take=None, notch_add=None, display=False, width_factor=1., alpha=0.5, bumps=None):
    """ 
    2D Fourier demodulation of a coherence imaging interferogram image, extracting the DC, phase and contrast.
    
    Option to output uncertainty info too.
    
    :param img: CIS interferogram image to be demodulated.
    :type img: array_like
    
    :param despeckle: Remove speckles from image.
    :type despeckle: bool.
    
    :param mask: End region masking to reduce Fourier artefacts
    :type mask: bool
    
    :param uncertainty_out: output information on the uncertainty in the demodulated quantities
    :type uncertainty_out: bool

    :param camera: optional, instance of pycis.model.Camera, used to calculate uncertainty in demodulated quantities
    :type camera: pycis.model.Camera

    :param nfringes: Manually set the carrier (fringe) frequency to be demodulated, in units of cycles per sequence --
    approximately the number of fringes present in the image. If no value is given, the fringe frequency is found
    automatically.
    :type nfringes: float.

    :param display: Display a plot.
    :type display: bool.
    
    :return: A tuple containing the DC component (intensity), phase and contrast.
    """

    # TODO cleanup
    start_time = time.time()
    pp_img = np.copy(img)
    #pp_img = scipy.ndimage.rotate(img, -22.5) 
    # pre-processing (pp): remove neutron speckles
    if despeckle:
        pp_img = pycis.demod.despeckle(pp_img)

    # since the input image is real, its FT is Hermitian -- all info contained in +ve frequencies -- use rfft2()
    fft_img = np.fft.rfft2(pp_img, axes=(1, 0))

    #plt.imshow(abs(fft_img))
    #plt.show()

    # estimate carrier (fringe) frequency, if not supplied
    if nfringes is None:
        nfringes = np.unravel_index(fft_img[15:][:].argmax(), fft_img.shape)[0] + 15

    # generate window function
    fft_length = fft_img.shape[0]
    window_1d = pycis.demod.window(fft_length, nfringes, width_factor=width_factor, fn='tukey', alpha=alpha)
    window_carrier = np.transpose(np.tile(window_1d, (fft_img.shape[1], 1)))
    #window_1d = pycis.demod.window(1338, nfringes, width_factor=width_factor, fn='tukey', alpha=alpha)
    #window_carrier = np.transpose(np.tile(window_1d, (1338, 1)))
    #window_carrier = scipy.ndimage.rotate(window_carrier, 22.5)
    #window_carrier = pycis.tools.get_roi(window_carrier, roi_dim=(1024,1024))    

    window_dc = 1 - copy.deepcopy(window_carrier)

#    window_carrier = window('blackmanharris', fft_img.shape)
 #   window_dc = 1 - copy.deepcopy(window_carrier)

    # if bumps is not None:
    #     wx = np.arange(window_carrier.shape[1])
    #     wy = np.arange(window_carrier.shape[0])
    #     wxx, wyy = np.meshgrid(wx, wy)
    #     for bump in bumps:
    #         b = 1 - np.exp(-1/2 * (((wxx - bump[1]) / 18) ** 2 + ((wyy - bump[0]) / 10) ** 2))
    #         window_carrier *= b
    #         window_dc *= b
    #
    # if notch_take is not None:
    #     # cut a vertical notch out in Fourier domain to remove artefacts
    #
    #     notch_window_width = 9
    #     pre_zeros = [0] * int(notch_take - notch_window_width / 2)
    #     mid_zeros = [0] * (img.shape[1] - 2 * notch_window_width - 2 * len(pre_zeros))
    #     notch_take = scipy.signal.tukey(notch_window_width, alpha=0.8)
    #
    #     notch_window = np.concatenate([pre_zeros, notch_take, mid_zeros, notch_take, pre_zeros])
    #
    #     window_carrier *= (1 - notch_window)

    # if notch_add is not None:
    #     # cut a vertical notch out in Fourier domain to remove artefacts
    #
    #     notch_window_width = 6
    #     pre_zeros = [0] * int(notch_add - notch_window_width / 2)
    #     mid_zeros = [0] * (img.shape[1] - 2 * notch_window_width - 2 * len(pre_zeros))
    #     notch_add = scipy.signal.tukey(notch_window_width, alpha=0.8)
    #
    #     notch_window = np.concatenate([pre_zeros, notch_add, mid_zeros, notch_add, pre_zeros])
    #
    #     notch_window = np.tile(notch_window, reps=[window_carrier.shape[0], 1])
    #
    #     window_carrier += notch_window
    #     window_carrier[window_carrier > 1] = 1
    #
    #     window_1d_na = pycis.demod.window(fft_length, nfringes, width_factor=2.3, fn='tukey', alpha=0.3)
    #     window_2d_na = np.transpose(np.tile(window_1d_na, (fft_img.shape[1], 1)))
    #     window_carrier *= window_2d_na

    if mask:
        # end region masking
        pp_img_erm_dc = pycis.demod.end_region_mask(pp_img, alpha=0.15, mean_subtract=True)
        pp_img_erm_phase = pycis.demod.end_region_mask(pp_img, alpha=(3 / nfringes), mean_subtract=True)

        fft_img_erm_dc = np.fft.rfft2(pp_img_erm_dc, axes=(1, 0))
        fft_img_erm_phase = np.fft.rfft2(pp_img_erm_phase, axes=(1, 0))

        # isolate DC
        fft_dc = fft_img_erm_dc * (1 - window_carrier)
        dc = np.fft.irfft2(fft_dc, axes=(1, 0))

        # isolate carrier
        fft_carrier_phase = fft_img_erm_phase * window_carrier
        fft_carrier_contrast = fft_img * window_carrier

        fft_carrier = fft_carrier_phase  # for the plotting, change

        carrier_phase = np.fft.irfft2(fft_carrier_phase, axes=(1, 0))
        carrier_contrast = np.fft.irfft2(fft_carrier_contrast, axes=(1, 0))

        # Hilbert transform to extract phase and contrast from carrier
        analytic_signal_phase = scipy.signal.hilbert(carrier_phase, axis=-2)
        analytic_signal_contrast = scipy.signal.hilbert(carrier_contrast, axis=-2)

        phase = np.angle(analytic_signal_phase)
        contrast = np.abs(analytic_signal_contrast) / dc

    else:
        # isolate DC
        fft_dc = fft_img * window_dc
        dc = np.fft.irfft2(fft_dc, axes=(1, 0))

        # isolate carrier
        fft_carrier = fft_img * window_carrier
        carrier = np.fft.irfft2(fft_carrier, axes=(1, 0))

        # Hilbert transform to extract phase and contrast from carrier
        analytic_signal = scipy.signal.hilbert(carrier, axis=-2)
        phase = np.angle(analytic_signal)
        contrast = np.abs(analytic_signal) / dc

        # Rotate Image back
        #phase = scipy.ndimage.rotate(phase, 22.5)
        #phase = pycis.tools.get_roi(phase, roi_dim=[1024,1024])
        
        #dc = scipy.ndimage.rotate(dc, 22.5)
        #dc = pycis.tools.get_roi(dc, roi_dim=[1024,1024])

        #contrast = scipy.ndimage.rotate(contrast, 22.5)
        #contrast = pycis.tools.get_roi(contrast, roi_dim=[1024,1024])

    # uncertainty calculation
    if uncertainty_out:

        if camera is None:
            # estimate the image noise using Fourier domain image

            padding_x = 200
            padding_y = 100

            # generate window for extraction of the 'empty' part of the image
            window_empty = np.zeros_like(fft_img, dtype=np.float64)
            window_empty[padding_y:, padding_x:-padding_x] = 1

            fft_img_empty = fft_img[np.where(window_empty == 1)]

            imag = np.imag(fft_img_empty).flatten()
            real = np.real(fft_img_empty).flatten()

            var_imag = np.var(imag)
            var_real = np.var(real)
            var_avg = (var_real + var_imag) / 2

            # predict image sigma
            var_img = (2 * var_avg) / (img.shape[0] * img.shape[1])
            std = np.ones_like(img) * np.sqrt(var_img)

        else:
            # estimate standard deviation of the noise
            std = (1 / camera.epercount) * np.sqrt(camera.cam_noise ** 2 + camera.epercount * img) # checked and correct

        area = window_carrier.shape[0] * window_carrier.shape[1]
        carrier_noise_coeff = np.sqrt(np.power(window_carrier, 2).sum() / area)
        dc_noise_coeff = np.sqrt(np.power(window_dc, 2).sum() / area)
        std_carrier = std * carrier_noise_coeff

        std_dc = std * dc_noise_coeff
        std_contrast = abs(contrast) * np.sqrt((std_dc / dc) ** 2 + (std_carrier / (contrast * dc)) ** 2)
        std_phase = std_carrier / (contrast * dc)

        # estimate covariance matrix for phase
        cov_phase_pixel_vert = np.fft.irfft2(abs(window_carrier) ** 2, axes=(1, 0))[:, 0]
        cov_phase = np.zeros([img.shape[0], img.shape[0]])

        # assemble into covariance matrix -- ugly
        for idx in range(len(cov_phase_pixel_vert)):
            if idx != 0:
                cov_phase_pixel_vert = np.roll(cov_phase_pixel_vert, 1)

            cov_phase[:, idx] = cov_phase_pixel_vert

        corr_phase = cov_phase / np.max(cov_phase)  # Pearson correlation coefficient matrix for image column

        uncertainty = {'std_dc': std_dc,
                       'std_phase': std_phase,
                       'std_contrast': std_contrast,
                       'snr': dc / std,
                       'corr_phase': corr_phase,
                       }  # only really appropriate for calibration images

    if display:
        print('-- fourier_demod_2d: nfringes = {}'.format(nfringes))
        print('-- fourier_demod_2d: time elapsed: {:.2f}s'.format(time.time() - start_time))

        fig1 = plt.figure(figsize=(10, 6), facecolor='white')

        ax11 = fig1.add_subplot(2, 3, 1)
        im11 = ax11.imshow(np.log10(abs(fft_img) ** 2))
        cbar11 = fig1.colorbar(im11, ax=ax11)
        ax11.set_title('FFT img')
        
        ax12 = fig1.add_subplot(2, 3, 2)
        im12 = ax12.imshow(np.log10(abs(fft_carrier) ** 2))
        cbar12 = plt.colorbar(im12, ax=ax12)
        ax12.set_title('FFT carrier')
        
        ax13 = fig1.add_subplot(2, 3, 3)
        im13 = ax13.imshow(np.log10(abs(fft_dc) ** 2))
        cbar13 = plt.colorbar(im13, ax=ax13)
        ax13.set_title('FFT DC')

        ax14 = fig1.add_subplot(2, 3, 4)
        ax14.semilogy(window_1d)
        plt.title('window 1D')

        ax15 = fig1.add_subplot(2, 3, 5)
        im15 = ax15.imshow(window_carrier, cmap='gray')
        cbar15 = fig1.colorbar(im15, ax=ax15)
        ax15.set_title('window 2D')

        plt.tight_layout()

        pycis.demod.display(img, dc, phase, contrast)

        if uncertainty_out:

            fig3 = plt.figure(figsize=(10, 6), facecolor='white')
            ax31 = fig3.add_subplot(1, 2, 1)
            im31 = ax31.imshow(std_phase)
            cbar31 = fig3.colorbar(im31, ax=ax31)
            ax31.set_title('phase noise')

            ax32 = fig3.add_subplot(1, 2, 2)
            im32 = ax32.imshow(np.log10(std_contrast), vmax=np.log10(10))
            cbar32 = fig3.colorbar(im32, ax=ax32)
            ax32.set_title('contrast noise')

        plt.show()

    if uncertainty_out:
        return dc, phase, contrast, uncertainty
    else:
        return dc, phase, contrast




