import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import time
import pycis


def fd_image_2d(img, noise_calc=False, nfringes=None, despeckle=False, display=False):
    """ 2-D Fourier demodulation of a coherence imaging interferogram image, extracting the DC, phase and contrast 
    components.
    
    Marginally faster than fd_image_1d (when fd_image_1d is running in parallel).
    
    :param img: CIS interferogram image to be demodulated.
    :type img: array_like
    :param nfringes: Manually set the carrier (fringe) frequency to be demodulated, in units of cycles per sequence -- approximately the number of fringes present in the image. If no value is given, the fringe frequency is found automatically.
    :type nfringes: float.
    :param despeckle: Remove speckles from image.
    :type despeckle: bool.
    :param display: Display a plot.
    :type display: bool.
    
     
    :return: A tuple containing the DC component (intensity), phase and contrast.
    """

    start_time = time.time()
    pp_img = np.copy(img)

    # pre-processing: remove neutron speckles
    if despeckle:
        pp_img = pycis.demod.despeckle(pp_img)

    # since the input image is real, its FFT output is Hermitian -- all info contained in +ve frequencies: use rfft2()
    fft_img = np.fft.rfft2(pp_img, axes=(1, 0))

    # locate carrier (fringe) frequency
    if nfringes is None:
        nfringes = np.unravel_index(fft_img[15:][:].argmax(), fft_img.shape)[0] + 15  # find carrier freq. position

    # generate window function
    fft_length = int(img.shape[0] / 2)
    window_1d = pycis.demod.window(fft_length, nfringes, width_factor=0.75, fn='tukey')
    window_2d = np.transpose(np.tile(window_1d, (fft_img.shape[1], 1)))

    # isolate DC
    fft_dc = fft_img * (1 - window_2d)
    dc = np.fft.irfft2(fft_dc, axes=(1, 0))

    # isolate carrier
    fft_carrier = fft_img * window_2d
    carrier = np.fft.irfft2(fft_carrier, axes=(1, 0))

    # Hilbert transform extracts phase and contrast from carrier
    analytic_signal = scipy.signal.hilbert(carrier, axis=-2)
    phase = np.angle(analytic_signal)
    contrast = abs(analytic_signal) / dc

    # Noise calculation
    if noise_calc:
        # this part actually assumes the camera is the Photron SA-4, will need changing if using a different camera
        i0 = 2 * dc
        gain = 0.0086  # [DN / e]
        cam_noise_var = 1700  # [e ^ 2]
        sigma = np.sqrt(gain ** 2 * cam_noise_var + gain * i0)

        # calculate 'power gain' of the windows used
        y_arr_window = np.arange(0, fft_length + 1)
        x_arr_window = np.arange(0, np.shape(img)[0]) - fft_length

        x_mesh_window, y_mesh_window = np.meshgrid(x_arr_window, y_arr_window)

        pg_dc_window = abs(1 - window_2d) ** 2
        pg_carrier_window = abs(window_2d) ** 2

        area = np.trapz(np.trapz(np.ones_like(window_2d), y_mesh_window, axis=0), x_arr_window)

        carrier_noise_coeff = np.sqrt(np.trapz(np.trapz(pg_carrier_window, y_mesh_window, axis=0), x_arr_window) / area)
        dc_noise_coeff = np.sqrt(np.trapz(np.trapz(pg_dc_window, y_mesh_window, axis=0), x_arr_window) / area)
        sigma_carrier = sigma * carrier_noise_coeff
        sigma_dc = sigma * dc_noise_coeff

        sigma_contrast = abs(contrast) * np.sqrt((sigma_dc / dc) ** 2 + (sigma_carrier / (contrast * dc)) ** 2)
        sigma_phase = sigma_carrier / (contrast * dc)


    if display:
        print('-- fd_image_2d: nfringes = {}'.format(nfringes))
        print('-- fd_image_2d: time elapsed: {:.2f}s'.format(time.time() - start_time))

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
        im15 = ax15.imshow(window_2d, cmap='gray')
        cbar15 = fig1.colorbar(im15, ax=ax15)
        ax15.set_title('window 2D')

        plt.tight_layout()

        pycis.demod.display(img, dc, phase, contrast)

        if noise_calc:


            fig3 = plt.figure(figsize=(10, 6), facecolor='white')
            ax31 = fig3.add_subplot(1, 2, 1)
            im31 = ax31.imshow(sigma_phase)
            cbar31 = fig3.colorbar(im31, ax=ax31)
            ax31.set_title('phase noise')

            ax32 = fig3.add_subplot(1, 2, 2)
            im32 = ax32.imshow(np.log10(sigma_contrast), vmax=np.log10(10))
            cbar32 = fig3.colorbar(im32, ax=ax32)
            ax32.set_title('contrast noise')



        plt.show()

    if noise_calc:
        return dc, phase, contrast, dc_noise_coeff, carrier_noise_coeff
    else:
        return dc, phase, contrast





