from matplotlib import pyplot as plt
import matplotlib.gridspec
import numpy as np
import pycis
import scipy.signal
import scipy.ndimage


def fourier_demod_column(col, nfringes=None, apodise=False, display=False):
    """ 1-D Fourier demodulation of single CIS interferogram raw_data column, extracting the DC component (intensity), phase and contrast.
    
    :param col: CIS interferogram column to be demodulated.
    :type col: array_like, 1-D.
    :param nfringes: Manually set the carrier (fringe) frequency to be demodulated, in units of cycles per sequence -- approximately the number of fringes present in the image. If no value is given, the fringe frequency is found automatically.
    :type nfringes: int.
    :param apodise: Apodise the extracted carrier at points of steep intensity gradient to minimise influence of artefacts and noise (See SS thesis.)
    :type apodise: bool.
    :param display: Display a plot.
    :type display: bool. 
    
    :return: A tuple containing the DC component (intensity), phase and contrast.
    """

    col = col.astype(np.float64)
    col_length = np.size(col)

    pixels = np.linspace(1, col_length, col_length)

    # locate carrier (fringe) frequency

    fft_col = np.fft.rfft(col)

    if nfringes is None:
        nfringes_min, nfringes_max = (40, 160) # Range of carrier frequencies within which to search
        nfringes = pycis.tools.indexes(abs(fft_col[nfringes_min:nfringes_max]), thres=0.7, min_dist=50)

        if np.size(nfringes) != 1:
            dc = 2 * col
            phase = 0 * col
            contrast = 0 * col

            if display:
                print('no carrier frequency found.')

            return dc, phase, contrast

        else:
            nfringes = nfringes.squeeze() + nfringes_min  # remove single-dimensional entries from the shape of array

    # generate window function
    fft_length = int(col_length / 2)
    window = pycis.demod.window(fft_length, nfringes)

    # isolate DC
    fft_dc = np.multiply(fft_col, 1 - window)
    dc = np.fft.irfft(fft_dc)
    dc_smooth = scipy.ndimage.gaussian_filter(dc, 10)

    fft_carrier = fft_col - fft_dc
    carrier = np.fft.irfft(fft_carrier)

    if apodise:
        # locate sharp edges:

        carrier_apodised = np.copy(carrier)

        grad = np.ones_like(dc, dtype=np.float32)
        grad[dc_smooth >= 0] = abs(np.gradient(dc_smooth[dc_smooth >= 0])) / dc_smooth[dc_smooth >= 0]

        max_grad, window_width = (0.05, 26)

        thres_normalised = (max_grad - min(grad)) / (max(grad) - min(grad))
        locs = pycis.tools.indexes(grad, thres=thres_normalised)
        window_apod = 1 - np.hanning(window_width*2)

        if np.size(locs) != 0:
            for i in range(0,np.size(locs)):
                if locs[i] > window_width and locs[i] < np.size(carrier) - window_width:
                    carrier_apodised[locs[i] - window_width: locs[i] + window_width] = carrier_apodised[locs[i] - window_width: locs[i] + window_width]*window_apod
                elif locs[i] < window_width:
                    carrier_apodised[locs[i]:locs[i] + window_width] = carrier_apodised[locs[i]:locs[i] + window_width] * window_apod[window_width : (2*window_width) + 1]

        analytic_signal_apodised = scipy.signal.hilbert(carrier_apodised)
        analytic_signal = scipy.signal.hilbert(carrier)
        phase = np.angle(analytic_signal_apodised)
        contrast = np.divide(abs(analytic_signal), dc_smooth)

    else:

        analytic_signal_1 = scipy.signal.hilbert(carrier)


        # plt.figure()
        # plt.plot(pixels, np.real(analytic_signal_1))
        # plt.plot(pixels, np.imag(analytic_signal_1))
        #
        # plt.show()


        phase = np.angle(analytic_signal_1)
        contrast = np.divide(abs(analytic_signal_1), dc)

    # contrast[contrast > 1.] = 1.
    # contrast[contrast < 0.] = 0.

    # Calculate upper and lower bounds for contrast envelope:
    contrast_envelope_lower = dc * (1 + contrast)
    contrast_envelope_upper = dc * (1 - contrast)

    # Now calculate interferogram using extracted quantities for comparison:
    S = dc * (1 + (contrast * np.cos(phase)))

    # Optional plot output:
    if display:

        fig1 = plt.figure(figsize=(9, 6))

        gs = matplotlib.gridspec.GridSpec(2, 2)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])
        ax4 = plt.subplot(gs[3])

        ax1.plot_raw(abs(fft_col))
        ax1.semilogy()

        ax2.plot_raw(abs(fft_col))
        ax2.plot_raw(abs(fft_dc))
        ax2.plot_raw((1 - window) * np.max(abs(fft_col)), lw=1)
        ax2.semilogy()

        ax3.plot_raw(abs(fft_col))
        ax3.plot_raw(abs(fft_carrier), lw=1)
        ax3.plot_raw(window * np.max(abs(fft_col)), lw=1)
        ax3.semilogy()

        fig2 = plt.figure()
        plt.subplot(2, 2, 1)
        plt.plot(dc, 'k', label='Extracted intensity')
        plt.plot(col, 'b', label='raw')
        plt.plot(contrast_envelope_lower, 'r', label='Contrast envelope')
        plt.plot(contrast_envelope_upper, 'r')
        plt.title(r'$I_0$')

        plt.legend(loc=0)
        plt.xlabel('y pix', size=9)
        plt.ylabel('[ADU]', size=9)
        plt.xlim(0, np.size(pixels) + 1)

        plt.subplot(2, 2, 2)

        plt.plot(dc, label='zeta')
        plt.plot(dc_smooth, label='zeta')
        plt.title('dc', size=15)
        plt.xlabel('y pix', size=9)
        plt.ylabel('[dimensionless]', size=9)
        plt.xlim(0, np.size(pixels) + 1)



        plt.subplot(2, 2, 3)
        plt.plot(contrast, label='zeta')
        plt.title(r'$\zeta$', size=15)
        plt.xlabel('y pix', size=9)
        plt.ylabel('[dimensionless]', size=9)
        plt.xlim(0, np.size(pixels) + 1)
        plt.ylim(0, 1)

        plt.subplot(2, 2, 4)
        plt.plot(phase)
        plt.title(r'$\phi$', size=15)
        plt.xlabel('y pix', size=9)
        plt.ylabel('[rad]', size=9)
        plt.xlim(0, np.size(pixels) + 1)

        plt.tight_layout()
        plt.show()

    return dc, phase, contrast








