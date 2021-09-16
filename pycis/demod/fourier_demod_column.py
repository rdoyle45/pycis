from matplotlib import pyplot as plt
import matplotlib.gridspec
import numpy as np
import pycis
import scipy.signal
import scipy.ndimage


def fourier_demod_column(max_grad, window_width, Ilim, wtype, wfactor, filtval, col, nfringes=None, apodise=False, display=False):
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

    nfringes = col[1]
    col = col[0]
    col = col.astype(np.float64)

    col_length = np.size(col)
    pixels = np.linspace(1, col_length, col_length)

    # locate carrier (fringe) frequency
    #col_filt = scipy.signal.medfilt(col, filtval)
    #fft_col = np.fft.rfft(col_filt)

  #  if nfringes is None:
        #nfringes_min, nfringes_max = (40, 160) # Range of carrier frequencies within which to search
        #nfringes = pycis.tools.indexes(abs(fft_col[nfringes_min:nfringes_max]), thres=0.7, min_dist=40)
   #     nfringes = abs(fft_col[100:]).argmax() + 100
        #if np.size(nfringes) != 1:
         #   dc = 2 * col
          #  phase = 0 * col
           # contrast = 0 * col

            #S_apodised = dc * (1 + (contrast * np.cos(phase)))
            #S = S_apodised

           # if display:
            #    print('no carrier frequency found.')

           # return dc, phase, contrast, col

        #else:
         #   nfringes = nfringes.squeeze() + nfringes_min  # remove single-dimensional entries from the shape of array

    # generate window function
#    fft_length = fft_col.size
 #   window = pycis.demod.window(fft_length, nfringes, width_factor=wfactor, fn=wtype)

    # isolate DC
  #  fft_dc = np.multiply(fft_col, 1 - window)
   # dc = 2*np.fft.irfft(fft_dc)
    #dc_smooth = scipy.ndimage.gaussian_filter(dc, fft_length/nfringes)

    ###### TEST SCOTTS CODE #######
    type(col_length)
    type(nfringes)
    w = round(col_length/nfringes)
    bandwidth = 0.8
    N = round(bandwidth*nfringes)

    wdw = np.ones(col_length,1)

    lp = nfringes
    up = col_length - nfringes

    wdw[lp-(N-1)/2:lp + (N-1)/2] = 1 - scipy.signal.hanning(N)
    wdw[up - (N-1)/2:up + (N-1)/2] = 1 - np.flipud(scipy.signal.hanning(N))

    fft_col = np.fft.fft(col)
    fft_dc = fft_col*wdw

    dc = 2*np.fft.ifft(fft_dc)
    dc = scipy.signal.medfilt(dc, w)

    col_in = np.copy(col)

    col_in[dc > Ilim] = 2*col_in[dc > Ilim]/dc[dc > Ilim] - 1
    col_in[dc <= Ilim] = 0
    #S_apodised = 2*col/dc - 1

    if apodise:
        # locate sharp edges:
        grad = np.ones_like(dc, dtype=np.float32)
        grad = abs(np.gradient(dc_smooth)) / dc_smooth

        window_width = int(window_width)

        thres_normalised = (max_grad - min(grad)) / (max(grad) - min(grad))
        locs = pycis.tools.indexes(grad, thres=thres_normalised, min_dist=window_width)
        
        window_apod = 1 - np.hanning(window_width*2)

        locs = locs[locs >= 20]

        if np.size(locs) != 0:
            for i in range(0,np.size(locs)):
                if window_width < locs[i] < np.size(col) - window_width:
                  #  S_apodised[locs[i] - window_width: locs[i] + window_width] = S_apodised[locs[i] - window_width: locs[i] + window_width]*window_apod
                    col_in[locs[i] - window_width: locs[i] + window_width] = col_in[locs[i] - window_width: locs[i] + window_width]*window_apod
                elif locs[i] < window_width:
                   # S_apodised[locs[i]:locs[i] + window_width] = S_apodised[locs[i]:locs[i] + window_width] * window_apod[window_width : (2*window_width) + 1]
                    col_in[locs[i]:locs[i] + window_width] = col_in[locs[i]:locs[i] + window_width] * window_apod[window_width : (2*window_width) + 1]

        col_in *= scipy.signal.windows.tukey(col_in.shape[0], alpha=0.1)
   #     S_apodised = grad
    fft_carrier = np.fft.rfft(col_in)
    fft_carrier = np.multiply(fft_carrier,2*window)
    carrier = np.fft.irfft(fft_carrier, n=col_length)

    analytic_signal = scipy.signal.hilbert(carrier)
    phase = np.angle(analytic_signal)
    contrast = np.divide(abs(analytic_signal), dc_smooth)

    # contrast[contrast > 1.] = 1.
    # contrast[contrast < 0.] = 0.

    # Calculate upper and lower bounds for contrast envelope:
    #contrast_envelope_lower = dc * (1 + contrast)
    #contrast_envelope_upper = dc * (1 - contrast)

    # Now calculate interferogram using extracted quantities for comparison:
    #S_apodised = np.asarray(dc * (1 + contrast*np.cos(phase)))
    #S = np.asarray(dc * (1 + (contrast * np.cos(phase_not_apodised))))

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
     
    return dc, phase, contrast#, S_apodised
