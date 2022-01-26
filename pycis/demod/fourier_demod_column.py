from matplotlib import pyplot as plt
import matplotlib.gridspec
import numpy as np
import pycis
import scipy.signal
import scipy.ndimage
import scipy.fft


def fourier_demod_column(max_grad, window_width, ilim, wtype, wfactor, filtval, col, apodise=False, display=False):
    """ 1-D Fourier demodulation of single CIS interferogram raw_data column, extracting the DC component (intensity), phase and contrast.

        Parameters:

            col   (np.array)       : Array containing Raw CIS Data Column
            max_grad  (float)      : Maximum intensity gradient considered a 'sharp edge' for filtering
            window_width (int)     : Width of appodisation window in pixels
            ilim  (int)            : Minimum Intensity value considered in demod - anything below this is set to 0
            wtype (str)            : Window function type for phase demodulation - 'hanning', 'blackmanharris' or 'tukey'
            wfactor (float)        : A multiplicative factor determining the width of the filters, multiplies nfringes.
            filtval  (int)         : Size (in pixels) of convolved filter applied pre-demod
            apodise  (bool)        : Turn apodisation on
    
        Returns:
            A tuple containing the DC component (intensity), phase and contrast.
    """

    col = col[0]
    col = col.astype(np.float64)

    col_length = np.size(col)
    pixels = np.linspace(1, col_length, col_length)

    win = scipy.signal.windows.hann(filtval)
    col_filt = scipy.signal.convolve(col, win, mode='same')/sum(win)
    fft_col = np.fft.rfft(col)

    # TEST NEW FIND PEAKS
    peaks, peakheights = pycis.tools.PeakDetect(range(len(fft_col)), abs(fft_col), w=31, thres=0.05)

    if peaks.size != 0 and 130>=max(peaks)>=100:
        index = peakheights[peaks >= 100].argmax()
        nfringes = peaks[peaks >= 100][index]
    else:
        nfringes = 113

    w = int(round(col_length/nfringes))
    if w % 2 == 0:
        w += 1
    bandwidth = wfactor
    N = int(round(bandwidth*nfringes))
    if N % 2 == 0:
        N = N+1

    halfwidth = int((N-1)/2)

    wdw = np.ones((col_length))

    lp = nfringes+1
    up = col_length - nfringes+1

    fns = {'hanning': scipy.signal.hanning,
           'blackmanharris': scipy.signal.windows.blackmanharris,
           'tukey': scipy.signal.windows.tukey}

    fn = fns['hanning']
    #fn= fns[wtype]

    wdw[lp-int(halfwidth):lp + int(halfwidth+1)] = 1 - fn(N)
    wdw[up - int(halfwidth):up + int(halfwidth+1)] = 1 - np.flipud(fn(N))

    fft_col = scipy.fft.fft(col_filt)
    fft_dc = fft_col*wdw.T

    dc = 2*scipy.fft.ifft(fft_dc)
    dc = scipy.ndimage.filters.median_filter(dc.real, w)
    dc_smooth = dc

    col_in = np.copy(col_filt)

    col_in[dc >= ilim] = 2 * col_in[dc >= ilim] / dc[dc >= ilim]
    col_in[dc < ilim] = 1

    col_in -= 1
    col_in[col_in < 0] = 0

    if apodise:
        # locate sharp edges:
        grad = abs(np.gradient(dc_smooth)) / dc_smooth

        window_width = int(window_width)

        locs, _ = scipy.signal.find_peaks(grad, height=max_grad, distance=window_width)
        
        window_apod = 1 - scipy.signal.windows.hann(window_width*2)

        locs = locs[locs >= 20]
        if np.size(locs) != 0:
            for i in range(0,np.size(locs)):
                if window_width < locs[i] < np.size(col) - window_width:
                    col_in[locs[i] - window_width: locs[i] + window_width] = col_in[locs[i] - window_width: locs[i] + window_width]*window_apod
                elif locs[i] < window_width:
                    col_in[locs[i]:locs[i] + window_width] = col_in[locs[i]:locs[i] + window_width] * window_apod[window_width : (2*window_width) + 1]

        col_in *= scipy.signal.windows.tukey(col_in.shape[0], alpha=0.1)

    fn = fns[wtype]
    fft_carrier = scipy.fft.fft(col_in)
    col_length = len(fft_carrier)
    wdw_carrier = np.zeros(col_length)
    wdw_carrier[nfringes-halfwidth:nfringes+halfwidth+1] = 2*fn(N)

    fft_carrier = fft_carrier*wdw_carrier.T
    carrier = scipy.fft.ifft(fft_carrier)

    phase = np.angle(carrier)
    contrast = np.divide(abs(carrier), dc_smooth)

    # contrast[contrast > 1.] = 1.
    # contrast[contrast < 0.] = 0.

    # Calculate upper and lower bounds for contrast envelope:
    #contrast_envelope_lower = dc * (1 + contrast)
    #contrast_envelope_upper = dc * (1 - contrast)


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
