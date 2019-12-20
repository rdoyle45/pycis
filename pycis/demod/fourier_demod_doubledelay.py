import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage

import pycis


def fourier_demod_doubledelay(img, nfringes=None, display=False):
    """
    2D Fourier demodulation of a coherence imaging interferogram image, extracting the DC, phase and contrast.

    Option to output uncertainty info too.

    :param img: CIS interferogram image to be demodulated.
    :type img: array_like

    :param nfringes: Manually set the carrier (fringe) frequency to be demodulated, in units of cycles per sequence --
    approximately the number of fringes present in the image. If no value is given, the fringe frequency is found
    automatically.
    :type nfringes: float.

    :param display: Display a plot.
    :type display: bool.

    :return: A tuple containing the DC component (intensity), phase and contrast.
    """

    # since the input image is real, its FT is Hermitian -- all info contained in +ve frequencies -- use rfft2()
    fft_img = np.fft.rfft2(img, axes=(1, 0))

    # estimate carrier (fringe) frequency, if not supplied
    if nfringes is None:
        nfringes = np.unravel_index(fft_img[15:][:].argmax(), fft_img.shape)[0] + 15

    # generate window function

    # window to isolate carrier
    fft_length = int(img.shape[0] / 2)
    window_iso_1d = pycis.demod.window(fft_length, nfringes, width_factor=1., fn='tukey')
    window_iso_2d = np.transpose(np.tile(window_iso_1d, (fft_img.shape[1], 1)))
    window_iso_2d *= (1 - scipy.signal.tukey(fft_img.shape[1], alpha=0.6))

    # window to suppress harmonics
    window_sup1_1d = pycis.demod.window(fft_length, fft_length - nfringes, window_width=nfringes, width_factor=1., fn='tukey')
    window_sup1_2d = np.transpose(np.tile(window_sup1_1d, (fft_img.shape[1], 1)))
    window_sup1_2d *= (1 - scipy.signal.tukey(fft_img.shape[1], alpha=0.6))
    window_sup1_2d = 1 - window_sup1_2d

    window_sup2_1d = pycis.demod.window(fft_length, nfringes, window_width=nfringes, width_factor=1., fn='tukey')
    window_sup2_2d = np.transpose(np.tile(window_sup2_1d, (fft_img.shape[1], 1)))
    window_sup2_2d *= np.concatenate((np.array([0] * (int(2 * fft_img.shape[1] / 5) + 1)),
                                      scipy.signal.tukey(int(fft_img.shape[1] / 5), alpha=0.95),
                                      np.array([0] * (int(2 * fft_img.shape[1] / 5) + 1))))[:fft_img.shape[1]]

    window_dc = (1 - window_iso_2d) * window_sup1_2d * (1 - window_sup2_2d)

    # extract DC
    fft_dc = fft_img * window_dc
    dc = np.fft.irfft2(fft_dc, axes=(1, 0))

    # extract carrier
    fft_carrier = fft_img * window_iso_2d
    carrier = np.fft.irfft2(fft_carrier, axes=(1, 0))
    analytic_signal = scipy.signal.hilbert(carrier, axis=-2)
    phase = np.angle(analytic_signal)
    contrast = np.abs(analytic_signal) / scipy.ndimage.filters.gaussian_filter(dc, sigma=2)

    if display:
        plt.figure()
        plt.imshow(window_iso_2d)
        plt.colorbar()

        plt.figure()
        plt.imshow(window_sup1_2d)
        plt.colorbar()

        plt.figure()
        plt.imshow(window_sup2_2d)
        plt.colorbar()

        plt.figure()
        plt.imshow(window_dc)
        plt.colorbar()

        plt.figure()
        plt.imshow(dc)
        plt.colorbar()

        plt.figure()
        plt.imshow(np.log(abs(np.fft.fftshift(np.fft.fft2(img, axes=(1, 0)), axes=(1, 0)))))
        plt.colorbar()

        plt.figure()
        plt.imshow(phase)
        plt.colorbar()

        plt.figure()
        plt.imshow(contrast)
        plt.colorbar()

        plt.figure()
        plt.imshow(img)
        plt.colorbar()

        plt.show(block=True)


    return dc, phase, contrast