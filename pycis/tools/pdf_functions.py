import numpy as np
import scipy.special
import scipy.stats
import matplotlib.pyplot as plt
import pycis


def norm_pdf(mean, std, x=None):
    if x is None:
        nstd = 6
        npts = 1000
        lb, ub = mean - nstd * std, mean + nstd * std
        x = np.linspace(lb, ub, npts)
    pdf = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))
    return pdf, x


def phase_pdf(amplitude, sigma, npts=3001, display=False):
    """ probability density function for the angle / argument / phase of a circular, bivariate normal distribution --
    used in handling Fourier demodulation uncertainty.

    :param amplitude:
    :param sigma:
    :param npts: number of points in phase axis
    :param disp:
    :return:
    """

    phase_sigma = sigma / amplitude

    # for high SNR, do the calculation in log-space to avoid numerical overflow in np.exp()
    if phase_sigma < 0.1 * np.pi:
        phase_err_axis = np.linspace(-np.pi / 2, np.pi / 2, 5001)
        q = amplitude * np.cos(phase_err_axis) / (np.sqrt(2) * sigma)

        log_pdf = np.log(1 / (2 * np.pi)) + \
                  -(q / np.cos(phase_err_axis)) ** 2 + \
                  scipy.special.logsumexp(np.array(
                      [np.zeros_like(q), np.log(q * np.sqrt(np.pi)) + q ** 2 + np.log(1 + scipy.special.erf(q))]),
                                          axis=0)
        phase_pdf = np.exp(log_pdf)
    else:
        phase_err_axis = np.linspace(-np.pi, np.pi, npts)
        q = amplitude * np.cos(phase_err_axis) / (np.sqrt(2) * sigma)

        phase_pdf = 1 / (2 * np.pi) * np.exp(-(q / np.cos(phase_err_axis)) ** 2) * \
              (1 + (q * 2 * np.sqrt(np.pi) * np.exp(q ** 2)) * (1 - (0.5 * (1 - scipy.special.erf(q)))))

    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(phase_err_axis, phase_pdf)
        plt.show()

    phase_pdf /= np.trapz(phase_pdf, phase_err_axis)

    return phase_pdf, phase_err_axis


def contrast_pdf(contdc, contdc_sigma, dc_tru, dc_sigma, contrast_axis, npts=8000, display=False):
    """  Numerically evaluate the Rician-Normal ratio distribution to calculate the contrast distribution.

    :param contdc:
    :param contdc_sigma:
    :param dc_tru:
    :param dc_sigma:
    :param contrast_axis:
    :param npts: number of points in contrast axis
    :param display:
    :return:
    """

    dc_axis = np.linspace(dc_tru - 8 * dc_sigma, dc_tru + 8 * dc_sigma, npts)
    dc_mesh, contrast_mesh = np.meshgrid(dc_axis, contrast_axis)
    contdc_mesh = dc_mesh * contrast_mesh

    pdf_contdc = scipy.stats.rice.pdf(contdc_mesh, contdc / contdc_sigma, scale=contdc_sigma, loc=0.)
    pdf_dc, _ = norm_pdf(dc_tru, dc_sigma, x=dc_mesh)
    joint_pdf = pdf_contdc * pdf_dc

    # normalise joint PDF
    area = np.trapz(np.trapz(joint_pdf, contdc_mesh, axis=0), dc_axis)
    joint_pdf /= area

    # calculate the ratio pdf
    integrand = abs(dc_mesh) * joint_pdf
    contrast_pdf = np.trapz(integrand, dc_mesh, axis=1)

    if display:
        plt.figure()
        plt.imshow(pdf_contdc)
        plt.colorbar()

        plt.figure()
        plt.imshow(pdf_dc)
        plt.colorbar()

        plt.figure()
        plt.imshow(joint_pdf)
        plt.colorbar()

        plt.figure()
        plt.imshow(integrand)
        plt.colorbar()

        plt.figure()
        plt.plot(contrast_axis, contrast_pdf)

        plt.show()

    return contrast_pdf


