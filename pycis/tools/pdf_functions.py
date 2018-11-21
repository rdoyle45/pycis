import numpy as np
import scipy.special
import scipy.stats
import matplotlib.pyplot as plt
import pycis


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
        print('1')

        phase_err_axis = np.linspace(-np.pi / 2, np.pi / 2, npts)
        q = amplitude * np.cos(phase_err_axis) / (np.sqrt(2) * sigma)

        log_pdf = np.log(1 / (2 * np.pi)) - (q / np.cos(phase_err_axis)) ** 2 + \
                  pycis.tools.logsumexp(np.array([np.zeros_like(q), np.log(q * np.sqrt(np.pi)) + q ** 2 + np.log(1 + scipy.special.erf(q))]), axis=0)

        phase_pdf = np.exp(log_pdf)
    else:
        print('2')
        phase_err_axis = np.linspace(-np.pi, np.pi, 5001)
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


def contrast_pdf(contdc_tru, sigma_contdc, dc_tru, sigma_dc, contrast_axis, npts=3000, display=False):
    """  Numerically evaluate the Rician-Normal ratio distribution to calculate the contrast distribution.

    :param contdc_tru:
    :param sigma_contdc:
    :param dc_tru:
    :param sigma_dc:
    :param contrast_axis:
    :param npts: number of points in contrast axis
    :param display:
    :return:
    """

    dc_axis = np.linspace(dc_tru - 5 * sigma_dc, dc_tru + 5 * sigma_dc, npts)

    dc_mesh, contrast_mesh = np.meshgrid(dc_axis, contrast_axis)
    contdc_mesh = dc_mesh * contrast_mesh

    pdf_contdc = scipy.stats.rice.pdf(contdc_mesh, contdc_tru / sigma_contdc, scale=sigma_contdc, loc=0.)
    pdf_dc = 1 / (sigma_dc * np.sqrt(2 * np.pi)) * np.exp(- (dc_mesh - dc_tru) ** 2 / (2 * sigma_dc ** 2))

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


