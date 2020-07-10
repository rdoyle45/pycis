import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.constants import c
import pycis


def measure_degree_coherence(spectrum, wl_axis, delay, material=None):
    """
    calculate the degree of (temporal) coherence (DOC) of a given intensity spectrum, at a given interferometer delay.

    In general, DOC is a complex quantity. In the absence of dispersion, DOC is the Fourier transform of the
    (normalised) frequency spectrum. Since the spectrum is real, DOC will be an even function of interferometer delay
    -- so only positive delays are returned. The presence of instrument dispersion breaks the Fourier pair relationship
    of the spectrum and DOC.

    :param spectrum: [ arb. ]
    :type spectrum: np.ndarray

    :param wl_axis: wavelengths [ m ]
    :type wl_axis: np.ndarray

    :param delay: interferometer delay [ rad ]
    :type delay: float

    :param material: if material is specified, dispersion will be accounted for to first order about the weighted mean
    frequency
    :type material: str

    :return: degree_coherence
    """

    assert len(spectrum) == len(wl_axis)
    freq_axis = c / wl_axis
    spectrum_freq = spectrum * wl_axis ** 2 / c
    freq_com = np.trapz(freq_axis * spectrum_freq, freq_axis) / np.trapz(spectrum_freq, freq_axis)  # weighted mean
    # print(freq_com)

    # account for interferometer dispersion if birefringent material is specified
    if material is not None:
        kappa = pycis.dispersion(c / freq_com, material=material, output_derivatives=True)[3]
    else:
        kappa = 1

    delay_time = delay * kappa / (2 * np.pi * freq_com)
    integrand = spectrum_freq * np.exp(-2 * np.pi * 1j * delay_time * freq_axis)
    degree_coherence = np.trapz(integrand[::-1], freq_axis[::-1])

    return degree_coherence


def measure_degree_coherence_xr(spectrum, delay, material=None):
    """
    calculate the degree of (temporal) coherence (DOC) of a given intensity spectrum, at a given interferometer delay.

    In general, DOC is a complex quantity. In the absence of dispersion, DOC is the Fourier transform of the
    (normalised) frequency spectrum. Since the spectrum is real, DOC will be an even function of interferometer delay
    -- so only positive delays are returned. The presence of instrument dispersion breaks the Fourier pair relationship
    of the spectrum and DOC.

    :param spectrum: [ arb. ]
    :type spectrum: xr.DataArray with dim 'wavelength' in units ( m )

    :param delay: interferometer delay [ rad ]
    :type delay: xr.DataArray with dim 'delay' in units rad

    :param material: if material is specified, dispersion will be accounted for to first order about the weighted mean
    frequency
    :type material: str

    :return: degree_coherence
    """

    # convert spectrum's wavelength coordinate to frequency
    spectrum_freq = spectrum.copy(deep=True)
    spectrum_freq = spectrum_freq.rename({'wavelength': 'frequency'})

    # conversion m --> Hz
    spectrum_freq['frequency'] = c / spectrum_freq['frequency']
    spectrum_freq = spectrum_freq * c / spectrum_freq['frequency'] ** 2

    freq_com = float((spectrum_freq * spectrum_freq['frequency']).integrate(dim='frequency')) / \
               float(spectrum_freq.integrate(dim='frequency'))

    # account for interferometer dispersion if birefringent material is specified
    if material is not None:
        kappa = pycis.dispersion(c / freq_com, material=material, output_derivatives=True)[3]
    else:
        kappa = 1

    delay_time = delay * kappa / (2 * np.pi * freq_com)  # if kappa != 1 this is the GROUP delay_time
    integrand = spectrum_freq * np.exp(-2 * np.pi * 1j * delay_time * spectrum_freq['frequency'])
    # degree_coherence = np.trapz(integrand.values[::-1], freq_axis[::-1])
    degree_coherence = integrand.integrate(dim='frequency')
    degree_coherence = degree_coherence.assign_coords(delay=delay)

    return degree_coherence


def calculate_degree_coherence_full(spectrum, wl_axis, l_wp, material='a-BBO'):
    """

    :param spectrum:
    :param wl_axis:
    :return:
    """

    assert len(spectrum) == len(wl_axis)

    freq_axis = c / wl_axis
    spectrum_freq = spectrum * wl_axis ** 2 / c
    freq_com = np.trapz(freq_axis * spectrum_freq, freq_axis) / np.trapz(spectrum_freq, freq_axis)  # weighted mean
    biref = pycis.dispersion(wl_axis, material=material)[0]
    biref_com = pycis.dispersion(c / freq_com, material=material)[0]

    integrand = spectrum_freq * np.exp(-2 * np.pi * 1j * l_wp * biref * freq_axis / c)
    degree_coherence = np.trapz(integrand, freq_axis, axis=0)
    delay_com = 2 * np.pi * l_wp * biref_com * freq_com / c

    return degree_coherence, delay_com




def plot_degree_coherence(spectrum, wl_axis, delay_max=5e4, material=None, npts=None, axes=None, display=True):
    """
    basically the same as above but evaluated over many interferometer delays to make a nice plot

    :param spectrum: [ arb. ]
    :type spectrum: np.ndarray

    :param wl_axis: wavelengths [ m ]
    :type wl_axis: np.ndarray

    :param delay_max: [ rad ]
    :type delay_max: float

    :param material:
    :type material: str

    :param npts: number of points for the delay axis
    :type npts: int

    :return: degree_coherence
    """

    #TODO merge this with the above function, use xarray for max flexibility

    if npts is None:
        npts = len(wl_axis)

    freq_axis = c / wl_axis
    spectrum_freq = spectrum * wl_axis ** 2 / c
    freq_com = np.trapz(freq_axis * spectrum_freq, freq_axis) / np.trapz(spectrum_freq, freq_axis)  # weighted mean

    # account for interferometer dispersion if birefringent material is specified
    if material is not None:
        kappa = pycis.dispersion(c / freq_com, material=material, output_derivatives=True)[3]
    else:
        kappa = 1

    # define interferometer delay axis and calculate DOC
    delay_axis = np.linspace(0, delay_max, npts)
    delay_time_axis = delay_axis * kappa / (2 * np.pi * freq_com)  # [ s ]

    delay_time_mesh, freq_mesh = np.meshgrid(delay_time_axis, freq_axis)
    _, spectrum_freq_mesh = np.meshgrid(delay_axis, spectrum_freq)
    integrand = spectrum_freq_mesh * np.exp(-2 * np.pi * 1j * delay_time_mesh * freq_mesh)
    degree_coherence = np.trapz(integrand, freq_mesh, axis=0)

    # fwhm_stark = pystark.get_fwhm_stark('H', 5, 2, 5e20)
    # nu0 = c / pystark.get_wl_centre('H', 5, 2)
    # contrast_lz = np.exp(-(fwhm_stark / nu0) / 2 * delay_axis * kappa)

    # plot
    if display:
        if axes is None:
            fig = plt.figure(figsize=(9, 4))
            axes = (fig.add_subplot(121), fig.add_subplot(122))

        ax1, ax2 = axes
        wl_axis_nm = wl_axis * 1e9
        ax1.plot(wl_axis_nm, spectrum)

        # ax2.plot(delay_axis, np.real(degree_coherence), color='C0', label='Re($\gamma$)', lw=0.7)
        # ax2.plot(delay_axis, np.imag(degree_coherence), color='C1', label='Im($\gamma$)', lw=0.7)
        ax2.plot(delay_axis, abs(degree_coherence), ls='--', label='$|\gamma|$', lw=2)
        # ax2.plot(delay_axis, contrast_lz, ls=':', lw=2)

        # fwhm = pystark.get_fwhm_stark(n_upper, e_dens)
        # contrast_lorentzian = np.exp(- np.pi * fwhm * np.abs(delay_time_axis))
        # ax2.plot(delay_axis, contrast_lorentzian, color='C3', ls='--', label='$|\gamma|$', lw=2)

        ax1.set_xlabel('wavelength (nm)')
        ax1.set_title('spectrum')
        ax2.set_xlabel('delay (rad)')
        ax2.set_title('coherence')

        ax1.set_xlim([wl_axis.min() * 1e9, wl_axis.max() * 1e9])
        ax2.set_xlim([0, delay_max])
        ax2.set_yticks([0., 1.])
        leg = ax2.legend()

    return degree_coherence, delay_axis


if __name__ == '__main__':
    import pystark

    temp = 1 # [eV]
    dens = 5e20  # [m-3]
    bfield = 0.  # [T]
    viewangle = 90 * np.pi / 180  # [rad]
    line_model = 'voigt'

    # specify line
    species = 'H'
    n_upper = 5
    n_lower = 2

    bls = pystark.StarkLineshape(species, n_upper, n_lower, dens, temp, bfield, view_angle=viewangle, line_model=line_model)

    spectrum = bls.ls_szd
    wl_axis = bls.wl_axis

    # spectrum_xr = xr.DataArray(spectrum, coords=(wl_axis,), dims=('wavelength',), name='spectrum')
    # delay = np.linspace(0, 50000, 1000)
    # delay_xr = xr.DataArray(delay, coords=(delay,), dims=('delay',), name='delay')
    # doc = measure_degree_coherence_xr(spectrum_xr, delay_xr)
    # measure_degree_coherence(spectrum, wl_axis, 3000)

    plot_degree_coherence(spectrum, wl_axis, delay_max=3e4)



