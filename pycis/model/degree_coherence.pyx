import cython
from libc.math cimport exp, cos, M_PI, fabs, sqrt
from pycis.model.phase_delay import uniaxial_crystal, savart_plate
from pycis.model.bbo import bbo

# functions from complex.h c library
cdef extern from "<complex.h>":

    double creal(double complex z)
    double complex cexp(double complex z)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)

cpdef degree_coherence_analytical(double[:] lines_wl, double [:] raw_lines_wl,
                    double[:] lines_rel_int, double inc_angle, double azim_angle_wp, double azim_angle_sp,
                    double l_wp, double l_sp, double v_thermal):

    """ Analytically calculate degree of coherence given .
    
    :param lines_wl: 
    :param raw_lines_wl: 
    :param lines_rel_int: 
    :param inc_angle: 
    :param azim_angle_wp: 
    :param azim_angle_sp: 
    :param l_wp: (m)
    :param l_sp: (m)
    :param v_thermal: (m/s)
    :return: 
    """

    # static type declaration

    cdef double C = 2.99792458e8  # speed of light [m/s]
    cdef int NUM_LINES = lines_wl.shape[0]

    cdef int m
    cdef double contrast, phase, freq_m, freq_md, wavelength_m, wavelength_md
    cdef double biref_m, n_e_m, n_o_m, kappa_m, phase_wp_m, phase_sp_m, phase_m, group_m, tau_m, deriv1, deriv2, deriv3
    cdef double complex degree_coherence, phase_factor, contrast_factor
    cdef double complex jj = 1j

    degree_coherence = 0

    # loop through lines

    for m in range(NUM_LINES):
        wavelength_md = lines_wl[m]
        wavelength_m = raw_lines_wl[m]

        freq_md = C / wavelength_md  # [Hz]
        freq_m = C / wavelength_m # [Hz]

        norm_freq_shift = (freq_md - freq_m) / freq_m

        biref_m, n_e_m, n_o_m = bbo(wavelength_m, 1)

        phase_wp_m = uniaxial_crystal(wavelength_m, n_e_m, n_o_m, l_wp, inc_angle, azim_angle_wp)  # waveplate delay (waves)
        phase_sp_m = savart_plate(wavelength_m, n_e_m, n_o_m, l_sp, inc_angle, azim_angle_sp)  # Savart plate delay (waves)

        phase_m = phase_wp_m + phase_sp_m  # (waves)
        group_m = phase_m * kappa_m  # (waves)

        tau_m = phase_m / freq_m

        phase_factor = cexp(2 * M_PI * jj * (phase_m + (group_m * (norm_freq_shift))))
        contrast_factor = exp(- (M_PI * freq_md * tau_m * kappa_m * (v_thermal / C)) ** 2)

        degree_coherence += lines_rel_int[m] * phase_factor * contrast_factor

    return degree_coherence


cpdef degree_coherence_numerical(double[:] wavelength_axis, double[:] spectrum, double inc_angle, double azim_angle_wp, double azim_angle_sp,
                    double l_wp, double l_sp):

    """ Calculates the interferogram instensity output for a --single pixel-- of a spatial heterodyne CIS instument.


    :return: 
    """

    # static type declaration:

    cdef int i, m
    cdef int n = wavelength_axis.shape[0]
    cdef double phase_i, phase_wp_i, phase_sp_i, phase_prev_i, igram, biref_i, n_e_i, n_o_i, d_wavelength_i, wavelength_i

    #
    # """ Given the line-integrated spectra, calculate the line-integrated coherence and output the phase and
    #      contrast."""
    #
    # freqs = C / wls[::-1]  # [Hz]
    #
    # # calculate line of sight integrated intensity:
    #
    # # convert spectra to frequency:
    # freq_spectrum = spectrum[::-1] * (C / freqs ** 2)  # [W / Hz]
    #
    # # integrate to obtain the 'intensity' and the normalised spectrum:
    # intensity = np.trapz(freq_spectrum, freqs)  # [W]
    #
    # norm_freq_spectrum = freq_spectrum / intensity  # [/ Hz]
    #
    # # setup numerical integral to obtain degree of coherence (doc)
    #
    # exp_factor = np.zeros_like(freqs, dtype=complex)
    # for k, freq in enumerate(freqs):
    #     wl_k = C / freq
    #     biref, n_e, n_o = pycis.model.bbo(wl_k, 0)
    #
    #     phase_wp_k = 2 * np.pi * pycis.model.uniaxial_crystal(wl_k, n_e, n_o, t_wp, inc_angle, azim_angle_wp)
    #     phase_sp_k = 2 * np.pi * pycis.model.savart_plate(wl_k, n_e, n_o, t_sp, inc_angle, azim_angle_sp)
    #
    #     phase_k = phase_wp_k + phase_sp_k
    #     exp_factor[k] = np.exp(1j * phase_k)
    #
    #
    # integrand = norm_freq_spectrum * exp_factor
    # doc = np.trapz(integrand, freqs)
    #
    # contrast = np.absolute(doc)
    # phase = np.angle(doc)
    #
    # return phase, contrast

    # Integrate over spectrum using trapezoidal rule on non-uniform grid

    igram = 0

    for i in range(1, n):
        wavelength_i = wavelength_axis[i]

        d_wavelength_i = wavelength_axis[i] - wavelength_axis[i-1]

        biref_i, n_e_i, n_o_i = bbo(wavelength_i, 1)

        phase_wp_i = uniaxial_crystal(wavelength_i, n_e_i, n_o_i, l_wp, inc_angle, azim_angle_wp)  # (waves)
        phase_sp_i = savart_plate(wavelength_i, n_e_i, n_o_i, l_sp, inc_angle, azim_angle_sp)  # (waves)

        phase_i = phase_wp_i + phase_sp_i # (waves)

        igram += (((spectrum[i - 1] / 2) * (1 + (cos(2 * M_PI * phase_prev_i)))) + ((spectrum[i] / 2) * (1 + (cos(2 * M_PI * phase_i))))) * d_wavelength_i

        phase_prev_i = phase_i

    return igram