import numpy as np
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import pycis
import dens_tools
import time

from scipy.constants import c
from skimage.transform import resize
import scipy.signal
import scipy.ndimage


def demo_1():
    """
    Conventional CIS instrument

    :return:
    """

    # define camera
    # pco.edge 5.5 camera
    bit_depth = 16
    # sensor_dim = (2560, 2160)
    sensor_dim = (1000, 1000)
    pix_size = 6.5e-6
    qe = 0.35
    epercount = 0.46  # [e / count]
    cam_noise = 2.5
    cam = pycis.Camera(bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise)

    # define imaging lens
    flength = 85e-3
    backlens = pycis.Lens(flength)

    # list interferometer components
    pol_1 = pycis.LinearPolariser(0)
    sp_1 = pycis.SavartPlate(np.pi / 4, 4.0e-3)
    wp_1 = pycis.UniaxialCrystal(np.pi / 4.1, 4.48e-3, 0)
    pol_2 = pycis.LinearPolariser(0)

    # first component in interferometer list is the first component that the light passes through
    interferometer = [pol_1, wp_1, sp_1, pol_2]

    # bringing it together into an instrument

    inst = pycis.Instrument(cam, backlens, interferometer)

    # wl = 466e-9
    # spec = 1e5

    wl0 = 464.9e-9
    std = 0.090e-9
    wl = np.linspace(wl0 - 3 * std, wl0 + 3 * std, 21)

    # generate spectrum
    spec = 1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(-1 / 2 * ((wl - wl0) / std) ** 2) * 1e5

    # pad speectrum to sensor array dimensions
    spec = np.tile(spec[:, np.newaxis, np.newaxis], [1, sensor_dim[0], sensor_dim[1]])

    # stokes parameters
    # a0 = np.zeros_like(spec)
    # spec = np.array([spec, a0, a0, a0])

    s = time.time()
    si = pycis.SynthImage(inst, wl, spec)
    e = time.time()
    print(e - s, ' seconds')

    si.img_igram()
    si.img_fft()
    plt.show()


def demo_2():
    """
    Polarisation camera CIS instrument

    """

    # FLIR BLACKFLY POLARISATION CAMERA
    bit_depth = 12
    sensor_dim = (2048, 2448)
    pix_size = 3.45e-6
    qe = 0.35
    epercount = 0.46  # [e / count]
    cam_noise = 2.5
    cam = pycis.PolCamera(bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise)
    # cam = pycis.Camera(bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise)

    # define imaging lens
    flength = 85e-3
    backlens = pycis.Lens(flength)

    # define interferometer components
    wp2 = pycis.UniaxialCrystal(np.pi / 4, 4.48e-3, 0, contrast=0.45)
    qwp = pycis.QuarterWaveplate(0)
    pol1 = pycis.LinearPolariser(0)
    pol2 = pycis.LinearPolariser(0)
    wp1 = pycis.UniaxialCrystal(np.pi / 4, 9.8e-3, 0, contrast=0.6)
    sp1 = pycis.SavartPlate(np.pi / 4, 6.2e-3)
    # first component in interferometer list is the first component that the light passes through
    interferometer = [pol1, sp1, wp1, pol2, wp2, qwp]
    # interferometer = [pol2, wp2, qwp]
    # interferometer = [pol1, sp1, wp1, pol2]

    # bringing it together into an instrument
    inst = pycis.Instrument(cam, backlens, interferometer)

    wl = 466e-9  # [ m ]
    spec = 1e4  # [ photons / pixel / time step ]

    s = time.time()
    si = pycis.SynthImage(inst, wl, spec)
    e = time.time()
    print(e - s, ' seconds')

    pycis.fourier_demod_doubledelay(si.igram, nfringes=62)

    # dc_1, phase_1, contrast_1 = pycis.fourier_demod_doubledelay(si.igram, display=True, nfringes=62)
    # dc_2, phase_2, contrast_2 = pycis.polcam_demod(dc_1)
    # pycis.fourier_demod_2d(I0, display=True)
    # I02, phi2, contrast2 = pycis.polcam_demod2(si.igram)

    plt.figure()
    plt.imshow(si.igram, 'gray')
    plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(np.log(abs(np.fft.rfft2(si.igram, axes=(1, 0)))))
    # plt.colorbar()
    # plt.show(block=True)

    # fig1 = plt.figure()
    # fig2 = plt.figure()
    #
    # gs1 = matplotlib.gridspec.GridSpec(nrows=2, ncols=2)
    # gs2 = matplotlib.gridspec.GridSpec(nrows=2, ncols=2)
    #
    # ax11 = fig1.add_subplot(gs1[0])
    # ax12 = fig1.add_subplot(gs1[1])
    # ax13 = fig1.add_subplot(gs1[2])
    # ax14 = fig1.add_subplot(gs1[3])
    #
    # ax21 = fig2.add_subplot(gs2[0])
    # ax22 = fig2.add_subplot(gs2[1])
    # ax23 = fig2.add_subplot(gs2[2])
    # ax24 = fig2.add_subplot(gs2[3])
    #
    # axes = (ax11, ax12, ax13, ax14, ax21, ax22, ax23, ax24)
    # ims = (si.igram, dc_1, phase_1, contrast_1, dc_1, dc_2, phase_2, contrast_2)
    # fs = (fig1, fig1, fig1, fig1, fig2, fig2, fig2, fig2)
    # for ax, im, f in zip(axes, ims, fs):
    #     i = ax.imshow(im)
    #     f.colorbar(i, ax=ax)

    # fig2 = plt.figure()
    # gs2 = matplotlib.gridspec.GridSpec(nrows=2, ncols=2)
    # ax1 = fig2.add_subplot(gs2[0])
    # ax2 = fig2.add_subplot(gs2[1])
    # ax3 = fig2.add_subplot(gs2[2])
    # ax4 = fig2.add_subplot(gs2[3])
    #
    # axes = (ax1, ax2, ax3, ax4)
    # ims = (si.igram, I02, phi2, contrast2)
    # for ax, im in zip(axes, ims):
    #     i = ax.imshow(im)
    #     fig1.colorbar(i, ax=ax)

    # plt.show()

def demo_pol_or():
    """
       Polarisation camera CIS instrument

       """

    # FLIR BLACKFLY POLARISATION CAMERA
    bit_depth = 12
    # sensor_dim = (2048, 2448)
    sensor_dim = (348, 348)
    pix_size = 3.45e-6
    qe = 0.35
    epercount = 0.46  # [e / count]
    cam_noise = 0.
    cam = pycis.PolCamera(bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise)

    # define imaging lens
    flength = 5e-3
    backlens = pycis.Lens(flength)

    # define interferometer components
    wp = pycis.UniaxialCrystal(-np.pi / 4, 4 * 1e-3, 0)
    qwp = pycis.QuarterWaveplate(np.pi / 2)
    pol_1 = pycis.LinearPolariser(np.pi / 2)
    # first component in interferometer list is the first component that the light passes through
    interferometer = [pol_1, wp, qwp]

    # bringing it together into an instrument
    inst = pycis.Instrument(cam, backlens, interferometer)

    # generate spectrum
    wl0 = 464.9e-9
    std = 1/2*0.090e-9
    wl = np.linspace(wl0 - 5 * std, wl0 + 5 * std, 50)
    i0_in = 5e2
    spec = 1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(-1 / 2 * ((wl - wl0) / std) ** 2) * i0_in
    # pad spectrum to sensor array dimensions
    spec = np.tile(spec[:, np.newaxis, np.newaxis], [1, sensor_dim[0], sensor_dim[1]])

    s = time.time()
    si = pycis.SynthImage(inst, wl, spec)
    e = time.time()
    print(e - s, ' seconds')

    img = si.igram

    i3 = img[::2, ::2]
    i2 = img[1::2, ::2]
    i4 = img[::2, 1::2]
    i1 = img[1::2, 1::2]

    i0 = i3 + i2 + i4 + i1
    phase = np.arctan2(i4 - i2, i3 - i1)
    contrast = 1 / i0 * np.sqrt(8 * ((i3 - i0 / 4) ** 2 + (i2 - i0 / 4) ** 2 + (i1 - i0 / 4) ** 2 + (i4 - i0 / 4) ** 2))

    phase0 = inst.calculate_ideal_phase_offset(wl0)
    doc_ideal = pycis.measure_degree_coherence(spec[:, 0, 0], wl, phase0, material=None)
    doc_disp = pycis.measure_degree_coherence(spec[:, 0, 0], wl, phase0, material='a-BBO')

    print(phase0)
    print(doc_ideal, abs(doc_ideal) / i0_in, np.angle(doc_ideal))
    print(doc_disp, abs(doc_disp) / i0_in, np.angle(doc_disp))

    # ref_ph = np.zeros_like(img)
    # ref_ph[::2, ::2] = 2 * np.pi / 2
    # ref_ph[1::2, ::2] = 2 * np.pi / 4
    # ref_ph[::2, 1::2] = 2 * 3 * np.pi / 4
    # ref_ph[1::2, 1::2] = 2 * 0
    # ref = np.exp(1j * ref_ph)
    # img_ref = img * ref
    # img_fft = np.fft.fftshift(np.fft.fft2(img))
    # img_ref_fft = np.fft.fftshift(np.fft.fft2(img_ref))

    fig = plt.figure()
    gs2 = matplotlib.gridspec.GridSpec(nrows=2, ncols=2)
    ax1 = fig.add_subplot(gs2[0])
    ax2 = fig.add_subplot(gs2[1])
    ax3 = fig.add_subplot(gs2[2])
    ax4 = fig.add_subplot(gs2[3])

    ax1.imshow(img)
    ax2.imshow(phase)
    ax3.imshow(contrast)
    ax4.imshow(i0)

    print(np.mean(contrast))

    plt.show()

    # I02, phi2, contrast2 = pycis.polcam_demod2(si.igram)
    #
    # fig2 = plt.figure()
    # gs2 = matplotlib.gridspec.GridSpec(nrows=2, ncols=2)
    # ax1 = fig2.add_subplot(gs2[0])
    # ax2 = fig2.add_subplot(gs2[1])
    # ax3 = fig2.add_subplot(gs2[2])
    # ax4 = fig2.add_subplot(gs2[3])
    #
    # axes = (ax1, ax2, ax3, ax4)
    # ims = (si.igram, I02, phi2, contrast2)
    # for ax, im in zip(axes, ims):
    #     i = ax.imshow(im)
    #     fig2.colorbar(i, ax=ax)
    #
    # plt.show()


def demo_testing_demodulation():
    # define camera
    # pco.edge 5.5 camera
    bit_depth = 16
    # sensor_dim = (2560, 2160)
    sensor_dim = (1000, 1000)
    pix_size = 6.5e-6
    qe = 0.35
    epercount = 0.46  # [e / count]
    cam_noise = 2.5
    cam = pycis.PolCamera(bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise)

    # define imaging lens
    flength = 85e-3
    backlens = pycis.Lens(flength)

    # list interferometer components
    pol_1 = pycis.LinearPolariser(0)
    wp_1 = pycis.UniaxialCrystal(np.pi / 4, 40e-3, 0)
    qwp = pycis.QuarterWaveplate(0)

    # first component in interferometer list is the first component that the light passes through
    interferometer = [pol_1, wp_1, qwp]

    # bringing it together into an instrument

    inst = pycis.Instrument(cam, backlens, interferometer)

    # wl = 466e-9
    # spec = 1e5

    wl = 464.9e-9
    spec = 1e5

    s = time.time()
    si = pycis.SynthImage(inst, wl, spec)
    e = time.time()
    print(e - s, ' seconds')

    # DEMODULATION
    img = si.igram

    # make low pass filter
    dy, dx = img.shape
    lpfilt_y = np.pad(scipy.signal.tukey(int(0.98 * dy), alpha=0.25), (int(0.01 * dy), int(0.01 * dy)), 'constant')
    lpfilt_x = np.pad(scipy.signal.tukey(int(0.98 * dx), alpha=0.25), (int(0.01 * dx), int(0.01 * dx)), 'constant')
    lpfilt_yy = np.tile(lpfilt_y, (dx, 1)).T
    lpfilt_xx = np.tile(lpfilt_x, (dy, 1))
    lpfilt = lpfilt_xx * lpfilt_yy

    dc = np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(img)) * lpfilt)))

    plt.figure()
    plt.imshow(dc)
    plt.colorbar()

    fig_p = plt.figure()
    ax_p = fig_p.add_subplot(111)
    imp = ax_p.imshow(pycis.wrap(-inst.calculate_ideal_phase_delay(wl)))
    fig_p.colorbar(imp, ax=ax_p)

    ref_ph = np.zeros_like(img, dtype=np.complex)
    ref_ph[::2, ::2] = 2 * 3 * np.pi / 4
    ref_ph[::2, 1::2] = 2 * np.pi / 2
    ref_ph[1::2, ::2] = 2 * 0
    ref_ph[1::2, 1::2] = 2 * np.pi / 4
    ref = np.exp(1j * ref_ph)

    img_ref = img.astype(float) * ref
    ft_img_ref = np.fft.fft2(img_ref)

    img_ref_lp = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(ft_img_ref) * lpfilt))

    si.img_igram()
    si.img_fft()

    plt.figure()
    plt.imshow(-np.angle(img_ref_lp))
    plt.colorbar()

    plt.figure()
    plt.imshow(2 * np.abs(img_ref_lp) / dc)
    plt.colorbar()

    plt.figure()
    plt.imshow(lpfilt)
    plt.colorbar()

    plt.show()


    return


def demo_multi_delay_cm():
    """
    multi-delay CIS instrument

    :return:
    """

    # define camera
    # pco.edge 5.5 camera
    bit_depth = 16
    # sensor_dim = (2560, 2160)
    sensor_dim = (1000, 1000)
    pix_size = 6.5e-6
    qe = 0.35
    epercount = 0.46  # [e / count]
    cam_noise = 2.5
    cam = pycis.Camera(bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise)

    # define imaging lens
    flength = 50e-3
    backlens = pycis.Lens(flength)

    # list interferometer components
    pol_1 = pycis.LinearPolariser(0)
    hwp = pycis.HalfWaveplate(-np.pi / 8)
    wp1 = pycis.UniaxialCrystal(-np.pi / 2, 6e-3, np.pi / 4, contrast=0.9)
    wp2 = pycis.UniaxialCrystal(0, 3e-3, np.pi / 4, contrast=0.9)
    wp3 = pycis.UniaxialCrystal(-np.pi / 4, 4e-3, np.pi / 4, contrast=0.9)
    pol_2 = pycis.LinearPolariser(-np.pi / 8)

    # first component in interferometer list is the first component that the light passes through
    interferometer = [pol_1, hwp, wp1, wp2, wp3, pol_2]

    # bringing it together into an instrument
    inst = pycis.Instrument(cam, backlens, interferometer)

    wl = 466e-9
    spec = 1e5

    # wl0 = 464.9e-9
    # std = 0.090e-9
    # wl = np.linspace(wl0 - 3 * std, wl0 + 3 * std, 21)
    #
    # generate spectrum
    # spec = 1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(-1 / 2 * ((wl - wl0) / std) ** 2) * 1e5

    # pad speectrum to sensor array dimensions
    # spec = np.tile(spec[:, np.newaxis, np.newaxis], [1, sensor_dim[0], sensor_dim[1]])

    # stokes parameters
    # a0 = np.zeros_like(spec)
    # spec = np.array([spec, a0, a0, a0])

    s = time.time()
    si = pycis.SynthImage(inst, wl, spec)
    e = time.time()
    print(e - s, ' seconds')

    si.img_igram()
    si.img_fft()
    plt.show()


def demo_multi_delay():
    """
    multi-delay CIS instrument

    :return:
    """

    # define camera
    # pco.edge 5.5 camera
    bit_depth = 16
    # sensor_dim = (2560, 2160)
    sensor_dim = (1000, 1000)
    pix_size = 6.5e-6
    qe = 0.35
    epercount = 0.46  # [e / count]
    cam_noise = 2.5
    cam = pycis.Camera(bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise)

    # define imaging lens
    flength = 50e-3
    backlens = pycis.Lens(flength)

    # list interferometer components
    pol_1 = pycis.LinearPolariser(np.pi / 4)
    sp_1 = pycis.SavartPlate(np.pi / 4, 15e-3)
    dp_1 = pycis.UniaxialCrystal(np.pi / 2, 2e-3, np.pi / 4, contrast=0.9)
    pol_2 = pycis.LinearPolariser(np.pi / 4)
    # wp_2 = pycis.UniaxialCrystal(-np.pi / 4, 9e-3, np.pi / 8, contrast=0.9)
    # pol_3 = pycis.LinearPolariser(0)

    # first component in interferometer list is the first component that the light passes through
    interferometer = [pol_1, sp_1, dp_1, pol_2]  # , wp_2, pol_3]

    # bringing it together into an instrument
    inst = pycis.Instrument(cam, backlens, interferometer)

    wl = 466e-9
    spec = 1e5

    # wl0 = 464.9e-9
    # std = 0.090e-9
    # wl = np.linspace(wl0 - 3 * std, wl0 + 3 * std, 21)
    #
    # generate spectrum
    # spec = 1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(-1 / 2 * ((wl - wl0) / std) ** 2) * 1e5

    # pad speectrum to sensor array dimensions
    # spec = np.tile(spec[:, np.newaxis, np.newaxis], [1, sensor_dim[0], sensor_dim[1]])

    # stokes parameters
    # a0 = np.zeros_like(spec)
    # spec = np.array([spec, a0, a0, a0])

    s = time.time()
    si = pycis.SynthImage(inst, wl, spec)
    e = time.time()
    print(e - s, ' seconds')

    si.img_igram()
    si.img_fft()
    plt.show()


if __name__ == '__main__':
    # demo_testing_demodulation()
    demo_multi_delay_consistency_check()
    # demo_2()
    # demo_pol_or()
    # demo_multi_delay()
