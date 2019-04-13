import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec
import pycis
import time


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

    # define imaging lens
    flength = 85e-3
    backlens = pycis.Lens(flength)

    # define interferometer components
    wp2 = pycis.UniaxialCrystal(np.pi / 4, 4.48e-3, 0)
    qwp = pycis.QuarterWaveplate(0)
    pol1 = pycis.LinearPolariser(0)
    pol2 = pycis.LinearPolariser(0)
    wp1 = pycis.UniaxialCrystal(np.pi / 4, 6.5e-3, 0)
    sp1 = pycis.SavartPlate(np.pi / 4, 4e-3)
    # first component in interferometer list is the first component that the light passes through
    interferometer = [pol1, sp1, wp1, pol2, wp2, qwp]

    # bringing it together into an instrument
    inst = pycis.Instrument(cam, backlens, interferometer)

    wl = 466e-9  # [ m ]
    spec = 1e4  # [ photons / pixel / time step ]

    s = time.time()
    si = pycis.SynthImage(inst, wl, spec)
    e = time.time()
    print(e - s, ' seconds')

    dc, phase, contrast = pycis.fourier_demod_2d(si.igram, display=True, nfringes=40)
    I0, phi, contrast = pycis.polcam_demod(dc)
    # I02, phi2, contrast2 = pycis.polcam_demod2(si.igram)

    fig1 = plt.figure()
    gs1 = matplotlib.gridspec.GridSpec(nrows=2, ncols=2)
    ax1 = fig1.add_subplot(gs1[0])
    ax2 = fig1.add_subplot(gs1[1])
    ax3 = fig1.add_subplot(gs1[2])
    ax4 = fig1.add_subplot(gs1[3])

    axes = (ax1, ax2, ax3, ax4)
    ims = (si.igram, I0, phi, contrast)
    for ax, im in zip(axes, ims):
        i = ax.imshow(im)
        fig1.colorbar(i, ax=ax)

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

    plt.show()


def demo_pol_or():
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

    # define imaging lens
    flength = 85e-3
    backlens = pycis.Lens(flength)

    # define interferometer components
    wp = pycis.UniaxialCrystal(np.pi / 4, 4.48e-3, 0)
    qwp = pycis.QuarterWaveplate(0)
    pol = pycis.LinearPolariser(0)
    # first component in interferometer list is the first component that the light passes through
    interferometer = [pol]

    # bringing it together into an instrument
    inst = pycis.Instrument(cam, backlens, interferometer)

    wl = 466e-9  # [ m ]
    spec = 1e4  # [ photons / pixel / time step ]

    s = time.time()
    si = pycis.SynthImage(inst, wl, spec)
    e = time.time()
    print(e - s, ' seconds')

    I02, phi2, contrast2 = pycis.polcam_demod2(si.igram)

    fig2 = plt.figure()
    gs2 = matplotlib.gridspec.GridSpec(nrows=2, ncols=2)
    ax1 = fig2.add_subplot(gs2[0])
    ax2 = fig2.add_subplot(gs2[1])
    ax3 = fig2.add_subplot(gs2[2])
    ax4 = fig2.add_subplot(gs2[3])

    axes = (ax1, ax2, ax3, ax4)
    ims = (si.igram, I02, phi2, contrast2)
    for ax, im in zip(axes, ims):
        i = ax.imshow(im)
        fig2.colorbar(i, ax=ax)

    plt.show()


if __name__ == '__main__':
    demo_2()
    # demo_pol_or()
