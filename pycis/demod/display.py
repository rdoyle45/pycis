import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

fsize_title = 12
fsize_cbar = 10
fsize_label = 8


def display(interferogram, dc, phase, contrast):
    """ Create plots for checking the performance of CIS demodulation.
    
    Called by the fd_image_#D functions when the 'display' kwarg is set to True.
    
    :param interferogram: 
    :param dc: 
    :param phase: 
    :param contrast: 
    :return: 
    """

    plt.figure(figsize=(9, 6))

    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(gs[2], sharex=ax1, sharey=ax1)
    ax4 = plt.subplot(gs[3], sharex=ax1, sharey=ax1)

    imgs = [interferogram, dc, phase, contrast]
    fns = [imshow_interferogram, imshow_dc, imshow_phase, imshow_contrast]
    axes = [ax1, ax2, ax3, ax4]

    for fn, ax, img in zip(fns, axes, imgs):
        # ax.set_adjustable('box-forced')
        fn(ax, img)

    plt.tight_layout()

    return


def cis_imshow(ax, im, cbar_label, vmin=None, vmax=None, **kwargs):
    if vmin is None:
        vmin = np.min(im)
    if vmax is None:
        vmax = np.max(im)

    im_obj = ax.imshow(im, interpolation='nearest', vmin=vmin, vmax=vmax, **kwargs)
    ax.set_xlabel('pix', size=fsize_label)
    ax.set_ylabel('pix', size=fsize_label)
    ax.tick_params(labelsize=fsize_label)

    cbar = plt.colorbar(im_obj, ax=ax)
    cbar.ax.tick_params(labelsize=fsize_cbar)
    cbar.set_label(cbar_label, size=fsize_cbar)


def imshow_interferogram(ax, im, vmin=None, vmax=None):
    ax.set_title('Interferogram', size=fsize_title)
    cis_imshow(ax, im, cbar_label='signal (DN)', vmin=vmin, vmax=vmax, cmap='gray')


def imshow_dc(ax, im, vmin=None, vmax=None):
    ax.set_title(r'DC', size=fsize_title)
    cis_imshow(ax, im, cbar_label='signal (DN)', vmin=vmin, vmax=vmax, cmap='gray')


def imshow_phase(ax, im):
    ax.set_title(r'Phase', size=fsize_title)
    cis_imshow(ax, im, cbar_label='Phase (rad)', cmap='viridis')


def imshow_contrast(ax, im):
    ax.set_title(r'Contrast', size=fsize_title)
    cis_imshow(ax, im, cbar_label='Contrast', cmap='viridis', vmin=0, vmax=1)


