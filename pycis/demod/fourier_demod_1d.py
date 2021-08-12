import time
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
import pycis
import multiprocessing as mp
from functools import partial

def fourier_demod_1d(img,grad, width, ilim, wtype, wfactor, dval, filtval, nfringes=None, column_range=None, despeckle=False, tilt_angle=0, multiproc=True, display=False, apodise=False):
    """ 1-D Fourier demodulation of a coherence imaging interferogram image, looped over image columns to extract the DC, phase and contrast components.
    
    :param img: Input interferogram image to be demodulated.
    :type img: array_like
    :param mask: Mask low intensity portions of the image with NaNs
    :type mask: bool.
    :param despeckle: Remove speckles from image.
    :type despeckle: bool.
    :param nfringes: Manually set the carrier (fringe) frequency to be demodulated, in units of cycles per sequence -- approximately the number of fringes present in the image. If no value is given, the fringe frequency is found automatically.
    :param tilt_angle: interferometer tilt angle in degrees, default is zero. Tilted fringes feature has not yet been tested rigorously.
    :param display: Display a plot.
    :type display: bool.
    
    :return: A tuple containing DC (intensity), phase and contrast images.
    """

    start_time = time.time()
    raw_y_dim, raw_x_dim = np.shape(img)
    current_y_dim, current_x_dim = np.shape(img)

    # prepocess (pp) image
    if display:
        print('-- preprocessing...')

    pp_img = np.copy(img)

    # account for interferometer tilt (fringes tilted from horizontal)
    if tilt_angle != 0:
        pp_img = scipy.ndimage.rotate(pp_img, tilt_angle)
        current_y_dim, current_x_dim = np.shape(pp_img)
        column_range = [0, current_x_dim]
        
    if column_range is None:
        column_range = [0, current_x_dim]

    # remove neutron speckles
    if despeckle:
        pp_img = pycis.demod.despeckle(pp_img, dval)

    # column-wise demodulation over specified range
    if display:
        print('-- demodulating...')

    pool = mp.Pool(processes=mp.cpu_count()-2)
    fd_column_results = pool.map(partial(pycis.demod.fourier_demod_column, grad, width, ilim, wtype, wfactor, filtval, nfringes=nfringes, apodise=apodise), list(pp_img[:, column_range[0]:column_range[1]].T))
    dc, phase, contrast, S_apodised = zip(*fd_column_results)
    pool.close()
    
    dc = np.array(dc).T
    phase = np.array(phase).T
    contrast = np.array(contrast).T
    S_apodised = np.array(S_apodised).T
        
    if tilt_angle != 0:
        dc = scipy.ndimage.rotate(dc, -tilt_angle)

        dc = pycis.tools.get_roi(dc, roi_dim=[raw_x_dim, raw_y_dim])

    if display:
        # plot output:
        print('-- fd_image_1d: time elapsed: {:.2f}s'.format(time.time() - start_time))
        print('-- fd_image_1d: creating display images...')

        pycis.demod.display(img, dc, phase, contrast)
        display_fig.masking(plasma_frame, mask_frame, devim_frame, tolim_frame)
        plt.show()

    return dc, phase, contrast, S_apodised
