import time
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
import pycis
import multiprocessing as mp
from functools import partial


def fourier_demod_1d(img, grad, width, ilim, wtype, wfactor, dval, filtval, despeckle=False, tilt_angle=0, display=False, apodise=False):

    """ 1-D Fourier demodulation of a coherence imaging interferogram image, looped over image columns to extract the DC, phase and contrast components.
    
        Parameters:

            img   (np.array)       : Array containing Raw CIS Data
            grad  (float)          : Maximum intensity gradient considered a 'sharp edge' for filtering
            width (int)            : Width of appodisation window in pixels
            ilim  (int)            : Minimum Intensity value considered in demod - anything below this is set to 0
            wtype (str)            : Window function type for phase demodulation - 'hanning', 'blackmanharris' or 'tukey'
            wfactor (float)        : A multiplicative factor determining the width of the filters, multiplies nfringes.
            dval  (int)            : Size (in pixels) of despeckle filter
            filtval  (int)         : Size (in pixels) of convolved filter applied pre-demod
            despeckle  (bool)      : Turn despeckle on
            apodise  (bool)        : Turn apodisation on
            tilt_angle (float)     : Angle of CIS fringes
    
        Return: A tuple containing DC (intensity), phase and contrast images.
    """

    start_time = time.time()
    raw_y_dim, raw_x_dim = np.shape(img)

    # prepocess (pp) image
    if display:
        print('-- preprocessing...')

    pp_img = np.copy(img)

    # account for interferometer tilt (fringes tilted from horizontal)
    if tilt_angle != 0:
        pp_img = scipy.ndimage.rotate(pp_img, tilt_angle)

    # remove neutron speckles
    if despeckle:
        pp_img = pycis.demod.despeckle(pp_img, dval)

    # column-wise demodulation over specified range
    if display:
        print('-- demodulating...')

    # run demodulation routine in parallel for each column
    pool = mp.Pool(processes=mp.cpu_count()-2)
    fd_column_results = pool.map(partial(pycis.demod.fourier_demod_column, grad, width, ilim, wtype, wfactor, filtval, apodise=apodise), pp_img.T)
    dc, phase, contrast = zip(*fd_column_results)
    pool.close()

    # Repackage all the outputs
    dc = np.array(dc).T
    phase = np.array(phase).T
    contrast = np.array(contrast).T
        
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

    return dc, phase, contrast
