
# jsallcock
# created: 04/10/2017

import matplotlib as mpl
import numpy as np
import scipy as sp
import scipy.io as sio
import os
import glob
import time
import sys

from matplotlib import pyplot as plt
from matplotlib import image as mpimg

def get_filename(path):
    """ Returns filename string given complete path string. """
    return os.path.splitext(os.path.basename(path))[0]


def is_scalar(var):
    """ True if variable is scalar """
    if hasattr(var, "__len__"):
        return False
    else:
        return True

def get_fwhm(x, y, disp=False):
    """ given a function with a SINGLE PEAK, find the FWHM without fitting. QUICK AND DIRTY. """

    # normalise height
    y_norm = y / np.max(y)

    # split array into two about the max value
    idx_max = np.argmax(y_norm)
    x_l, x_u, y_l, y_u = x[:idx_max], x[idx_max + 1:], y_norm[:idx_max], y_norm[idx_max + 1:]

    # 1D interpolation
    hm_idx_l, hm_idx_u = np.interp(0.5, y_l, x_l), np.interp(0.5, y_u[::-1], x_u[::-1])

    fwhm = hm_idx_u - hm_idx_l

    if disp:
        print(fwhm)
        plt.figure()
        plt.plot(x, y_norm, '-')
        plt.plot(hm_idx_l, 0.5, '.')
        plt.plot(hm_idx_u, 0.5, '.')

        plt.show()

    return fwhm


def is_number(s):
    """
    TODO: Test on numbers and strings and arrays
    """
    try:
        n = str(float(s))
        if n == "nan" or n == "inf" or n == "-inf": return False
    except ValueError:
        try:
            complex(s)  # for complex
        except ValueError:
            return False
    return True


def safe_len(var):
    """ Length of variable returning 1 instead of type error for scalars """
    if is_scalar(var):  # checks if has atribute __len__
        return 1
    elif len(np.array(var) == np.nan) == 1 and var == np.nan:  # If value is NaN return zero length
        return 0
    else:
        return len(var)


def get_roi(input_image, centre=None, roi_dim=(50, 50), display=False):
    """ Given an input image, returns the centred region of interest (ROI) with user-specified dimensions. """

    y_dim, x_dim = np.shape(input_image)

    if centre is None:
        y_dim_h = y_dim / 2
        x_dim_h = x_dim / 2
    else:
        y_dim_h = centre[0]
        x_dim_h = centre[1]

    roi_width_h = roi_dim[1] / 2
    roi_height_h = roi_dim[0] / 2

    y_lo = int(y_dim_h - roi_height_h)
    y_hi = int(y_dim_h + roi_height_h)

    x_lo = int(x_dim_h - roi_width_h)
    x_hi = int(x_dim_h + roi_width_h)

    img_roi = input_image[y_lo:y_hi, x_lo:x_hi]

    if display:
        roi_display = np.zeros_like(input_image)
        roi_display[y_lo:y_hi, x_lo:x_hi] = 1

        plt.figure()
        plt.imshow(input_image)
        plt.colorbar()

        plt.figure()
        plt.imshow(roi_display)
        plt.colorbar()

        plt.show()


    return img_roi


def printProgress(iteration, total, prefix='', suffix='', frac=False, t0=None,
                  decimals=2, nth_loop=1, barLength=50):
    """
    from http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

python - Text Progress Bar in the Console - Stack Overflow
stackoverflow.com
Is there a good way to do the following? I wrote a simple console app to upload and download files from an FTP server using the ftplib. Each time some raw_data chunks are ...

    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    if iteration % nth_loop != 0:  # Only print every nth loop to reduce slowdown from printing
        return
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    filledLength = int(round(barLength * iteration / float(total)))
    percents = round(100.00 * (iteration / float(total)), decimals)
    bar = '█' * filledLength + '-' * (barLength - filledLength)
    frac = '{}/{} '.format(iteration, total) if frac else ''
    if t0 is None:
        time = ''
    else:
        t1 = datetime.now()
        t_diff_past = relativedelta(t1, t0)  # time past in loop
        mul = float(total - iteration) / iteration if iteration > 0 else 0
        t_diff_rem = t_diff_past * mul  # estimate of remaining time
        t_diff_past = '({h}h {m}m {s}s)'.format(h=t_diff_past.hours, m=t_diff_past.minutes, s=t_diff_past.seconds)
        if t_diff_rem.hours > 0:  # If expected to take over an hour display date and time of completion
            t_diff_rem = (datetime.now() + t_diff_rem).strftime("(%d/%m/%y %H:%M)")
        else:  # Display expected time remaining
            t_diff_rem = '({h}h {m}m {s}s)'.format(h=t_diff_rem.hours, m=t_diff_rem.minutes, s=t_diff_rem.seconds)
        time = ' {past} -> {remain}'.format(past=t_diff_past, remain=t_diff_rem)

    sys.stdout.write('\r %s |%s| %s%s%s%s %s' % (prefix, bar, frac, percents, '%', time, suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()


def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return rad * 180 / np.pi


def normalise(array):
    return array / np.max(array)