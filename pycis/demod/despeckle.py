from matplotlib import pyplot as plt
import numpy as np
import scipy.signal
from pycis.tools.find_nearest import find_nearest
from pycis.tools.find_peaks import indexes


def despeckle(image, display=False):
    """ Remove random 'speckles' in a raw image caused by neutrons. 
    
    Function uses a median filters to get rid of single pixels and a 1st derivative threshold test to remove larger 
    speckles. Tune the three '_lim' parameters if not getting desired results.
    
    :param image: Input image to be despeckled.
    :type image: array_like
    :param display: Display a plot.
    :type display: bool.
    
    :return: The despeckled image.
    """

    # apply median filters (smooths while preserving edges)
    image_out = scipy.signal.medfilt(image, 5)

    # loop over image rows
    # y_dim, x_dim = image_out.shape
    # filtered_pixels = np.zeros_like(image, dtype=bool)
    #
    # for row in range(y_dim):
    #
    #     data_row = image_out[row, :]
    #
    #     # tuning parameters as in Scott's original code
    #     intensity_lim = 5  # Ilim - don't bother with parts of the image below this many counts
    #     detection_lim = 0.05  # Detection limit for speckles, lower = more sensitive.
    #     size_lim = 3  # Size limit - maximum number of pixels across to be considered a speckle.
    #
    #     # Intensity gradient down column
    #     grad = np.gradient(data_row)
    #     grad[data_row < intensity_lim] = 0
    #     grad = grad / np.std(grad)
    #
    #     # define the normalised detection threshold fed into peakutils.peak.indexes
    #     thres_normalised = (detection_lim - min(grad)) / (max(grad) - min(grad))
    #     starts = indexes(grad, thres=thres_normalised)
    #     ends = indexes(-1 * grad, thres=thres_normalised)
    #
    #     if np.size(starts) == 0:
    #         if display:
    #             print('-- despeckle: no large speckles found')
    #     else:
    #         # If detected speckle is below size limit, replace with the mean of its neighbouring pixels
    #         for i in range(np.size(starts)):
    #             endsind = find_nearest(ends, starts[i])
    #             if (np.size(endsind) != 0 and ends[endsind] <= (starts[i] + size_lim)) and (starts[i] > 2) and ((ends[endsind] + 2) < x_dim):
    #
    #                 x = np.arange(starts[i], ends[endsind] + 1)  # The x-coordinates of the interpolated values
    #                 xp = [starts[i] - 2, ends[endsind] + 2]  # The x-coordinates of the synth_data points, must be increasing
    #                 fp = data_row[xp]  # The y-coordinates of the synth_data points, same length as xp
    #                 data_row[x] = np.interp(x, xp, fp)
    #
    #                 filtered_pixels[row, starts[i]:ends[endsind]] = 1
    #                 image_out[row, :] = data_row
    #
    #             elif display: print('-- despeckle: no speckles found in row')
    #
    #
    # if display:
    #     # pix = np.linspace(0, x_dim - 1, x_dim)
    #
    #     f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    #
    #     im1 = ax1.imshow_raw(image)
    #     # cbar1 = plt.colorbar(im1, ax=ax1)
    #
    #     im2 = ax2.imshow_raw(image_out, vmax=np.max(image))
    #     # cbar2 = plt.colorbar(im2, ax=ax2)
    #
    #     im3 = ax3.imshow_raw(filtered_pixels)
    #     # cbar2 = plt.colorbar(im2, ax=ax2)
    #
    #     plt.show()

    return image_out



