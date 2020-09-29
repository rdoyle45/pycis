import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage


def end_region_mask(img, alpha=0.1, mean_subtract=True, display=False):
    """
    Smoothly transition data to the image mean at the top and bottom edges, to reduce Gibbs phenomenon.
    
    :param img: 
    :param alpha: "Shape parameter of the Tukey window, representing the fraction of the window inside the cosine 
    tapered region. If zero, the Tukey window is equivalent to a rectangular window. If one, the Tukey window is 
    equivalent to a Hann window."
    :param mean_subtract: Does the mean need subtracting from the signal?
    :param display: 
    :return: 
    """

    img_dim = (np.shape(img)[0], np.shape(img)[1])
    img_out = np.copy(img).astype(np.float64)

    # create a tukey window for end-region masking:
    end_mask = np.tile(scipy.signal.tukey(img_dim[0], alpha=alpha), reps=(img_dim[1], 1)).T

    if mean_subtract:
        # estimate underlying mean
        img_mean = scipy.ndimage.filters.gaussian_filter(img, 50)
        img_out -= img_mean
        img_out *= end_mask
        img_out += img_mean

    else:
        img_out *= end_mask

    if display:
        plt.figure()
        plt.imshow(img_out, origin='lower')
        plt.colorbar()
        plt.show()

    return img_out





if __name__ == '__main__':
    import laser_scan_ga
    import pycis

    temp_inst = 34
    lsga = laser_scan_ga.LaserScanGA(temp_inst)
    image, wl, time = lsga.get_raw_img(15)

    # img_erm = end_region_mask(image)
    pycis.demod.fourier_demod_2d(image, mask=True, display=True)







