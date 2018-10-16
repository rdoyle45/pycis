import numpy as np
from matplotlib import pyplot as plt
import colorsys


def flow_intensity_composite(intensity, velocity, intensity_display_range=None, velocity_display_range=None):
    """ create an intensity-velocity composite image.
    
    Given intensity and velocity images, will take the intensity output from IFdemod, and
    the ~calibrated flow images together to create a composite image where
    flow is represented by image colour and intensity by image brightness.
    
    :param intensity: intensity image
    :param velocity: velocity image
    :param intensity_display_range: 
    :param velocity_display_range: 
    :return: 
    """

    x_pix, y_pix = np.shape(intensity)

    # Default intensity scale [0, max]
    if intensity_display_range is None:
        intensity_display_range=[0, np.max(intensity)]
    # Default velocity scale [min, max]
    if velocity_display_range is None:
        velocity_display_range=[np.min(velocity), np.max(velocity)]

    # wrap intensity and velocity into display range and normalise [0, 1] :
    intensity = (intensity - intensity_display_range[0]) / (intensity_display_range[1] - intensity_display_range[0])
    intensity[intensity > 1] = 1
    intensity[intensity < 0] = 0
    velocity = (velocity - velocity_display_range[0]) / (velocity_display_range[1] - velocity_display_range[0])
    velocity[velocity > 1] = 1
    velocity[velocity < 0] = 0

    # convert velocity from grayscale to RGB with colormap:
    cmap = plt.get_cmap('bwr')
    rgba_v_avg = cmap(velocity)
    rgb_v_avg = np.delete(rgba_v_avg, 3, 2)

    # convert from RGB to HSV:
    hsv_v_avg = np.zeros_like(rgb_v_avg)
    rgb_composite = np.zeros_like(rgb_v_avg)
    for y in range(y_pix):
        for x in range(x_pix):
            hsv_v_avg[y, x, :] = colorsys.rgb_to_hsv(rgb_v_avg[y, x, 0], rgb_v_avg[y, x, 1], rgb_v_avg[y, x, 2])
            hsv_v_avg[y, x, 2] = intensity[y, x]
            rgb_composite[y, x, :] = colorsys.hsv_to_rgb(hsv_v_avg[y, x, 0], hsv_v_avg[y, x, 1], hsv_v_avg[y, x, 2])

    return rgb_composite
