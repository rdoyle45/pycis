import numpy as np


def unwrap(phase, flipwrap=False, centre=True):
    """ 
    1D and 2D phase unwrapping for CIS interferograms.
     
    In 2D, an ideal phase unwrapping algorithm sequentially unwraps image columns 
    and then image rows (or vice versa). For noisy raw_data however, this method performs very poorly. This 'pseudo 2D' 
    unwrapping algorithm relies on the strong vertical phase shear present in the CIS raw_data.
    
    In an ideal system the optical axis is projected onto the detector centre. On axis, the Savart plate introduces 
    zero net phase delay, so the (wrapped) phase measured at the centre of the detector array is the (wrapped) waveplate 
    phase delay -- a handy quantity to know. Numpy's np.unwrap function sets the 0th array element to be wrapped, set 
    'centre' to False to return this numpy-like behaviour. 'centre' defaults to True, wrapping the phase at the array 
    centre.
    
    (The good ideas here are Scott's.)
    
    :param phase: Input phase in radians, 1-D or 2-D.
    :type phase: array_like
    :param centre: Unwrap the phase such that the centre of the input array is wrapped. 
    :type centre: bool
    
    :return: Unwrapped phase (radians).
    """

    if phase.ndim == 1:
        # 1D phase unwrap:
        phase_uw = np.unwrap(phase)

        if centre:
            # wrap array column centre into [-pi, +pi] (assumed projection of optical axis onto detector)
            centre_idx = np.round((len(phase_uw) - 1) / 2).astype(np.int)
            phase_uw_centre = phase_uw[centre_idx]

            if phase_uw_centre > 0:
                while abs(phase_uw_centre) > np.pi:
                    phase_uw -= 2 * np.pi
                    phase_uw_centre = phase_uw[centre_idx]
            else:
                while abs(phase_uw_centre) > np.pi:
                    phase_uw += 2 * np.pi
                    phase_uw_centre = phase_uw[centre_idx]

    elif phase.ndim == 2:
        # pseudo 2D phase unwrap:

        if flipwrap:
            phase = np.flipud(phase)

        y_pix, x_pix = np.shape(phase)
        phase_contour = -np.unwrap(phase[int(np.round(y_pix / 2)), :])

        # sequentially unwrap image columns:
        phase_uw_col = np.zeros_like(phase)
        for i in range(0, x_pix):
            phase_uw_col[:, i] = np.unwrap(phase[:, i])

        phase_contour = phase_contour + phase_uw_col[int(np.round(y_pix/2)), :]
        phase_0 = np.tile(phase_contour, [y_pix, 1])
        phase_uw = phase_uw_col - phase_0

        if centre:
            # wrap image centre into [-pi, +pi] (assumed projection of optical axis onto detector)
            y_centre_idx = np.round((np.size(phase_uw, 0) - 1) / 2).astype(np.int)
            x_centre_idx = np.round((np.size(phase_uw, 1) - 1) / 2).astype(np.int)
            phase_uw_centre = phase_uw[y_centre_idx, x_centre_idx]

            if phase_uw_centre > 0:
                while abs(phase_uw_centre) > np.pi:
                    phase_uw -= 2 * np.pi
                    phase_uw_centre = phase_uw[y_centre_idx, x_centre_idx]
            else:
                while abs(phase_uw_centre) > np.pi:
                    phase_uw += 2 * np.pi
                    phase_uw_centre = phase_uw[y_centre_idx, x_centre_idx]

        if flipwrap:
            phase_uw = np.flipud(phase_uw)
                    


    else:
        raise Exception('# ERROR #   Phase input must be 1D or 2D array_like.')

    return phase_uw





