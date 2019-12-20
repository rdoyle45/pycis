import numpy as np
import copy

def wrap(phase, units='rad'):
    """
    Wrap N-D phase profile within (- pi, pi] radian interval.

    :param phase: input phase
    :type phase: array_like
    
    :param units: Units of phase (accepted: 'rad', 'waves', 'fringes').
    :type units: str
    
    :return: wrapped values (radians).
    """

    units_period = {'rad': 2 * np.pi, 'waves': 1, 'fringes': 1}
    assert units in units_period
    period = units_period[units]

    return (phase + period / 2) % period - (period / 2)


def unwrap(phase, centre=True):
    """
    Unwrap 1-D / 2-D phase profiles.

    In 2-D, a simple phase unwrapping algorithm sequentially unwraps columns and then rows (or vice versa). For noisy
    data however, this method can perform poorly.

    This 'pseudo 2D' unwrapping algorithm relies on the strong vertical phase shear present in the CIS raw_data.
    In an ideal system the optical axis is projected onto the detector centre. On axis, the Savart plate introduces
    zero net phase delay, so the (wrapped) phase measured at the centre of the detector array is the (wrapped) waveplate
    phase delay -- a handy quantity to know. Numpy's np.unwrap function sets the 0th array element to be wrapped, set
    'centre' to False to return this numpy-like behaviour. 'centre' defaults to True, wrapping the phase at the array
    centre.

    :param phase: input phase [ rad ], 1-D or 2-D.
    :type phase: np.ndarray

    :param centre: Unwrap the phase such that the centre of the input array is wrapped.
    :type centre: bool

    :return: Unwrapped phase [ rad ].
    """

    assert isinstance(phase, np.ndarray)

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
        # pseudo 2-D phase unwrap:

        y_pix, x_pix = np.shape(phase)
        phase_contour = -np.unwrap(phase[int(np.round(y_pix / 2)), :])

        phase_uw_col = np.unwrap(phase, axis=0)
        phase_contour = phase_contour + phase_uw_col[int(np.round(y_pix / 2)), :]
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

    else:
        raise Exception('# ERROR #   Phase input must be 1D or 2D array_like.')

    return phase_uw


# def wrap_centre(phase):
#     """
#     Wrap centre of a phase image into (- pi, pi] radian interval.
#
#     :param phase: [ rad ]
#
#     :return: phase_wc [ rad ]
#     """
#
#     assert isinstance(phase, np.ndarray)
#     assert phase.ndim == 2
#
#     phase_wc = copy.deepcopy(phase)
#
#     # wrap image centre into [-pi, +pi] (assumed projection of optical axis onto detector)
#     y_centre_idx = np.round((np.size(phase, 0) - 1) / 2).astype(np.int)
#     x_centre_idx = np.round((np.size(phase, 1) - 1) / 2).astype(np.int)
#     phase_centre = phase_wc[y_centre_idx, x_centre_idx]
#
#     if phase_centre > 0:
#         while abs(phase_centre) > np.pi:
#             phase_wc -= 2 * np.pi
#             phase_centre = phase_wc[y_centre_idx, x_centre_idx]
#     else:
#         while abs(phase_centre) > np.pi:
#             phase_wc += 2 * np.pi
#             phase_centre = phase_wc[y_centre_idx, x_centre_idx]
#
#     return phase_wc









