import numba
import xarray as xr

d_lambda = 1.e-10
d_lambda_mic = d_lambda * 1.e6

# dispersion data
d = {'kato1986':
         {'sellmeier_coefs': {'e': [2.3753, 0.01224, -0.01667, -0.01516],
                              'o': [2.7359, 0.01878, -0.01822, -0.01354]},
          'material': 'b-BBO', 'sellmeier_eqn_form': 1},
     'kato2010':
         {'sellmeier_coefs': {'e': [3.33469, 0.01237, -0.01647, 79.0672, -82.2919],
                              'o': [3.63357, 0.018778, -0.01822, 60.9129, -67.8505]},
          'material': 'b-BBO', 'sellmeier_eqn_form': 2},
     'eimerl':
         {'sellmeier_coefs': {'e': [2.3730, 0.0128, -0.0156, -0.0044],
                              'o': [2.7405, 0.0184, -0.0179, -0.0155]},
          'material': 'b-BBO', 'sellmeier_eqn_form': 1},
     'kim':
         {'sellmeier_coefs': {'e': [2.37153, 0.01224, -0.01667, -0.01516],
                              'o': [2.7471, 0.01878, -0.01822, -0.01354]},
          'material': 'a-BBO', 'sellmeier_eqn_form': 1},
     'agoptics':
         {'sellmeier_coefs': {'e': [2.3753, 0.01224, -0.01667, -0.01516],
                              'o': [2.7471, 0.01878, -0.01822, -0.01354]},
          'material': 'a-BBO', 'sellmeier_eqn_form': 1},
     'newlightphotonics':
         {'sellmeier_coefs': {'e': [2.31197, 0.01184, -0.01607, -0.00400],
                              'o': [2.67579, 0.02099, -0.00470, -0.00528]},
          'material': 'a-BBO', 'sellmeier_eqn_form': 1},

     'ghosh':
         {'sellmeier_coefs': {'e': [1.35859695, 0.82427830, 1.06689543e-2, 0.14429128, 120],
                              'o': [1.73358749, 0.96464345, 1.94325203e-2, 1.82831454, 120]},
          'material': 'calcite', 'sellmeier_eqn_form': 2},

     'shi':
         {'sellmeier_coefs': {'e': [4.607200, 0.108087, 0.052495, 0.014305],
                              'o': [3.778790, 0.070479, 0.045731, 0.009701]},
          'material': 'YVO', 'sellmeier_eqn_form': 1},
     # 'zelmon': ...,
     }

# set material default sources
default_sources = {'calcite': 'ghosh',
                   'b-BBO': 'eimerl',
                   'a-BBO': 'agoptics',
                   'YVO': 'shi'}


def calculate_dispersion(wl, material, source=None, ):
    """

    :param wl: wavelength [ m ]
    :param material: valid: 'a-BBO', 'b-BBO', 'calcite', 'YVO'
    :param source: shorthand citation for author
    :return:
    """

    wl_mic = wl * 1e6
    if source is None:
        source = default_sources[material]

    if material == 'a-BBO' and source == 'agoptics':
        # optimised ufuncs
        n_e = _sellmeier_eqn_abbo_e_ufunc(wl_mic, )
        n_o = _sellmeier_eqn_abbo_o_ufunc(wl_mic, )
    else:
        dd = d[source]
        sellmeier_coefs = dd['sellmeier_coefs']
        form = dd['sellmeier_eqn_form']
        sc_e = sellmeier_coefs['e']
        sc_o = sellmeier_coefs['o']
        n_e = _sellmeier_eqn(wl_mic, sc_e, form)
        n_o = _sellmeier_eqn(wl_mic, sc_o, form)

    biref = n_e - n_o
    return biref, n_e, n_o


def calculate_kappa(wl, material, source=None, ):
    """
    calculate kappa, the unitless first-order dispersion parameter

    :param wl:
    :param material:
    :param source:
    :return:
    """

    wl_mic = wl * 1e6

    if source is None:
        source = default_sources[material]

    if material == 'a-BBO' and source == 'agoptics':
        # optimised ufuncs
        kappa = _calculate_kappa_abbo_ufunc(wl_mic, )
    else:
        dd = d[source]
        sellmeier_coefs = dd['sellmeier_coefs']
        form = dd['sellmeier_eqn_form']
        sc_e = sellmeier_coefs['e']
        sc_o = sellmeier_coefs['o']

        wl_p1_mic = wl_mic + d_lambda_mic
        wl_m1_mic = wl_mic - d_lambda_mic

        biref, n_e, n_o = calculate_dispersion(wl, material, source=source, )
        biref_p1 = _sellmeier_eqn(wl_p1_mic, sc_e, form) - _sellmeier_eqn(wl_p1_mic, sc_o, form)
        biref_m1 = _sellmeier_eqn(wl_m1_mic, sc_e, form) - _sellmeier_eqn(wl_m1_mic, sc_o, form)

        biref_deriv = (biref_p1 - biref_m1) / (2 * d_lambda)
        kappa = 1 - (wl / biref) * biref_deriv

    return kappa


def _sellmeier_eqn(wl_mic, sellmeier_coefs, form):
    """

    :param wl_mic: wavelength /s [ microns ]
    :param sellmeier_coefs: list of coefficients
    :param form:
    :return:
    """

    if form == 1:
        return (sellmeier_coefs[0] + (sellmeier_coefs[1] / ((wl_mic ** 2) + sellmeier_coefs[2])) + (sellmeier_coefs[3] * (wl_mic ** 2))) ** 0.5
    elif form == 2:
        return (sellmeier_coefs[0] + (sellmeier_coefs[1] / ((wl_mic ** 2) + sellmeier_coefs[2])) + (sellmeier_coefs[3] / ((wl_mic ** 2) + sellmeier_coefs[4]))) ** 0.5


"""
for optimised, parallelised calculations, these numba-vectorized functions with hard-coded Sellmeier coefficients 
(a-BBO, source=agoptics) are used, wrapped into ufuncs using xarray.
"""


@numba.vectorize([numba.float64(numba.float64), ], nopython=True, fastmath=True, cache=True, )
def _sellmeier_eqn_abbo_e(wl_mic, ):
    a = 2.3753
    b = 0.01224
    c = -0.01667
    d = -0.01516
    return (a + (b / ((wl_mic ** 2) + c)) + d * (wl_mic ** 2)) ** 0.5


@numba.vectorize([numba.float64(numba.float64), ], nopython=True, fastmath=True, cache=True, )
def _sellmeier_eqn_abbo_o(wl_mic, ):
    a = 2.7471
    b = 0.01878
    c = -0.01822
    d = -0.01354
    return (a + (b / ((wl_mic ** 2) + c)) + d * (wl_mic ** 2)) ** 0.5


@numba.vectorize([numba.float64(numba.float64), ], nopython=True, fastmath=True, cache=True, )
def _calculate_kappa_abbo(wl_mic, ):

    d_lambda = 1.e-10
    d_lambda_mic = d_lambda * 1.e6
    wl_p1_mic = wl_mic + d_lambda_mic
    wl_m1_mic = wl_mic - d_lambda_mic

    biref_p1 = _sellmeier_eqn_abbo_e(wl_p1_mic) - _sellmeier_eqn_abbo_o(wl_p1_mic)
    biref_m1 = _sellmeier_eqn_abbo_e(wl_m1_mic) - _sellmeier_eqn_abbo_o(wl_m1_mic)
    biref = _sellmeier_eqn_abbo_e(wl_mic) - _sellmeier_eqn_abbo_o(wl_mic)

    biref_deriv = (biref_p1 - biref_m1) / (2 * d_lambda_mic)
    return 1 - (wl_mic / biref) * biref_deriv


def _sellmeier_eqn_abbo_e_ufunc(wl_mic, ):
    return xr.apply_ufunc(_sellmeier_eqn_abbo_e, wl_mic, dask='allowed', )


def _sellmeier_eqn_abbo_o_ufunc(wl_mic, ):
    return xr.apply_ufunc(_sellmeier_eqn_abbo_o, wl_mic, dask='allowed', )


def _calculate_kappa_abbo_ufunc(wl_mic, ):
    return xr.apply_ufunc(_calculate_kappa_abbo, wl_mic, dask='allowed', )
