d_lambda = 1.e-10
d_lambda_micron = d_lambda * 1.e6

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
                   # 'b-BBO': 'kato2010',
                   # 'a-BBO': 'newlightphotonics',
                   'a-BBO': 'agoptics',
                   # 'a-BBO': 'kim',
                   'YVO': 'shi'}


def dispersion(wl, material, output_derivatives=False, source=None):
    """
    
    :param wl: wavelength [ m ]
    :param material: valid: 'a-BBO', 'b-BBO', 'calcite', 'YVO' 
    :param source: shorthand citation for author
    :param output_derivatives: If info on birefringence derivatives wrt. wavelength is needed, smash True
    :return: 
    """

    if source is None:
        source = default_sources[material]

    dd = d[source]
    sellmeier_coefs = dd['sellmeier_coefs']
    form = dd['sellmeier_eqn_form']
    sc_e = sellmeier_coefs['e']
    sc_o = sellmeier_coefs['o']

    wl_mic = wl * 1e6
    n_e = sellmeier_eqn(wl_mic, sc_e, form=form)
    n_o = sellmeier_eqn(wl_mic, sc_o, form=form)

    biref = n_e - n_o
    
    if output_derivatives is False:
        return biref, n_e, n_o

    # first symmetric derivative of birefringence wrt. wavelength
    biref_deriv1 = (biref_dif(wl_mic, 1, sc_e, sc_o, form) - biref_dif(wl_mic, -1, sc_e, sc_o, form)) / (2 * d_lambda)

    # second symmetric derivative of birefringence wrt. wavelength
    biref_deriv2 = (biref_dif(wl_mic, 1, sc_e, sc_o, form) - (2 * biref) + biref_dif(wl_mic, -1, sc_e, sc_o, form)) / (d_lambda ** 2)

    # third symmetric derivative of birefringence wrt. wavelength
    biref_deriv3 = (biref_dif(wl_mic, 1.5, sc_e, sc_o, form) - (3 * biref_dif(wl_mic, 0.5, sc_e, sc_o, form)) + (
                   3. * biref_dif(wl_mic, -0.5, sc_e, sc_o, form)) - biref_dif(wl_mic, -1.5, sc_e, sc_o, form)) / (d_lambda ** 3)

    # first order dispersion factor kappa:
    kappa = 1 - (wl / biref) * biref_deriv1

    return biref, n_e, n_o, kappa, biref_deriv1, biref_deriv2, biref_deriv3


def dispersion_indices(wl, material, source=None):
    """

    :param wl:
    :param material:
    :param source:
    :return:
    """

    if source is None:
        source = default_sources[material]

    dd = d[source]
    sellmeier_coefs = dd['sellmeier_coefs']
    form = dd['sellmeier_eqn_form']
    sc_e = sellmeier_coefs['e']
    sc_o = sellmeier_coefs['o']

    wl_mic = wl * 1e6
    n_e = sellmeier_eqn(wl_mic, sc_e, form=form)
    n_o = sellmeier_eqn(wl_mic, sc_o, form=form)

    # first symmetric derivative of indices wrt. wavelength
    wl_dif_1p = wl_mic + d_lambda_micron
    wl_dif_1m = wl_mic - d_lambda_micron
    n_e_deriv1 = (sellmeier_eqn(wl_dif_1p, sc_e, form=form) - sellmeier_eqn(wl_dif_1m, sc_e, form=form)) / \
                 (2 * d_lambda)
    n_o_deriv1 = (sellmeier_eqn(wl_dif_1p, sc_o, form=form) - sellmeier_eqn(wl_dif_1m, sc_o, form=form)) / \
                 (2 * d_lambda)

    # second symmetric derivative of indices wrt. wavelength
    n_e_deriv2 = (sellmeier_eqn(wl_dif_1p, sc_e, form=form) - 2 * n_e + sellmeier_eqn(wl_dif_1m, sc_e, form=form)) / \
                 (d_lambda ** 2)
    n_o_deriv2 = (sellmeier_eqn(wl_dif_1p, sc_o, form=form) - 2 * n_o + sellmeier_eqn(wl_dif_1m, sc_o, form=form)) / \
                 (d_lambda ** 2)

    # third symmetric derivative of indices wrt. wavelength
    wl_dif_05p = wl_mic + 0.5 * d_lambda_micron
    wl_dif_05m = wl_mic - 0.5 * d_lambda_micron
    wl_dif_15p = wl_mic + 1.5 * d_lambda_micron
    wl_dif_15m = wl_mic - 1.5 * d_lambda_micron
    n_e_deriv3 = (sellmeier_eqn(wl_dif_15p, sc_e, form=form) - (3 * sellmeier_eqn(wl_dif_05p, sc_e, form=form)) +
                  (3. * sellmeier_eqn(wl_dif_05m, sc_e, form=form)) - sellmeier_eqn(wl_dif_15m, sc_e, form=form)) / (
                               d_lambda ** 3)
    n_o_deriv3 = (sellmeier_eqn(wl_dif_15p, sc_o, form=form) - (3 * sellmeier_eqn(wl_dif_05p, sc_o, form=form)) +
                  (3. * sellmeier_eqn(wl_dif_05m, sc_o, form=form)) - sellmeier_eqn(wl_dif_15m, sc_o, form=form)) / (
                         d_lambda ** 3)

    return n_e, n_e_deriv1, n_e_deriv2, n_e_deriv3, n_o, n_o_deriv1, n_o_deriv2, n_o_deriv3,


def sellmeier_eqn(wl_mic, sellmeier_coefs, form):
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


def biref_dif(wl_mic, dif_coef, sellmeier_coefs_e, sellmeier_coefs_o, form):
    """ 
    calculate birefringence at some product of d_lambda distance from wavelength. 
    """

    wl_dif = wl_mic + (dif_coef * d_lambda_micron)
    return sellmeier_eqn(wl_dif, sellmeier_coefs_e, form=form) - sellmeier_eqn(wl_dif, sellmeier_coefs_o, form=form)









