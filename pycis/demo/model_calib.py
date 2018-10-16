import numpy as np
import matplotlib.pyplot as plt
import pycis
from PIL import Image


# This script is currently the best place to keep up with changes to how pycis.model is working:


def camera():

    # define a new instrument
    inst = pycis.model.Instrument('mastu_heii', 0.9)  # 0.56)

    # add components
    inst.add_camera('photron_SA4')
    inst.add_lens_3('sigma_150mm')
    inst.add_filter('ccfe_HeII_m')

    inst.add_crystal('ccfe_4.6mm_waveplate', orientation=3 * np.pi / 4)
    inst.add_crystal('ccfe_6.2mm_savartplate', orientation=1 * np.pi / 4)
    inst.save()

    # Specfify spectral line to observe
    line_name = 'Cd_lamp'
    I0 = 7e5
    vi = 0
    ti = 1

    spectra = pycis.model.SpectraCalib(line_name, I0, vi, ti, inst)
    image = pycis.model.create_synth_image(inst, spectra, 'demo_calib')

    image.img_raw()
    plt.show()
    image.save()

    return


def cd_calib():
    """ Create two Cd calib images"""


    inst_508nm = pycis.model.load_component('mastu_cdi_508nm', type='instrument')
    inst_468nm = pycis.model.load_component('mastu_cdi_468nm', type='instrument')

    line_name_508nm = 'Cd_508nm'
    line_name_468nm = 'Cd'

    i0_508nm = inst_508nm.get_snr_intensity(line_name_508nm, 8)
    ti_508nm = inst_508nm.get_contrast_ti(line_name_508nm, 0.5)

    i0_468nm = inst_468nm.get_snr_intensity(line_name_468nm, 8)
    ti_468nm = inst_468nm.get_contrast_ti(line_name_468nm, 0.5)

    vi = 0

    spectra_508nm = pycis.model.SpectraCalib(line_name_508nm, i0_508nm, vi, ti_508nm, inst_508nm)
    image_508nm = pycis.model.create_synth_image(inst_508nm, spectra_508nm, 'Cd_calib_508nm')

    spectra_468nm = pycis.model.SpectraCalib(line_name_468nm, i0_468nm, vi, ti_468nm, inst_468nm)
    image_468nm = pycis.model.create_synth_image(inst_468nm, spectra_468nm, 'Cd_calib_468nm')

    image_508nm.img_raw(limits=[0, 100])
    image_468nm.img_raw(limits=[0, 100])

    plt.show()

    image_508nm.save()
    image_468nm.save()
    return


if __name__ == '__main__':
    # novelty()
    # camera()
    # cd_calib()
    camera()

# def fibre():
#
#     camera_config = get_camera_config('fibre')
#     instrument_config = get_instrument_config('CIII_default')
#     x_pix = camera_config.x_pix
#     y_pix = camera_config.y_pix
#
#     I0_plasma = 1e20
#     I0_calib = 1e20
#
#     vi_plasma = 5000
#     vi_calib = 0
#
#     Ti_plasma = 20
#     Ti_calib = 3000
#
#     Ti_calib = 10
#     line_name = 'CIII'
#     B = 0
#     theta = 0
#
#     print('--Making SynthSpectra..')
#     synth_spectra_plasma = SynthSpectra(line_name=line_name, I0=I0_plasma, vi=vi_plasma, Ti=Ti_plasma, B=B,
#                                         theta=theta,
#                                         camera_config=camera_config, LUT=True)
#     synth_spectra_calib = SynthSpectra(line_name=line_name, I0=I0_calib, vi=vi_calib, Ti=Ti_calib, B=B, theta=theta,
#                                        camera_config=camera_config, LUT=True)
#
#     print('--Making SynthImage..')
#     synth_image_plasma = SynthImage(synth_spectra=synth_spectra_plasma, instrument_config=instrument_config)
#     synth_image_calib = SynthImage(synth_spectra=synth_spectra_calib, instrument_config=instrument_config)
#
#     plasma_savename = 'CIII_fibre'
#     calib_savename = 'CIII_fibre'
#
#     # save pickles
#     synth_image_plasma.save(plasma_savename)
#     synth_image_calib.save(calib_savename, type='calib')
#
#     return



