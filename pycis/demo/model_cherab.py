import inspect
import pycis
import pickle

import matplotlib.pyplot as plt


from raysect.core.workflow import SerialEngine
from raysect.optical import World
from raysect.optical.observer import VectorCamera
from raysect.primitive.mesh import import_stl
from raysect.optical.material.lambert import Lambert
from raysect.optical.material.absorber import AbsorbingSurface
from raysect.optical.spectralfunction import ConstantSF
from raysect.optical.observer import RGBPipeline2D, SpectralPipeline2D, PowerPipeline2D

# Cherab and raysect imports
from cherab.core.atomic import Line
from cherab.core.atomic.elements import deuterium, carbon
from cherab.core.model import ExcitationLine, RecombinationLine
from cherab.tools.observers import load_calcam_calibration
from cherab.solps import load_solps_from_mdsplus
from cherab.openadas import OpenADAS

# Demo script for producing synthetic CIS images using cherab
plt.ion()

solps_ref_numbers = [69665]

save_path = '/home/jallcock/cherab/cherab_cis/initial_tests_oct17/SXD_10MW_cryo_Hmode/cherab_spectral_output/'
mds_server = 'solps-mdsplus.aug.ipp.mpg.de:8001'

world = World()

# Load all parts of mesh with chosen material
MESH_PARTS = ['/projects/cadmesh/mast/mastu-light/mug_centrecolumn_endplates.stl',
            '/projects/cadmesh/mast/mastu-light/mug_divertor_nose_plates.stl']

for path in MESH_PARTS:
    import_stl(path, parent=world, material=AbsorbingSurface())  # Mesh with perfect absorber
    # import_stl(path, parent=world, material=Lambert(ConstantSF(0.25)))  # Mesh with 25% Lambertian reflectance
    # import_stl(path, parent=world, material=Debug(Vector3D(0.0, 1.0, 0.0)))  # Mesh with debugging material


# Load plasma from SOLPS model
solps_ref_number = 69665  #69637
sim = load_solps_from_mdsplus(mds_server, solps_ref_number)
plasma = sim.create_plasma(parent=world)
plasma.atomic_data = OpenADAS(permit_extrapolation=True)
mesh = sim.mesh
vessel = mesh.vessel


# Pick emission models
# d_alpha = Line(deuterium, 0, (3, 2))
# plasma.models = [ExcitationLine(d_alpha), RecombinationLine(d_alpha)]
#
# d_gamma = Line(deuterium, 0, (5, 2))
# plasma.models = [ExcitationLine(d_gamma), RecombinationLine(d_gamma)]

ciii_465 = Line(carbon, 2, (11, 10))
plasma.models = [ExcitationLine(ciii_465)]


# Select from available Cameras
camera_config = load_calcam_calibration('/home/jallcock/calcam/VirtualCameras/cis_div_view.nc')


# RGB pipeline for visualisation
rgb = RGBPipeline2D(display_unsaturated_fraction=0.96, name="sRGB")

# Get the power and raw spectral data for scientific use.
power_unfiltered = PowerPipeline2D(display_unsaturated_fraction=0.96, name="Unfiltered Power (W)")
power_unfiltered.display_update_time = 15
spectral = SpectralPipeline2D()


# Setup camera for interactive use...
pixels_shape, pixel_origins, pixel_directions = camera_config
camera_cherab = VectorCamera(pixel_origins, pixel_directions, pipelines=[rgb, power_unfiltered, spectral], parent=world)
camera_cherab.min_wavelength = 464.9
camera_cherab.max_wavelength = 465.1

camera_cherab.spectral_bins = 50
camera_cherab.pixel_samples = 1
camera_cherab.observe()

pycis_name = 'demo_cherab_' + str(solps_ref_number)

# load cis instrument:
cis_instrument = pycis.model.load_component('demo_instrument', type='instrument')

# create SpectraCherab instance:
pycis_spectra = pycis.model.SpectraCherab(spectral.wavelengths, spectral.frame.mean, cis_instrument, pycis_name)

# save spectra once generated to avoid having to generate it again!
pycis_spectra.save()

# create synthetic cis image:
pycis_image = pycis.model.create_synth_image(cis_instrument, pycis_spectra, pycis_name)


pycis_image.img_raw()
plt.show()

pycis_image.save()



# pickle.dump(spectral.wavelengths, open(save_path + str(solps_ref_numbers[solps_idx]) + '_wavelength_axis.p', 'wb'))
# pickle.dump(spectral.frame.mean, open(save_path + str(solps_ref_numbers[solps_idx]) + '_spectra.p', 'wb'))

