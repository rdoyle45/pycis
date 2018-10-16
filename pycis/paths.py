import os
import inspect

# get pycis paths:
root_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
demod_path = os.path.join(root_path, 'uncertainty')
model_path = os.path.join(root_path, 'model')
demos_path = os.path.join(root_path, 'demo')
general_path = os.path.join(root_path, 'tools')

# define user data paths:
config_path = os.path.join(model_path, 'config')

synth_images_path = os.path.join(config_path, 'synth_image')
camera_path = os.path.join(config_path, 'camera')
instrument_path = os.path.join(config_path, 'instrument')
lines_path = os.path.join(config_path, 'lines')
images_path = os.path.join(config_path, 'images')
filter_path = os.path.join(config_path, 'filter')
crystal_path = os.path.join(config_path, 'crystal')
lens_path = os.path.join(config_path, 'lens')
spectra_path = os.path.join(config_path, 'spectra')

# Check whether the config directory exists. If it doesn't, create it:
if not os.path.isdir(config_path):
    print('***** pycis Setup *****', 'Creating config directory at:')
    print(config_path)
    os.makedirs(config_path)

# Check whether individual sub directories exist, create them if they don't:
sub_directory_paths = [synth_images_path, camera_path, instrument_path, lines_path, images_path, filter_path,
                       crystal_path, lens_path, spectra_path]

for path in sub_directory_paths:
    if not os.path.isdir(path):
        print('***** pycis Setup *****', 'Creating directory at:')
        print(path)
        os.makedirs(path)









