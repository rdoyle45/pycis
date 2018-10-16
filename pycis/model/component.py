import pycis
import pickle
import os.path
import glob
import pycis.paths as pp


def load_component(name, type):
    """ Load component by specifying component name. """

    if type == 'instrument':
        load_path = pp.instrument_path
    elif type == 'camera':
        load_path = pp.camera_path
    elif type == 'filter':
        load_path = pp.filter_path
    elif type == 'lens':
        load_path = pp.lens_path
    elif type == 'crystal':
        load_path = pp.crystal_path
    else:
        raise Exception('# ERROR # Please enter valid type.')

    try:
        component = pickle.load(open(os.path.join(load_path, name + '.p'), 'rb'))
    except FileNotFoundError:
        print("# ERROR #  Component file not found, to list exisiting component names use "
              "'pycis.model.list_components()'.")

    return component


def list_components():
    """ print list of all available model config of each type currently available to local pycis package. """

    component_paths = {'SynthImage': pp.synth_images_path, 'Lines': pp.lines_path, 'Instrument': pp.instrument_path,
                       'Camera': pp.camera_path, 'Filter': pp.filter_path, 'Crystal': pp.crystal_path,
                       'Lens': pp.lens_path}

    for key, value in component_paths.items():
        names = [pycis.tools.get_filename(x) for x in glob.glob(os.path.join(value, '*.p'))]
        names.sort()

        print('##### ' + key + ' #####')
        print()
        for name in names:
            print(name)
        print()
        print()



