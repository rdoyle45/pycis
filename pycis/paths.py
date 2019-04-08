import os, inspect

"""
The saved filter bandpass data will probably be moved elsewhere, but for now, the saved filter bandpass data can be 
accessed using pycis.model.FilterFromName or else directly using pycis.paths.filters_path

"""

root_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
filters_path = os.path.join(root_path, 'data', 'bandpass_filters')