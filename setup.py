from setuptools import setup
from distutils.extension import Extension
import Cython
import numpy as np
import os.path
from Cython.Build import cythonize
from Cython.Distutils import build_ext

path = os.path.dirname(os.path.realpath(__file__))

phase_delay_pyx_file = os.path.join(path, 'model', 'phase_delay.pyx'),
degree_coherence_pyx_file = os.path.join(path, 'model', 'degree_coherence.pyx'),
bbo_pyx_file = os.path.join(path, 'model', 'bbo.pyx'),


ext_modules = [
    Extension('model.phase_delay', [phase_delay_pyx_file[0]], include_dirs=[np.get_include()]),
    Extension('model.degree_coherence', [degree_coherence_pyx_file[0]], include_dirs=[np.get_include()]),
    Extension('model.bbo', [bbo_pyx_file[0]], include_dirs=[np.get_include()]),
]

setup(
    name='pycis',
    version='0.1',
    description='Analysis and modelling for the Coherence Imaging Spectroscopy (CIS) plasma diagnostic',
    url='https://github.com/jsallcock/pycis',
    cmdclass={'build_ext': build_ext},
    ext_modules = ext_modules,
)


