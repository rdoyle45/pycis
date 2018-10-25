import setuptools
import numpy as np
import os.path
from Cython.Build import cythonize


compilation_includes = [".", np.get_include()]
compilation_args = []
cython_directives = {
    'language_level': 3
}
setup_path = os.path.dirname(os.path.abspath(__file__))

# try compiling cython files
try:
    # build .pyx extension list -- this is taken from cherab
    extensions = []
    for root, dirs, files in os.walk(setup_path):
        for file in files:
            if os.path.splitext(file)[1] == ".pyx":
                pyx_file = os.path.relpath(os.path.join(root, file), setup_path)
                module = os.path.splitext(pyx_file)[0].replace("/", ".")
                extensions.append(setuptools.Extension(module, [pyx_file], include_dirs=compilation_includes, extra_compile_args=compilation_args),)
    ext_modules = cythonize(extensions)

except Exception as e:
    print('--pycis: failed cython file compilation, you will not be able to import pycis.model')
    print(e)
    ext_modules = []


setuptools.setup(
    name='pycis',
    version='0.1',
    author='Joseph Allcock',
    description='Analysis and modelling for the Coherence Imaging Spectroscopy (CIS) plasma diagnostic',
    url='https://github.com/jsallcock/pycis',
    packages=setuptools.find_packages(),
    # cmdclass={'build_ext': build_ext},
    ext_modules = ext_modules,  # generate .c files from .pyx
)


