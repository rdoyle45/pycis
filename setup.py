import setuptools
import numpy as np
import os.path


# compilation_includes = ['.', np.get_include()]
# compilation_args = []
# cython_directives = {'language_level': 3}
# setup_path = os.path.dirname(os.path.abspath(__file__))
# extensions = []
# for root, dirs, files in os.walk(setup_path):
#     for file in files:
#         if os.path.splitext(file)[1] == ".pyx":
#             pyx_file = os.path.relpath(os.path.join(root, file), setup_path)
#             module = os.path.splitext(pyx_file)[0].replace("/", ".")
#             extensions.append(setuptools.Extension(module, [pyx_file], include_dirs=compilation_includes))
# ext_modules = cythonize(extensions, compiler_directives=cython_directives)

setuptools.setup(
    name='pycis',
    version='0.1',
    author='Joseph Allcock',
    description='Analysis and modelling for the Coherence Imaging Spectroscopy (CIS) plasma diagnostic',
    url='https://github.com/jsallcock/pycis',
    packages=setuptools.find_packages(),
    # cmdclass={'build_ext': build_ext},
    # ext_modules=ext_modules,  # generate .c files from .pyx
)
