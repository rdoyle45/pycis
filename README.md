# pycis

Analysis and modelling for the Coherence Imaging Spectroscopy (CIS) plasma diagnostic. 

Authorship:

- Joseph Allcock
- Scott Silburn: Author of the original Matlab scripts upon which most of the code is based. Author of ```pycis/data/get.py```  

# Modules

Package consists of three modules and a directory, demos, containing short demonstration scripts.

- **pycis.demod**: For the manipulation and demodulation of raw CIS data.
- **pycis.model**: For forward modelling the CIS instrument: this one is in a constant state of flux but the ```demos/model_calib.py``` and ```demos/model_cherab.py``` scripts will at least be kept up to date.
- **pycis.tools**: Some handy scripts that have popped up along the way.

# Prerequisites

Runs on python 3, compiled C for cython scripts is provided but will likely need to be re-compiled. Prerequisite packages (rough version numbers):

- numpy: 1.13.1
- scipy: 0.19.1
- matplotlib: 2.0.2
- cython: 0.25.2
- pandas
- imageio

# Setup

After cloning, the cython .pyx files in model need to be compiled, this is done by executing: 

```
python setup.py build_ext --inplace
```

from the terminal. This will be made easier!

# To Do

- demod:
  - Optimise window widths and functions to demodulation of realistic synthetic data using cherab.
  - How to pass kwargs to pool.map?

- model:
  - Remove repeated parts of code.
  - Implement realistic instrument throughput.
  - Incorporate multiprocessing in demod and model for a speedup





