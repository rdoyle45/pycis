# pycis

Analysis and modelling for the Coherence Imaging Spectroscopy (CIS) plasma diagnostic. 

Authorship:

- Joseph Allcock
- Scott Silburn: Author of the original Matlab scripts. Author of ```pycis/data/get.py```  

### Subpackages

- **pycis.demod**: Manipulation and demodulation of raw CIS data.
- **pycis.model**: Forward modelling the diagnostic and generating synthetic images
- **pycis.tools**: Some handy scripts that have popped up along the way.
- **pycis.demo**: demo scripts.
- **pycis.calib**: calibration tools (work in progress)
- **pycis.data**: loading CIS data from the 2013 MAST campaign (will only work on the Freia computing cluster)


### Prerequisites

Python 3. Prerequisite 
packages:

- numpy
- scipy
- matplotlib
- pandas
- imageio

### Setup

After cloning, setup the package from the top directory using: 

```
pip install -e .
```

from the terminal.






