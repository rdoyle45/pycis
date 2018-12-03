#!python numbers=disabled
# read_spe.py
import time
import numpy as np


class ReadSPE(object):

    def __init__(self, fname):
        self._fid = open(fname, 'rb')
        self._load_size()

    def _load_size(self):
        self._xdim = np.int64(self.read_at(42, 1, np.int16)[0])
        self._ydim = np.int64(self.read_at(656, 1, np.int16)[0])

    def _load_date_time(self):
        rawdate = self.read_at(20, 9, np.int8)
        rawtime = self.read_at(172, 6, np.int8)
        strdate = ''
        for ch in rawdate :
            strdate += chr(ch)
        for ch in rawtime:
            strdate += chr(ch)
        self._date_time = time.strptime(strdate,"%d%b%Y%H%M%S")

    def get_size(self):
        return (self._xdim, self._ydim)

    def read_at(self, pos, size, ntype):
        self._fid.seek(pos)
        return np.fromfile(self._fid, ntype, size)

    def load_img(self):
        img = self.read_at(4100, self._xdim * self._ydim, np.uint16)
        return img.reshape((self._ydim, self._xdim))

    def close(self):
        self._fid.close()

def load(fname):
    fid = ReadSPE(fname)
    img = fid.load_img()
    fid.close()
    return img

if __name__ == "__main__":
    fname = '/Users/jsallcock/Documents/physics/phd/hardware/filters/og_cis_filters/tx_measured/12122017/raw/HeII/filtered_color_slit.SPE'

    rspe = ReadSPE(fname)
    a = 5