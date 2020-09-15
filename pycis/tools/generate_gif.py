# import imageio
import re
import os
import numpy as np
from pprint import pprint

# imageio.plugins.freeimage.download()

try:
    from natsort import natsorted as sorted
except ImportError:
    pass


def fn_filter(dir, pattern, recursive=False, unique=False):
    """ Filenames in a given directory that match the search pattern
    TODO: add compatibility for non raw string file paths
    """
    fns = os.listdir(dir)
    p = re.compile(pattern)
    matches = []
    for fn in fns:
        if p.search(fn):
            matches.append(fn)
    if matches == []:
        print('No files match the supplied pattern: "%s"' % pattern)
    if unique:  # expect unique match so return scalar string that matches
        if len(matches) == 1:
            return matches[0]
        else:
            raise ('WARNING: fn_filter(unique=True): {} matches: {}'.format(len(matches), matches))
    else:
        return matches


def regexp_range(lo, hi, compile=False):
    fmt = '%%0%dd' % len(str(hi))
    if compile:
        return re.compile('(%s)' % '|'.join(fmt % i for i in range(lo, hi + 1)))
    else:
        return '(%s)' % '|'.join('{:d}'.format(i) for i in range(lo, hi + 1))


def gen_gif(path_in, pattern='.*', fn_out='movie.gif', duration=0.5, file_range=None, repeat={}, path_out=None,
            user_confirm=True):
    """Generate a gif from a collection of images in a given directory.
    path_in:        path of input images
    pattern         regular expression matching file to include in gif
    fn_out          filename for output gif
    duration        duration between frames in seconds
    file_range      replaces "{number}" in pattern with re matching a range of numbers
    repeat          dict of frame numbers and the number of times those frames should be repeated
    path_out        directory to write gif to
    """
    assert os.path.isdir(path_in)
    if path_out is None:
        path_out = path_in
    assert os.path.isdir(path_out)

    if (file_range is not None) and ('{range}' in fn_out):
        fn_out.format(range='{}-{}'.format(file_range[0], file_range[1]))

    if file_range is not None:
        assert '{number}' in pattern, 'Include "{number}" in pattern when using file range'
        pattern = pattern.format(number=regexp_range(*file_range))

    filenames = fn_filter(path_in, pattern)
    filenames = sorted(filenames)

    nframes = len(filenames)
    assert nframes > 0, 'No frames to create gif from'

    if -1 in repeat.keys():  # If repeating final frame, replace '-1' with index
        repeat[nframes-1] = repeat[-1]
        repeat.pop(-1)

    if user_confirm:
        print('{} frames will be combined into gif in: {}'.format(nframes, os.path.join(path_in, fn_out)))
        if nframes < 60:
            pprint(filenames)
        choice = input('Proceed? [y/n]: ')
        if not choice == 'y':
            print('gif was not produced')
            return  ## return from function without renaming

    with imageio.get_writer(os.path.join(path_out, fn_out), mode='I', format='GIF-FI', duration=0.12) as writer:  # duration = 0.4

        for i, filename in enumerate(filenames):
            image = imageio.imread(os.path.join(path_in, filename))
            writer.append_data(image)
            if repeat is not None and i in repeat.keys():
                for j in np.arange(repeat[i]):
                    writer.append_data(image)

    print('Wrote gif containing {} frames to: {}'.format(nframes, os.path.join(path_in, fn_out)))


if __name__ == '__main__':
    path_in = '/Users/jsallcock/Documents/physics/phd/output/animations/uniaxial_crystal_cut_angle/imgs/'
    fn_out = 'movie.gif'  # Output file name

    # pattern = '.*f{number}.png'
    # pattern = '.*{}.png'.format(regexp_range(*file_number_range) if file_number_range else '')
    pattern = '.*.png'

    file_number_range = None  # No number filter

    repeat = {0: 5, -1: 10}  # Number of additional times to repeat each frame number
    duration = 1.0  # Frame duration in seconds

    gen_gif(path_in, pattern, duration=duration, fn_out=fn_out, file_range=file_number_range, repeat=repeat)