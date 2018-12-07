# Image Clustering
# Artificial Intelligence Workshop
# Pylie.com, developed by

from os import listdir
from os.path import isfile, join


def read_file_dirs(dir):

    names = [f for f in listdir(dir) if isfile(join(dir, f))]

    names_no_hidden = [f for f in names if f[0] != '.']

    names_with_dot = [dir + f for f in names_no_hidden if '.' in f]

    return names_with_dot
