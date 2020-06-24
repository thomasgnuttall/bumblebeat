import os 

import yaml 

def load_yaml(fname):
    """
    Load yaml at <path> to dictionary, d

    returns: dict
    """
    if not os.path.isfile(fname):
        return None

    with open(fname) as f:
        conf = yaml.load(f, Loader=yaml.Loader)

    return conf


def split_range(r1, r2, n):
    """
    Split range <r1> - <r2> into <n> equal size buckets
    """
    step = (r2 - r1)/n
    return [r1+step*i for i in range(n+1)]


def get_bucket_number(value, srange):
    """
    Return index of bucket that <value> falls into

    srange is a list of bucket divisions from split_range()
    """
    assert srange == (sorted(srange)),\
        "srange must be sorted list"
    assert len(set(srange)) == len(srange),\
        "srange buckets must be unique"
    assert value <= max(srange) and value >= min(srange),\
        "value is not in any srange bucket"

    for i in range(len(srange)-1):
        if value <= srange[i+1]:
            return i


def create_dir_if_not_exists(path):
    """
    If the directory at <path> does not exist, create it empty
    """
    directory = os.path.dirname(path)
    # Do not try and create directory if path is just a filename
    if (not os.path.exists(directory)) and (directory != ''):
        os.makedirs(directory)


def round_to_nearest_multiple(n, mul):
    return mul * round(n/mul)