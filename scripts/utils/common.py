import pickle
from collections import defaultdict
from functools import reduce
from pathlib import Path

import numpy as np


def mkdir_path(path):
    base_dir = Path(path).parent
    Path(base_dir).mkdir(exist_ok=True, parents=True)

def group_by(key, seq):
    return reduce(lambda grp, val: grp[key(val)].append(val) or grp, seq, defaultdict(list))

def subtract_lists(list1, list2):
    return [item for item in list1 if item not in list2]


def lists_intersect(list1, list2):
    for item1 in list1:
        if item1 in list2:
            return True
    return False


def zero_division_safe(lambd):
    try:
        return lambd("whatever")
    except ZeroDivisionError:
        return 0


def split_into_sets_absolute(items, absolutes):
    splits = []

    current_index = 0

    for a in absolutes:
        last_index = current_index + a

        split = items[current_index:last_index]
        splits.append(split)

        current_index = last_index

    return splits


def split_into_sets(items, percentages):
    items = np.array(items)
    np.random.shuffle(items)

    splits = []

    length = len(items)
    current_index = 0

    for p in percentages:
        last_index = current_index + int(p * length)

        split = items[current_index:last_index]
        splits.append(split)

        current_index = last_index

    return splits


def get_n_random(items, n):
    np.random.shuffle(items)
    return items[:n]


def create_dirs(dirs):
    for d in dirs:
        Path(d).mkdir(exist_ok=True, parents=True)


def index_list_to_dict(array, key_provider, unique=True):
    result = {}
    for item in array:
        key = key_provider(item)

        if unique:
            if key not in result:
                result[key] = item
        else:
            if key not in result:
                result[key] = []
            result[key].append(item)

    return result


def pickle_obj(obj, path):
    dir_name = Path(path).parent
    Path(dir_name).mkdir(exist_ok=True, parents=True)
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_obj(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)
