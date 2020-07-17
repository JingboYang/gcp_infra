import copy
import argparse
import pathlib


def namespace_to_dict(args):
    """Turn a nested Namespace object into a nested dictionary."""
    args_dict = vars(copy.deepcopy(args))

    for arg in args_dict:
        obj = args_dict[arg]
        if isinstance(obj, argparse.Namespace):
            item = namespace_to_dict(obj)
            args_dict[arg] = item
        else:
            if isinstance(obj, pathlib.PosixPath):
                args_dict[arg] = str(obj)

    return args_dict

def fix_nested_namespaces(args):
    """Convert a Namespace object to a nested Namespace."""
    group_name_keys = []

    for key in args.__dict__:
        if '.' in key:
            group, name = key.split('.')
            group_name_keys.append((group, name, key))

    for group, name, key in group_name_keys:
        if group not in args:
            args.__dict__[group] = argparse.Namespace()

        args.__dict__[group].__dict__[name] = args.__dict__[key]
        del args.__dict__[key]

    return args

def get_experiment_number(storage, experiments_dir, experiment_name):
    """Parse directory to count the previous copies of an experiment."""
    dir_structure = storage.list_files(experiments_dir)

    if dir_structure[1] is None:
        return 0

    # import pdb; pdb.set_trace()

    dirnames = [exp_dir.split('/')[-1] for exp_dir in dir_structure[1]]

    ret = 1
    for d in dirnames:
        if d[:d.rfind('_')] == experiment_name:
            ret = max(ret, int(d[d.rfind('_') + 1:]) + 1)
    return ret