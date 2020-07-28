import importlib

def get_dataset(alias):
    module = importlib.import_module('loaders.' + alias)
    return module.Dataset
