import importlib

def get_model(alias):
    module = importlib.import_module('models.' + alias)
    return module.Model
