from .backbone import *
from .searchspace import *

def model_entry(config):
    return globals()[config['type']](**config['kwargs'])