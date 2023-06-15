from .nac_induc_controller import NACInductiveController as nac
from .sane_induc_controller import SANEInductiveController as sane

def controller_entry(config):
    return globals()[config['type']](**config['kwargs'])