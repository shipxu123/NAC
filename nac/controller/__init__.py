from .nac_induc_controller import NACInductiveController as nac

def controller_entry(config):
    return globals()[config['type']](**config['kwargs'])