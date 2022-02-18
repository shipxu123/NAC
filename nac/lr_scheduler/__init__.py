from torch.optim.lr_scheduler import _LRScheduler, StepLR, CosineAnnealingLR

def scheduler_entry(config):
    return globals()[config.type](**config.kwargs)