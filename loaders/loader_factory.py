from loaders.liverct import LiverCtLoader
from loaders.kits import KitsLoader
from loaders.toy import ToyLoader

def init_loader(dataset):
    """
    Factory method for initialising data loaders by name.
    """
    if dataset == 'liverct':
        return LiverCtLoader()
    elif dataset == 'kits':
        return KitsLoader()
    elif dataset == 'toy':
        return ToyLoader()
    return None