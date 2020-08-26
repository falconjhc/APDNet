from loaders import toy
loader = toy

params = {
    'normalise': 'batch',
    'seed': 1,
    'folder': 'experiment_unet_toy',
    'epochs': 35,
    'batch_size': 8,
    'split': 0,
    'dataset_name': 'toy',
    'test_dataset': 'toy',
    'prefix': 'norm',  # Prefix used to load a dataset, e.g. norm_baseline for data_by_dog
    'augment': True,
    'model': 'unet.UNet',
    'executor': 'base_executor.Executor',
    'num_masks': loader.ToyLoader().num_patho_masks + loader.ToyLoader().num_anato_masks,
    'out_channels': loader.ToyLoader().num_patho_masks + loader.ToyLoader().num_anato_masks + 1,
    'outputs':1, # harric added 20191004
    'residual': False,
    'deep_supervision': False,
    'filters': 32,
    'downsample': 3,
    'input_shape': loader.ToyLoader().input_shape,  # harric modified
    'modality': 'LGE',
    'image_downsample': 1,
    'lr': 0.0001,
    'l_mix': 1,
    'decay': 0.,
    'regularizer': 0,
    'ce_weight': 0.3
}


def get(segmentation_option):
    return params
