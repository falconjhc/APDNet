import numpy as np
from configuration.kits import unet_config_kits
from configuration import discriminator_config
from loaders import kits
import copy

loader = kits

params = {
    'seed': 1,
    'folder': 'experiment_sdnet_kits-Base',
    'data_len': 0,
    # 'epochs': 35,
    'batch_size': 8,
    'pool_size': 50,
    'split': 0,
    'description': '',
    'dataset_name': 'kits',
    'test_dataset': 'kits',
    'input_shape': loader.KitsLoader().input_shape,
    'image_downsample': 1,
    'modality': 'LGE',
    'prefix': 'norm',                         # Prefix used to load a dataset, e.g. norm_baseline for data_by_dog
    'augment': True,
    'model': 'psdnet.SDNet',
    'executor': 'psdnet_executor.SDNetExecutor',
    'num_anatomy_masks': loader.KitsLoader().num_anato_masks,
    'out_anatomy_channels': loader.KitsLoader().num_anato_masks + 1,
    'num_pathology_masks': loader.KitsLoader().num_patho_masks,
    'out_pathology_channels': loader.KitsLoader().num_patho_masks + 1,
    # 'l_mix': '1.-1.', # percentage of the labeled data to be used during the training
    'ul_mix': 0, # percentage of the unlabled data to be used during the training
    'rounding': 'encoder',
    'num_mask_channels': 8,
    'num_z': 8,
    'w_adv_X': 0.0,
    'w_rec_X': 1.0,
    'w_rec_Z': 1.,
    'w_rec_s': 0.,
    'w_rec_p': 0.,
    'w_kl': 0.1,
    'w_sup_AnatoMask': 3,
    'w_sup_PathoMask': 4.5,
    'w_dc': 0,
    'lr': 0.0001,
    'decay': 0.,
    'regularizer': 0.,
    'pseudo_health_weight_rate': 0.,
    'real_pathology_weight_rate': 1.0,
    'pred_pathology_weight_rate': 1.0,
    'real_anatomy_weight_rate': 1.0,
    'pred_anatomy_weight_rate': 1.0,
    'patience': 25,
    'min_delta': 0.001,
    'spectrum_regularization_anatomy': 20,
    'spectrum_regularization_pathology': 10,
    'spectrum_regularization_reconstruction': 10,
    'triplet_margin': 0.3,
    'triplet_weight': 0.,
    'pe_weight':0.,
    'ce_focal_anato_weight':1.,
    'ce_focal_patho_weight':1.,
    'decoder':'Film', # 'Film' or 'Film-Mod'
    'pathology_encoder': 'unet' # 'unet' or 'plain'
}

d_mask_anato_params = copy.copy(discriminator_config.params)
d_mask_anato_params['name'] = 'D_Mask_Anato'
d_mask_anato_params['decay'] = params['decay']
d_mask_anato_params['auxiliary_classifier'] = False
d_mask_anato_params['spectrum_regularization'] = params['spectrum_regularization_anatomy']
d_mask_anato_params['triplet_output'] = False


d_mask_patho_params = copy.copy(discriminator_config.params)
d_mask_patho_params['name'] = 'D_Mask_Patho'
d_mask_patho_params['decay'] = params['decay']
d_mask_patho_params['auxiliary_classifier'] = False
d_mask_patho_params['spectrum_regularization'] = params['spectrum_regularization_pathology']
d_mask_patho_params['triplet_output'] = False



d_reconstruct_params = copy.copy(discriminator_config.params)
d_reconstruct_params['name'] = 'D_Reconstruction'
d_reconstruct_params['decay'] = params['decay']
d_reconstruct_params['output'] = '2D'
d_reconstruct_params['auxiliary_classifier'] = True
d_reconstruct_params['spectrum_regularization'] = params['spectrum_regularization_reconstruction']
d_reconstruct_params['triplet_output'] = True



anatomy_encoder_params = copy.copy(unet_config_kits.params)
anatomy_encoder_params['out_channels'] = params['num_mask_channels']
pathology_encoder_params = copy.copy(unet_config_kits.params)
pathology_encoder_params['out_channels'] = params['out_pathology_channels']


def get():
    shp = params['input_shape']
    ratio = params['image_downsample']
    shp = (int(np.round(shp[0] / ratio)), int(np.round(shp[1] / ratio)), shp[2])

    params['input_shape'] = shp
    params.update({'epochs':unet_config_kits.params['epochs']})
    d_mask_anato_params['input_shape'] = (shp[:-1]) + (loader.KitsLoader().num_anato_masks,)
    d_mask_patho_params['input_shape'] = (shp[:-1]) + (loader.KitsLoader().num_patho_masks,)
    d_reconstruct_params['input_shape'] = shp

    params.update({'anatomy_encoder_params': anatomy_encoder_params,
                   'pathology_encoder_params':pathology_encoder_params,
                   'd_mask_anato_params': d_mask_anato_params,
                   'd_mask_patho_params': d_mask_patho_params,
                   'd_reconstruct_params': d_reconstruct_params})
    return params
