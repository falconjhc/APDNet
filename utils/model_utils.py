import h5py
MODEL_PATH = '/home/hjiang2/Harric/Codes/anatomy_modality_decomposition_imp/experiment_sdnet_cmr_segopt23_losstype_agis_split0/G_supervised_trainer'


print("读取模型中...")
with h5py.File(MODEL_PATH, 'r') as f:

    basic_keys = list(f.keys())
    for k in basic_keys:
        current_sub = list(f[k])
        for var in current_sub:
            print(f[k][var].name + ': ' )
            print(list(f[k][var].values()))