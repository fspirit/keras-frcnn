import train_frcnn

options = {
    'train_path': './train_data/frcnn_train.csv',
    'parser': 'simple',
    'num_rois': 32,
    'network': 'resnet50',
    'horizontal_flips': False,
    'vertical_flips': False,
    'rot_90': False,
    'num_epochs': 2000,
    'config_filename': 'config.pickle',
    'output_weight_path': './model_frcnn.hdf5'
}

train_frcnn.run(options)



