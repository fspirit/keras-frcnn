import train_frcnn

options = {
    'train_path': '/data/nexar/train/faster_rcnn_imput.csv',
    'parser': 'simple',
    'num_rois': 32,
    'network': 'resnet50',
    'num_epochs': 2000,
    'config_filename': 'config.pickle',
    'output_weight_path': './model_frcnn.hdf5'
}

train_frcnn.run(options)



