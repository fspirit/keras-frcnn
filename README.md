# Faster RCNN for Nexar2

# Prerequisites

Trained on
- Ubuntu Linux 16
- GPUs
- CUDA drivers + CUDA Toolkit 8.0
- cuDNN
- Conda
- Conda env file:

```nexar2.yml
name: nexar2
channels:
    - https://conda.anaconda.org/menpo
    - conda-forge
    - soumith
dependencies:
    - python==2.7.13
    - numpy
    - matplotlib
    - jupyter
    - opencv3    
    - scikit-learn
    - scikit-image
    - scipy
    - h5py
    - pandas
    - keras
    - tensorflow-gpu
    - six
    - pytorch
    - torchvision
    - cuda80
    - pip:
        - imgaug
```

## Train

```buildoutcfg
python train_frcnn.py
```

## Validate
```buildoutcfg
python validate.py
```

## Generate submission file

Modify `test_image_path` in test_frcnn.py, then run

```buildoutcfg
python test_frcnn.py
```
