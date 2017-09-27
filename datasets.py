import copy
import gzip
import os
from itertools import chain

import cv2
import imgaug as ia
import numpy as np
import pandas as pd
from torch.utils import data

from imgaug import augmenters as iaa


seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Crop(percent=(0, 0.2)),  # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 0.5))),
    # Strengthen or weaken the contrast in each image.
    # iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.Sometimes(0.2,
        iaa.OneOf([
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            iaa.Emboss(alpha=(0.0, 1.0)),
    ])),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
], random_order=True)  # apply augmenters in random order


class ImgaugTransformFunction(object):
    def __init__(self, seq):
        self.seq = seq

    def __call__(self, img, bboxes):
        kp = [[ia.Keypoint(x=bbox['x0'], y=bbox['y0']), ia.Keypoint(x=bbox['x1'], y=bbox['y1'])] for bbox in
              bboxes]
        keypoints = ia.KeypointsOnImage(list(chain.from_iterable(kp)), shape=img.shape)

        seq_det = seq.to_deterministic()

        img_aug = seq_det.augment_images([img])[0]
        keypoints_aug = seq_det.augment_keypoints([keypoints])[0]

        bboxes_aug = copy.deepcopy(bboxes)

        # convert boxes back
        for i in range(len(keypoints_aug.keypoints)):
            bbox = bboxes_aug[int(i / 2.0)]
            if i % 2 == 0:
                bbox['x0'] = keypoints_aug.keypoints[i].x
                bbox['y0'] = keypoints_aug.keypoints[i].y
            else:
                bbox['x1'] = keypoints_aug.keypoints[i].x
                bbox['y1'] = keypoints_aug.keypoints[i].y

        return (img_aug, bboxes_aug)


class DatasetBase(data.Dataset):

    @staticmethod
    def _normalize(x):
        x = x[:, :, (2, 1, 0)]
        x = x.astype(np.float32)

        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68

        # x /= 255.
        # x -= 0.5
        # x *= 2.
        return x


class Nexar2DatasetBase(DatasetBase):

    def load_external_val_filenames(self, val_filenames_list_path, images_dir):

        if val_filenames_list_path.endswith('.gz'):
            val_filenames = [l.rstrip('\n') for l in gzip.open(val_filenames_list_path, 'r').readlines()]
        else:
            val_filenames = [l.rstrip('\n') for l in open(val_filenames_list_path, 'r').readlines()]
        return [os.path.join(images_dir, f) for f in val_filenames]

    def load_boxes(self, boxes_path, images_dir):
        boxes = pd.read_csv(boxes_path)
        boxes['image_filename'] = boxes['image_filename'].apply(lambda f: os.path.join(images_dir, f))
        boxes[['x0', 'y0', 'x1', 'y1']] = boxes[['x0', 'y0', 'x1', 'y1']].astype(int)

        return boxes

    def load_data(self, annotations_path, images_dir, val_filenames_list_path):
        val_filenames = self.load_external_val_filenames(val_filenames_list_path, images_dir)
        boxes = self.load_boxes(annotations_path, images_dir)

        self.selected_boxes = self.select_boxes(boxes, val_filenames)

        self.data = []
        for f, group in self.selected_boxes.groupby('image_filename'):
            bb = group[['x0', 'y0', 'x1', 'y1']].to_dict(orient='index')
            self.data.append((f, bb.values()))

    def __len__(self):
        return len(self.data)


class Nexar2TrainDataset(Nexar2DatasetBase):

    def select_boxes(self, boxes, validation_filenames):
        return boxes[~boxes.image_filename.isin(validation_filenames)]

    def __init__(self, annotations_path, images_dir, val_filenames_list_path, transform=None):
        self.transform = transform

        self.load_data(annotations_path, images_dir, val_filenames_list_path)

    def __getitem__(self, index):
        (image_path, bboxes) = self.data[index]

        img = cv2.imread(image_path)

        if self.transform is not None:
            (img, bboxes) = self.transform(img, bboxes)

        img = self._normalize(img)

        bboxes = [[b['x0'], b['y0'], b['x1'], b['y1']] for b in bboxes]
        bboxes = np.array(bboxes)

        return dict(img=img, boxes=bboxes)


class Nexar2ValidationDataset(Nexar2DatasetBase):

    def select_boxes(self, boxes, validation_filenames):
        return boxes[boxes.image_filename.isin(validation_filenames)]

    def __init__(self, annotations_path, images_dir, val_filenames_list_path):
        self.load_data(annotations_path, images_dir, val_filenames_list_path)

    def __getitem__(self, index):
        (image_path, boxes) = self.data[index]

        img = cv2.imread(image_path)
        img = self._normalize(img)

        boxes = [[b['x0'], b['y0'], b['x1'], b['y1']] for b in boxes]
        boxes = np.array(boxes)

        return dict(img=img, boxes=boxes, image_path=image_path)

    def generate_gt_file(self, file_path):
        self.selected_boxes.to_csv(file_path, index=False)

from os import listdir
from os.path import isfile, join

class Nexar2TestDataset(Nexar2DatasetBase):
    def __init__(self, images_dir):
        self.files = [join(images_dir, f) for f in listdir(images_dir) if isfile(join(images_dir, f))]

    def __getitem__(self, index):
        image_path = self.files[index]

        img = cv2.imread(image_path)
        img = self._normalize(img)

        return dict(img=img, image_path=image_path)

    def __len__(self):
        return len(self.files)
