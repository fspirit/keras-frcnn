from itertools import chain
import imgaug as ia
from imgaug import augmenters as iaa

import pandas as pd
import cv2

images_path = '/data/img'
boxes_path = '/data/boxes.csv'

def read_annotations():
    frames = {}
    with open(boxes_path, 'r') as f:
        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(',')
            (xmin, ymin, xmax, ymax, frame, label) = line_split
            if label == 'Pedestrian':
                continue
            if frame not in frames:
                frames[frame] = dict(filename=frame, bboxes=[])
            frame['bboxes'].append(dict(x1=xmin, y1=ymin, x2=xmax, y2=ymax))

    return frames

def crop(frames):

    cropped_frames = []
    for img_path, img_data in frames.iteritems():

        img = cv2.imread(img_path)

        print img.shape

        kp = [[ia.Keypoint(x=bbox['x1'], y=bbox['y1']), ia.Keypoint(x=bbox['x2'], y=bbox['y2'])] for bbox in
              img_data['bboxes']]
        keypoints = ia.KeypointsOnImage(list(chain.from_iterable(kp)), shape=img.shape)

        seq = iaa.Sequential([iaa.Crop(px=(0, 0, 0, img.shape[1]/2), keep_size=False)])

        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_images([img])[0]
        keypoints_aug = seq_det.augment_keypoints([keypoints])[0]

        for i in range(len(keypoints_aug.keypoints)):
            bbox = img_data['bboxes'][int(i / 2.0)]
            if i % 2 == 0:
                bbox['x1'] = keypoints_aug.keypoints[i].x
                bbox['y1'] = keypoints_aug.keypoints[i].y
            else:
                bbox['x2'] = keypoints_aug.keypoints[i].x
                bbox['y2'] = keypoints_aug.keypoints[i].y
        cropped_frames.append((image_aug, img_data))

    return cropped_frames


if __name__ == '__main__':
    # test
    frames = { '/Users/fs/Documents/Code/keras-frcnn/train_data/img/frame_817c47b8-22c4-438a-8dc6-0e3f67f299ee_00000-1280_720.jpg':
                   dict(bboxes=[{'x1': 602, 'y1': 270, 'x2': 727, 'y2': 421}]) }
    result = crop(frames)
    (img, img_data_aug) = result[0]

    cv2.rectangle(img,
                  (img_data_aug['bboxes'][0]['x1'], img_data_aug['bboxes'][0]['y1']),
                  (img_data_aug['bboxes'][0]['x2'], img_data_aug['bboxes'][0]['y2']),
                  (255, 255, 255))

    cv2.imwrite('./test_crop.jpg', img)
