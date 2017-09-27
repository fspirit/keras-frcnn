from itertools import chain
import imgaug as ia
from imgaug import augmenters as iaa

import numpy as np

import pandas as pd
import cv2

images_path = '/data/img'
boxes_path = '/data/boxes.csv'

def calc_avg_height_to_width_ratio(boxes_path):
    with open(boxes_path, 'r') as f:
        print('Parsing annotation files')

        height_to_width_ratios = []
        for line in f:
            line_split = line.strip().split(',')
            (image_filename, x0, y0, x1, y1, label, confidence) = line_split
            print x0, y0, x1, y1
            if image_filename == 'image_filename':
                continue
            if float(x1) - float(x0) <= 0 or float(y1) - float(y0) <= 0:
                continue
            height_to_width_ratios.append((float(x1) - float(x0)) / (float(y1) - float(y0)))

    return np.mean(height_to_width_ratios)

def read_annotations():
    frames = {}
    with open(boxes_path, 'r') as f:
        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(',')
            (xmin, ymin, xmax, ymax, frame, label, preview) = line_split
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

def draw_image(img):
    x0, y0, x1, y1 = 785, 533, 905, 644

    (x0, y0, x1, y1) = transform_bbox(x0, y0, x1, y1)
    img = cv2.imread('/Users/fs/Downloads/object-detection-crowdai/1479498371963069978.jpg')
    cv2.rectangle(img,
                  (x0, y0),
                  (x1, y1),
                  (255, 255, 255))

    cv2.imwrite('./test.jpg', img)


def transform_bbox(x0, y0, x1, y1):
    expected_width = int(1.13 * (y1 - y0))
    optimal_width = min(x1 - x0, expected_width)
    print expected_width - optimal_width
    return x1 - optimal_width, y0, x1, y1

def test_box_update():
    boxes = pd.read_csv('/Users/fs/Downloads/object-detection-crowdai/labels.csv')

    sampled_boxes = boxes.sample(n=20)
    print sampled_boxes.info()
    def test_box_update_on_sample(sample):
        print type(sample)
        (x0, y0, x1, y1) = sample['xmin'], sample['xmax'], sample['ymin'], sample['ymax']

        img = cv2.imread('/Users/fs/Downloads/object-detection-crowdai/' + sample['Frame'])
        cv2.rectangle(img,
                      (x0, y0),
                      (x1, y1),
                      (255, 255, 255))

        (x0, y0, x1, y1) = transform_bbox(x0, y0, x1, y1)
        cv2.rectangle(img,
                      (x0, y0),
                      (x1, y1),
                      (255, 255, 0))
        cv2.imwrite('./results_imgs/' + sample['Frame'], img)
    sampled_boxes.apply(lambda sample: test_box_update_on_sample(sample), axis=1)


def update_boxes():
    boxes = pd.read_csv('/Users/fs/Downloads/object-detection-crowdai/labels.csv')

    boxes = boxes.rename(columns={'xmin': 'x0', 'xmax': 'y0', 'ymin': 'x1', 'ymax': 'y1'})

    prev =  boxes['x0']

    def transform_box(s):
        # print s[]
        return transform_bbox(s['x0'], s['y0'], s['x1'], s['y1'])[0]

    boxes['x0'] = boxes.apply(transform_box, axis=1)

    print (boxes['x0'] - prev).mean()

    boxes.to_csv('/Users/fs/Downloads/object-detection-crowdai/upd_labels.csv', index=False)

if __name__ == '__main__':
    # test
    # frames = { '/Users/fs/Documents/Code/keras-frcnn/train_data/img/frame_817c47b8-22c4-438a-8dc6-0e3f67f299ee_00000-1280_720.jpg':
    #                dict(bboxes=[{'x1': 602, 'y1': 270, 'x2': 727, 'y2': 421}]) }
    # result = crop(frames)
    # (img, img_data_aug) = result[0]
    #
    # cv2.rectangle(img,
    #               (img_data_aug['bboxes'][0]['x1'], img_data_aug['bboxes'][0]['y1']),
    #               (img_data_aug['bboxes'][0]['x2'], img_data_aug['bboxes'][0]['y2']),
    #               (255, 255, 255))
    #
    # cv2.imwrite('./test_crop.jpg', img)

    update_boxes()
