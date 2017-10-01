from itertools import chain
import imgaug as ia
from imgaug import augmenters as iaa

import numpy as np

import pandas as pd
import cv2
from os.path import isfile, join
from torch.utils.data import DataLoader
import os

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

def crop_frame(img_path, bboxes, images_dir):

    img = cv2.imread(os.path.join(images_dir, img_path))

    kp = [[ia.Keypoint(x=bbox['x0'], y=bbox['y0']), ia.Keypoint(x=bbox['x1'], y=bbox['y1'])] for bbox in
          bboxes]
    keypoints = ia.KeypointsOnImage(list(chain.from_iterable(kp)), shape=img.shape)

    seq = iaa.Sequential([iaa.Crop(px=(0, 0, 0, img.shape[1]/2), keep_size=False)])

    seq_det = seq.to_deterministic()

    image_aug = seq_det.augment_images([img])[0]
    keypoints_aug = seq_det.augment_keypoints([keypoints])[0]

    for i in range(len(keypoints_aug.keypoints)):
        bbox = bboxes[int(i / 2.0)]
        if i % 2 == 0:
            bbox['x0'] = keypoints_aug.keypoints[i].x
            bbox['y0'] = keypoints_aug.keypoints[i].y
        else:
            bbox['x1'] = keypoints_aug.keypoints[i].x
            bbox['y1'] = keypoints_aug.keypoints[i].y

    return image_aug, bboxes



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
    h = (y1 - y0)
    w = x1 - x0
    expected_width = int(1.13 * h)
    optimal_width = min(w, expected_width)
    # print expected_width - optimal_width
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

    print boxes.info()

    def is_valid(row):
        x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
        w = x1 - x0
        h = y1 - y0

        if h == 0 or w == 0:
            return False

        if (w / h) > 1.2 or (h / w) > 1.2:
            return False

        return True

    boxes = boxes[boxes.apply(is_valid, axis=1)]

    def transform_box(s):
        # print s[]
        return transform_bbox(s['x0'], s['y0'], s['x1'], s['y1'])[0]

    boxes['x0'] = boxes.apply(transform_box, axis=1)

    boxes.to_csv('/Users/fs/Downloads/object-detection-crowdai/upd_labels.csv', index=False)

def crop_dataset():

    from datasets import UdacityCrowdAIDataset, ImgaugTransformFunction

    seq = iaa.Sequential([iaa.Crop(px=(0, 0, 0, 960), keep_size=False)])
    seq_det = seq.to_deterministic()

    ds = UdacityCrowdAIDataset('/Users/fs/Downloads/object-detection-crowdai/upd_labels.csv',
                               '/Users/fs/Downloads/object-detection-crowdai',
                               transform=ImgaugTransformFunction(seq))

    dataloader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4)

    iter = dataloader.__iter__()

    for index in range(10):
        item = next(iter)

        # print item

        img_batch, boxes_batch = item['img'], item['boxes']

        img = img_batch[0].numpy()
        boxes = boxes_batch[0]

        # print boxes
        bboxes = [dict(x0=box[0], y0=box[1], x1=box[2], y1=box[3]) for box in boxes]
        for box in boxes:
            cv2.rectangle(img,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (255, 255, 255))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('./results/' + str(index) + '.jpg', img)

def crop_dataset_2(annotations_path, images_dir):

    boxes = pd.read_csv(annotations_path)

    # boxes = boxes.sample(n=10)

    print boxes.info()

    boxes = boxes[boxes['Label'] != 'Pedestrian']

    data = []
    for f, group in boxes.groupby('Frame'):
        bb = group[['x0', 'y0', 'x1', 'y1']].to_dict(orient='index')
        data.append((f, bb.values()))

    # data = data[:50]

    def is_valid(b):
        x0, y0, x1, y1 = b['x0'], b['y0'], b['x1'], b['y1']

        if x0 <= 0 and x1 <= 0:
            return False

        if x0 <= 0:
            x0 = 0
        if x1 <= 0:
            x1 = 0

        w = x1 - x0
        h = y1 - y0

        if h / w > 1.3 or w / h > 1.3:
            return False

        return True

    results = []
    dest_path='/Users/fs/Downloads/crowdai/results'
    dest_path_bboxes = '/Users/fs/Downloads/crowdai/results_with_bboxes'


    for index, (img_path, bboxes) in enumerate(data):

        img, bboxes = crop_frame(img_path, bboxes, images_dir)

        path = str(index) + '.jpg'

        img_with_bboxes = np.copy(img)
        valid_bbox_exists = False
        for bbox in bboxes:
            if is_valid(bbox):
                valid_bbox_exists = True

                if bbox['x0'] <= 0:
                    bbox['x0'] = 0

                if bbox['x1'] <= 0:
                    bbox['x1'] = 0

                cv2.rectangle(img_with_bboxes,
                              (bbox['x0'], bbox['y0']),
                              (bbox['x1'], bbox['y1']),
                              (255, 255, 255))

                results.append(dict(Frame=path, x0=bbox['x0'], y0=bbox['y0'], x1=bbox['x1'], y1=bbox['y1']))

        if valid_bbox_exists:
            cv2.imwrite(os.path.join(dest_path, path), img)
            cv2.imwrite(os.path.join(dest_path_bboxes, path), img_with_bboxes)

    df = pd.DataFrame(columns=['Frame', 'x0', 'y0', 'x1', 'y1'])
    df = df.append(results)
    df.to_csv('./crowdai_labels.csv', index=False)

def get_data_from_csv(csv_path):

    boxes = pd.read_csv(csv_path)

    # boxes = boxes.sample(n=10)

    print boxes.info()

    boxes = boxes[boxes['Label'] != 'Pedestrian']

    data = []
    for f, group in boxes.groupby('Frame'):
        bb = group[['x0', 'y0', 'x1', 'y1']].to_dict(orient='index')
        data.append((f, bb.values()))

    return data

def filter_annotations(csv_path='./crowdai_labels.csv'):

    boxes = pd.read_csv(csv_path)

    print boxes.info()

    filtered_images_dir = '/Users/fs/Downloads/crowdai/results_with_bboxes'
    files = [f for f in os.listdir(filtered_images_dir) if isfile(join(filtered_images_dir, f))]

    def is_valid(box):
        if box['Frame'] not in files:
            return False
        if box['x1'] >= 955:
            return False

        w = box['x1'] - box['x0']
        h = box['y1'] - box['y0']

        if box['x0'] < 4 and h > w:
            return False
        return True

    boxes = boxes[boxes.apply(is_valid, axis=1)]

    print boxes.info()

    boxes.to_csv('./crowdai_labels_2.csv', index=False)

def transform_frame(img, bboxes, seq):

    kp = [[ia.Keypoint(x=bbox['x0'], y=bbox['y0']), ia.Keypoint(x=bbox['x1'], y=bbox['y1'])] for bbox in
          bboxes]
    keypoints = ia.KeypointsOnImage(list(chain.from_iterable(kp)), shape=img.shape)

    seq_det = seq.to_deterministic()

    image_aug = seq_det.augment_images([img])[0]
    keypoints_aug = seq_det.augment_keypoints([keypoints])[0]

    for i in range(len(keypoints_aug.keypoints)):
        bbox = bboxes[int(i / 2.0)]
        if i % 2 == 0:
            bbox['x0'] = keypoints_aug.keypoints[i].x
            bbox['y0'] = keypoints_aug.keypoints[i].y
        else:
            bbox['x1'] = keypoints_aug.keypoints[i].x
            bbox['y1'] = keypoints_aug.keypoints[i].y

    return np.copy(image_aug), bboxes

def aug_udacity_ds():
    boxes = pd.read_csv('./crowdai_labels_2.csv')

    images_dir = '/Users/fs/Downloads/crowdai/results'
    target_dir = '/Users/fs/Downloads/crowdai/results_after'

    data = []
    for f, group in boxes.groupby('Frame'):
        bb = group[['x0', 'y0', 'x1', 'y1']].to_dict(orient='index')
        data.append((f, bb.values()))

    tranforms = dict(
        blur=iaa.Sequential([iaa.Fliplr(0.5), iaa.GaussianBlur(sigma=(0.0, 10.0))]),
        dropout=iaa.Sequential([iaa.Fliplr(0.5), iaa.Dropout(p=(0.2, 0.3))]),
        multiply=iaa.Sequential([iaa.Fliplr(0.5), iaa.Multiply((0.3, 0.3))])
    )

    new_boxes = []

    for f, bboxes in data:
        img = cv2.imread(join(images_dir, f))

        for name, t in tranforms.iteritems():

            import copy
            img_aug, boxes_aug = transform_frame(img, copy.deepcopy(bboxes), t)

            # print bboxes
            # print boxes_aug

            new_filename = f.replace('.', '-') + '-' + name + '.jpg'

            for box in boxes_aug:

                # cv2.rectangle(img_aug,
                #               (box['x0'], box['y0']),
                #               (box['x1'], box['y1']),
                #               (255, 255, 255))

                new_boxes.append(dict(Frame=new_filename, x0=box['x0'], y0=box['y0'], x1=box['x1'], y1=box['y1']))

            cv2.imwrite(join(target_dir, new_filename), img_aug)

    df = pd.DataFrame(columns=['Frame', 'x0', 'y0', 'x1', 'y1'])
    df = df.append(new_boxes)
    df.to_csv('./new_crowdai.csv', index=False)

def merge_boxes_files_for_udacity_ds():
    boxes_1 = pd.read_csv('./crowdai_labels_2.csv')
    boxes_2 = pd.read_csv('./new_crowdai.csv')

    print boxes_1.info()
    print boxes_2.info()

    boxes_final = pd.concat([boxes_1, boxes_2], axis=0)

    print boxes_final.info()

    boxes_final.to_csv('./crowdai_boxes_final.csv', index=False)

def move_to_different_folders():
    filtered_images_dir = '/Users/fs/Downloads/crowdai/results_after'
    files = [f for f in os.listdir(filtered_images_dir) if isfile(join(filtered_images_dir, f))]

    dropout_path = '/Users/fs/Downloads/crowdai/img_dropout'
    blur_path = '/Users/fs/Downloads/crowdai/img_blur'
    multiply_path = '/Users/fs/Downloads/crowdai/img_multiply'
    original_path = '/Users/fs/Downloads/crowdai/img_original'

    import shutil

    for f in files:
        if 'dropout' in f:
            shutil.copyfile(join(filtered_images_dir, f), join(dropout_path, f))
        elif 'blur' in f:
            shutil.copyfile(join(filtered_images_dir, f), join(blur_path, f))
        elif 'multiply' in f:
            shutil.copyfile(join(filtered_images_dir, f), join(multiply_path, f))
        else:
            shutil.copyfile(join(filtered_images_dir, f), join(original_path, f))



if __name__ == '__main__':
    # update_boxes()
    # crop_dataset()
    # crop_dataset_2('/Users/fs/Downloads/object-detection-crowdai/upd_labels.csv',
    #                '/Users/fs/Downloads/object-detection-crowdai')
    # filter_annotations()
    # aug_udacity_ds()
    # merge_boxes_files_for_udacity_ds()
    move_to_different_folders()


