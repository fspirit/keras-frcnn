from __future__ import division

import cv2
import numpy as np
import sys
import pickle
import time
import pandas as pd
import os

from torch.utils.data import DataLoader

from keras_frcnn import config, resnet, vgg, roi_helpers

from keras import backend as K
from keras.layers import Input
from keras.models import Model

import datasets

import logging
logging.basicConfig(filename='test.log', level=logging.DEBUG)

def log(msg):
	print(msg)
	logging.info(msg)

sys.setrecursionlimit(40000)

def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


def find_objects(C, F, R, bbox_threshold, class_mapping, model_classifier):
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0] // C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // C.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier.predict([F, ROIs])

        for ii in range(P_cls.shape[1]):

            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass

            bboxes[cls_name].append(
                [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))
    return bboxes, probs


def construct_models(C, class_mapping, options):
    if C.network == 'resnet50':
        nn = resnet.Resnet50()
    elif C.network == 'vgg':
        nn = vgg.VGG16()

    C.num_rois = int(options['num_rois'])

    if C.network == 'resnet50':
        num_features = 1024
    elif C.network == 'vgg':
        num_features = 512

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
        input_shape_features = (num_features, None, None)
    else:
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_features)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)
    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    return model_classifier, model_rpn


def read_class_mapping_from_config(class_mapping):
    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)
    reversed_class_mapping = {v: k for k, v in class_mapping.items()}
    # print(reversed_class_mapping)
    class_to_color = {reversed_class_mapping[v]: np.random.randint(0, 255, 3) for v in reversed_class_mapping}
    return reversed_class_mapping, class_to_color

def load_config(options):
    config_output_filename = options['config_filename']
    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)
    return C


def draw_box(all_dets, class_to_color, img, key, prob, bbox):
    (real_x1, real_y1, real_x2, real_y2) = bbox

    cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                  (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 2)

    textLabel = '{}: {}'.format(key, int(100 * prob))

    all_dets.append((key, 100 * prob))

    (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
    textOrg = (real_x1, real_y1 - 0)
    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                  (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                  (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
    cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

def save(boxes, options):
    result = pd.DataFrame(columns=['image_filename', 'x0', 'y0', 'x1', 'y1', 'label', 'confidence'])

    if len(boxes) != 0:
        result = result.append(boxes)
        if 'cut_path' in options and options['cut_path'] is True:
            result['image_filename'] = result['image_filename'].apply(lambda f: os.path.basename(f))
        result.to_csv(options['bboxes_output'], index=False)
    else:
        print "No boxes found!"

def run_test(options, dataset):

    C = config.Config()

    class_mapping, class_to_color = read_class_mapping_from_config({'car': 0})

    model_classifier, model_rpn = construct_models(C, class_mapping, options)

    bbox_threshold = 0.8

    detected_bboxes = []

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    start_time = time.time()

    for index, item in enumerate(dataloader):

        img_batch, filepath_batch = item['img'],  item['image_path']

        img = img_batch[0].numpy()
        filepath = filepath_batch[0]

        # print img_batch.shape
        print str(index) + ' ' + filepath

        X, ratio = format_img_size(img, C)

        X = np.transpose(X, (2, 0, 1))
        X = np.expand_dims(X, axis=0)

        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        class_bboxes, class_probs = find_objects(C, F, R, bbox_threshold, class_mapping, model_classifier)

        all_dets = []

        for class_name in class_bboxes:
            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(np.array(class_bboxes[class_name]),
                                                                        np.array(class_probs[class_name]),
                                                                        overlap_thresh=0.5)
            print("{0} boxes found".format(len(new_boxes)))

            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]

                bbox = get_real_coordinates(ratio, x1, y1, x2, y2)

                draw_box(all_dets, class_to_color, img, class_name, new_probs[jk], bbox)

                (real_x0, real_y0, real_x1, real_y1) = bbox
                detected_bboxes.append({'image_filename': filepath, 'x0': real_x0, 'y0': real_y0,
                                        'x1': real_x1, 'y1': real_y1, 'label': 'car', 'confidence': new_probs[jk]})
        if index % 1000 == 0:
            save(detected_bboxes, options)
            time_elapsed = time.time() - start_time
            log('Saving test results after ' + str(index) +'. Time elapsed for last 1000 = ' + str(time_elapsed))
            start_time = time.time()


if __name__ == "__main__":

    test_images_path = '/Users/fs/Documents/Code/keras-frcnn/test_data'

    dataset = datasets.Nexar2TestDataset(test_images_path)

    validation_dt_path = './submission_2.csv'

    options = {
        'num_rois': 32,
        'network': 'resnet50',
        'config_filename': './config.pickle',
        'bboxes_output': validation_dt_path,
        'cut_path': True
    }

    run_test(options, dataset)
